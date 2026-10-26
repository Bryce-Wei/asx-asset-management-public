"""检查滚动回测的信息截止、调仓时点、净值计算和失败处理。"""

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from config import SETTINGS
from asset_management import backtest, macro, portfolio
from asset_management.backtest import (
    compute_signals, performance_metrics, point_in_time_inputs, prepare_news,
    rebalance_schedule, run_backtest, simulate,
)

TICKERS = ["AAA", "BBB", "CCC"]
SIGNAL = pd.Timestamp("2020-06-30")
# 缩短窗口，使三年合成数据足以覆盖优化、情绪和宏观三个分量。
TEST_SETTINGS = {
    **SETTINGS,
    "backtest_lookback_days": 120,
    "backtest_macro_window_months": 18,
    "sentiment_recent_articles": 2,
}


def synthetic_inputs(seed=0):
    """生成三只股票的日价格、正值月度指标和带 ISO 时间戳的新闻。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", "2020-12-31")
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, (len(dates), len(TICKERS))), axis=0)),
        index=dates, columns=TICKERS,
    )
    months = pd.date_range("2016-01-31", "2020-12-31", freq="ME")
    features = list(macro.DEFAULT_SETTINGS["macro_features"])
    indicators = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0.002, 0.01, (len(months), len(features))), axis=0),
        index=months, columns=features,
    )
    rows = []
    for day in dates[::7]:
        for ticker in TICKERS:
            rows.append({"Equity": ticker, "Source": "SYN",
                         "Date/Time": f"{day:%Y-%m-%d} {int(rng.integers(9, 18)):02d}:00",
                         "Headline": str(rng.choice(["good", "bad", "flat"]))})
    # 信号日收盘前后各一条，用于检查 16:00 截止边界。
    rows.append({"Equity": "AAA", "Source": "SYN", "Date/Time": "2020-06-30 15:00", "Headline": "good"})
    rows.append({"Equity": "AAA", "Source": "SYN", "Date/Time": "2020-06-30 18:00", "Headline": "bad"})
    return {"prices": prices, "news": pd.DataFrame(rows), "indicators": indicators}


def poison_future(inputs, signal, settings):
    """把信号日之后（及尚未发布）的数据改成极端值，信号若泄漏未来必然改变。"""
    cfg = {**backtest.DEFAULT_SETTINGS, **settings}
    rng = np.random.default_rng(99)
    prices = inputs["prices"].copy()
    future = prices.index > signal
    prices.loc[future] *= rng.uniform(0.2, 5.0, (future.sum(), prices.shape[1]))
    indicators = inputs["indicators"].copy()
    released = (signal.to_period("M") - cfg["backtest_macro_release_lag_months"]).end_time
    indicators.loc[indicators.index > released] *= 7.0
    extra = pd.DataFrame([
        {"Equity": ticker, "Source": "LEAK", "Date/Time": stamp, "Headline": "good"}
        for ticker in TICKERS
        for stamp in ("2020-06-30 16:01", "2020-07-01 09:00", "2020-08-15 12:00")
        for _ in range(10)
    ])
    return {"prices": prices, "news": pd.concat([inputs["news"], extra], ignore_index=True),
            "indicators": indicators}


@contextmanager
def fixed_vader():
    """固定标题分值，测试不依赖本地 VADER 词典。"""
    scores = {
        "good": {"pos": 1.0, "neg": 0.0, "neu": 0.0, "compound": 0.8},
        "bad": {"pos": 0.0, "neg": 1.0, "neu": 0.0, "compound": -0.8},
        "flat": {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0},
    }
    with patch("nltk.sentiment.vader.SentimentIntensityAnalyzer") as analyzer:
        analyzer.return_value.polarity_scores.side_effect = scores.__getitem__
        yield


def signals_at(inputs, signal, settings=TEST_SETTINGS):
    """按回测流程准备新闻并计算单期信号。"""
    data = {**inputs, "news": prepare_news(inputs["news"], settings)}
    return compute_signals(point_in_time_inputs(data, signal, settings), settings)


class PointInTimeTests(unittest.TestCase):
    """信号只能依赖信号日当时已可获得的价格、新闻和宏观指标。"""

    def test_future_data_cannot_change_signals(self):
        """篡改信号日之后的全部数据后，本期三个分量和五个版本的权重必须完全不变。

        同时确认篡改在更晚的信号日确实会改变结果，证明测试并非空转。
        """
        original = synthetic_inputs()
        poisoned = poison_future(original, SIGNAL, TEST_SETTINGS)
        with fixed_vader():
            clean = signals_at(original, SIGNAL)
            leaked = signals_at(poisoned, SIGNAL)
            later = pd.Timestamp("2020-09-30")
            clean_later = signals_at(original, later)
            leaked_later = signals_at(poisoned, later)
        self.assertEqual(clean["status"], {"base": "ok", "sentiment": "ok", "macro": "ok"})
        self.assertEqual(clean["status"], leaked["status"])
        assert_frame_equal(clean["components"], leaked["components"])
        self.assertEqual(set(clean["targets"]), {"equal_weight", *backtest.VARIANTS})
        for name, weights in clean["targets"].items():
            assert_series_equal(weights, leaked["targets"][name])
        self.assertFalse(clean_later["components"].equals(leaked_later["components"]))

    def test_news_cutoff_and_macro_release_lag(self):
        """16:00 之前的新闻进入当期，之后的不进入；指标只到信号月减发布滞后。"""
        data = {**synthetic_inputs(), "news": prepare_news(synthetic_inputs()["news"], TEST_SETTINGS)}
        pit = point_in_time_inputs(data, SIGNAL, TEST_SETTINGS)
        self.assertEqual(pit["news"]["timestamp"].max(), pd.Timestamp("2020-06-30 15:00"))
        self.assertEqual(pit["prices"].index.max(), SIGNAL)
        # 默认滞后 2 个月：6 月底只能看到 4 月及以前的指标。
        self.assertEqual(pit["indicators"].index.max(), pd.Timestamp("2020-04-30"))
        # 新闻按时间由新到旧排列，情绪模块的“近期”窗口才对应真正的最新标题。
        self.assertTrue(pit["news"]["timestamp"].is_monotonic_decreasing)

    def test_news_timestamps_do_not_swap_day_and_month(self):
        """ISO 日期不能被按日在前误读；澳洲格式和带时区时间按规则换算。"""
        iso = pd.DataFrame({"Date/Time": ["2020-03-04 10:00", "2020-03-05 09:30"]})
        self.assertEqual(prepare_news(iso, {})["timestamp"].min(), pd.Timestamp("2020-03-04 10:00"))
        australian = pd.DataFrame({"Date/Time": ["04/03/2020 10:00", "13/03/2020 09:30"]})
        self.assertEqual(prepare_news(australian, {})["timestamp"].min(), pd.Timestamp("2020-03-04 10:00"))
        # 2020 年 3 月悉尼为夏令时 UTC+11。
        aware = pd.DataFrame({"Date/Time": ["2020-03-04T00:30:00+00:00"]})
        self.assertEqual(prepare_news(aware, {})["timestamp"].iloc[0], pd.Timestamp("2020-03-04 11:30"))
        with self.assertRaises(ValueError):
            prepare_news(pd.DataFrame({"Date/Time": ["yesterday"]}), {})


class ScheduleAndSimulationTests(unittest.TestCase):
    """用可手算的日期和价格验证调仓时点与净值。"""

    def test_schedule_uses_month_end_signals_and_lagged_execution(self):
        """2020 年 1 月有 23 个工作日；未结束的 4 月不生成信号。"""
        index = pd.bdate_range("2020-01-01", "2020-04-15")
        schedule = rebalance_schedule(index, {"backtest_lookback_days": 22, "backtest_execution_lag_days": 1})
        self.assertEqual(list(schedule["signal_date"]), list(pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31"])))
        self.assertEqual(list(schedule["execution_date"]), list(pd.to_datetime(["2020-02-03", "2020-03-02", "2020-04-01"])))
        # 1 月底只有 22 个收益，窗口要求 23 个时 1 月不能作为首个信号日。
        later = rebalance_schedule(index, {"backtest_lookback_days": 23})
        self.assertEqual(later["signal_date"].iloc[0], pd.Timestamp("2020-02-28"))

    def test_simulation_matches_hand_calculation(self):
        """成交日当天的上涨不属于新持仓；两次调仓之间权重漂移，成本按换手扣除。"""
        dates = pd.bdate_range("2020-01-06", periods=5)
        prices = pd.DataFrame({"A": [100, 110, 121, 121, 133.1], "B": [100, 100, 90, 99, 99]}, index=dates)
        half = pd.Series([0.5, 0.5], index=["A", "B"])
        nav, log = simulate(prices, {dates[1]: half, dates[3]: half}, cost_bps=10, start_date=dates[1])
        a, b = 0.5 * 0.999, 0.5 * 0.999
        expected = [0.999]
        a, b = a * 1.1, b * 0.9
        expected.append(a + b)
        a, b = a * 1.0, b * 1.1
        value = a + b
        turnover = abs(0.5 - a / value) + abs(0.5 - b / value)
        value *= 1 - turnover * 0.001
        expected.append(value)
        expected.append(value * (0.5 * 1.1 + 0.5))
        np.testing.assert_allclose(nav.to_numpy(), expected)
        np.testing.assert_allclose(log["turnover"], [1.0, turnover])
        self.assertEqual(list(log["initial"]), [True, False])
        # 没有任何目标权重时资金保持为现金，净值恒为 1。
        cash, empty = simulate(prices, {}, cost_bps=10, start_date=dates[1])
        np.testing.assert_allclose(cash.to_numpy(), 1.0)
        self.assertTrue(empty.empty)

    def test_metrics_on_known_series(self):
        """峰值 1.1 跌到 0.99 的回撤为 -10%；总收益以初始资金 1 为基准。"""
        nav = pd.DataFrame(
            {"equal_weight": [1.0, 1.05, 1.0, 1.1], "base": [1.0, 1.1, 0.99, 1.2]},
            index=pd.to_datetime(["2020-01-01", "2020-07-01", "2021-01-01", "2022-01-01"]),
        )
        rebalances = pd.DataFrame(columns=["variant", "status", "turnover", "cost", "initial"])
        metrics = performance_metrics(nav, rebalances, {})
        self.assertAlmostEqual(metrics.loc["base", "max_drawdown"], 0.99 / 1.1 - 1)
        self.assertAlmostEqual(metrics.loc["base", "total_return"], 0.2)
        self.assertIn("information_ratio", metrics.columns)

    def test_sharpe_subtracts_daily_risk_free_rate(self):
        """常数利率序列与常数设置的夏普比率相同；变动的利率按日期逐日扣除。"""
        dates = pd.bdate_range("2020-01-01", periods=6)
        nav = pd.DataFrame({"equal_weight": [1.0, 1.01, 1.0, 1.02, 1.03, 1.02]}, index=dates)
        rebalances = pd.DataFrame(columns=["variant", "status", "turnover", "cost", "initial"])
        scalar = performance_metrics(nav, rebalances, {"backtest_risk_free_rate": 0.02})
        series = performance_metrics(nav, rebalances, {}, pd.Series(0.02, index=dates))
        self.assertAlmostEqual(scalar.loc["equal_weight", "sharpe"], series.loc["equal_weight", "sharpe"])
        rates = pd.Series([0.0, 0.0, 0.0, 0.10, 0.10, 0.10], index=dates)
        daily = nav["equal_weight"].pct_change().iloc[1:]
        excess = daily - ((1 + rates.iloc[1:]) ** (1 / 252) - 1)
        expected = excess.mean() / excess.std(ddof=1) * np.sqrt(252)
        actual = performance_metrics(nav, rebalances, {}, rates).loc["equal_weight", "sharpe"]
        self.assertAlmostEqual(actual, expected)


class FailureHandlingTests(unittest.TestCase):
    """单期优化失败时沿用原持仓，而不是中断回测或使用失败的权重。"""

    def test_failed_optimizer_holds_previous_weights(self):
        """第二期优化失败：依赖优化器的版本记为沿用，等权基准照常调仓。"""
        real = portfolio.analyze_portfolios
        calls = {"count": 0}

        def flaky(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("synthetic optimizer failure")
            return real(*args, **kwargs)

        with fixed_vader(), patch.object(backtest.portfolio, "analyze_portfolios", side_effect=flaky):
            result = run_backtest(synthetic_inputs(), TEST_SETTINGS)
        rebalances = result["tables"]["backtest_rebalances"].reset_index()
        second = rebalances.loc[rebalances["signal_date"] == rebalances["signal_date"].unique()[1]]
        status = second.set_index("variant")["status"]
        self.assertEqual(status["equal_weight"], "rebalanced")
        for name in backtest.VARIANTS:
            self.assertEqual(status[name], "held")
        self.assertEqual(result["summary"]["base_failures"], 1)
        self.assertTrue(np.isfinite(result["tables"]["backtest_nav"].to_numpy()).all())


class MarketBenchmarkTests(unittest.TestCase):
    """大盘基准在首个成交日买入并持有，各版本另报告相对大盘的指标。"""

    def test_market_is_bought_once_and_held(self):
        """大盘净值 = 扣除 10 个基点建仓成本后随价格变化，之后不再调仓。"""
        inputs = synthetic_inputs()
        dates = inputs["prices"].index
        market = pd.DataFrame({"STW": np.linspace(100.0, 130.0, len(dates))}, index=dates)
        with fixed_vader():
            result = run_backtest(inputs, TEST_SETTINGS, market)
        nav = result["tables"]["backtest_nav"]
        start = nav.index[0]
        expected = 0.999 * market["STW"].loc[start:] / market["STW"].loc[start]
        np.testing.assert_allclose(nav[backtest.MARKET].to_numpy(), expected.to_numpy())
        for column in ("active_return_vs_asx200", "tracking_error_vs_asx200", "information_ratio_vs_asx200"):
            self.assertTrue(np.isfinite(result["metrics"].loc["full", column]))
        rebalances = result["tables"]["backtest_rebalances"].reset_index()
        self.assertEqual(int((rebalances["variant"] == backtest.MARKET).sum()), 1)
        self.assertEqual(result["summary"]["market_benchmark"], backtest.MARKET)

    def test_risk_free_changes_apply_from_effective_date(self):
        """每个净值日期使用当日已生效的利率；利率记录晚于首个成交日时报错。"""
        inputs = synthetic_inputs()
        changes = pd.Series([0.015, 0.0025], index=pd.to_datetime(["2015-01-01", "2020-03-20"]))
        with fixed_vader():
            result = run_backtest(inputs, TEST_SETTINGS, risk_free=changes)
        nav = result["tables"]["backtest_nav"]
        expected = changes.reindex(changes.index.union(nav.index)).ffill().reindex(nav.index).mean()
        self.assertAlmostEqual(result["summary"]["risk_free_rate_mean"], expected)
        with self.assertRaises(ValueError):
            run_backtest(inputs, TEST_SETTINGS, risk_free=pd.Series([0.01], index=[nav.index[-1]]))

    def test_market_must_cover_backtest_period(self):
        """大盘数据提前结束时直接报错，而不是把最后价格一直延用下去。"""
        inputs = synthetic_inputs()
        dates = inputs["prices"].index
        market = pd.DataFrame({"STW": 100.0}, index=dates[: len(dates) // 2])
        with self.assertRaises(ValueError):
            run_backtest(inputs, TEST_SETTINGS, market)


if __name__ == "__main__":
    unittest.main()

"""检查复权价格的下载整理与读取规则，不访问网络。"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from asset_management.load_data import PRICE_FILE, load_cash_rate, load_price_table

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_yahoo_prices.py"
_spec = importlib.util.spec_from_file_location("fetch_yahoo_prices", SCRIPT)
fetch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch)


def yahoo_frame(dates, adj, close, dividends=None, splits=None):
    """按 yfinance.download(group_by="column") 的结构构造 (字段, 代码) 两层列。"""
    tickers = list(adj)
    zero = {ticker: [0.0] * len(dates) for ticker in tickers}
    fields = {
        "Adj Close": adj, "Close": close,
        "Dividends": dividends or zero, "Stock Splits": splits or zero,
    }
    columns = pd.MultiIndex.from_tuples(
        [(field, f"{ticker}.AX") for field in fields for ticker in tickers], names=["Price", "Ticker"]
    )
    values = np.column_stack([fields[field][ticker] for field in fields for ticker in tickers])
    return pd.DataFrame(values, index=pd.DatetimeIndex(pd.to_datetime(dates), name="Date"), columns=columns)


class FetchTests(unittest.TestCase):
    """build_outputs 只做整理与校验，用构造的 yfinance 输出检查每条规则。"""

    dates = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]

    def test_uses_adjusted_close_in_requested_order_and_records_events(self):
        """价格取 Adj Close 而非 Close；分红、拆股与超阈值涨跌写入元数据。"""
        frame = yahoo_frame(
            self.dates,
            adj={"BBB": [10.0, 10.5, 10.4, 10.6], "AAA": [20.0, 20.2, 26.0, 26.1]},
            close={"BBB": [11.0, 11.5, 11.4, 11.6], "AAA": [20.0, 20.2, 26.0, 26.1]},
            dividends={"BBB": [0.0, 0.3, 0.0, 0.0], "AAA": [0.0] * 4},
            splits={"BBB": [0.0] * 4, "AAA": [0.0, 0.0, 0.0, 2.0]},
        )
        prices, meta = fetch.build_outputs(frame, ("AAA", "BBB"), "2020-01-02", "2020-01-07", threshold=0.15)
        self.assertEqual(list(prices.columns), ["AAA", "BBB"])
        self.assertEqual(prices.index.name, "date")
        np.testing.assert_allclose(prices["BBB"], [10.0, 10.5, 10.4, 10.6])
        self.assertEqual(meta["tickers"]["BBB"]["dividends"], [{"date": "2020-01-03", "value": 0.3}])
        self.assertEqual(meta["tickers"]["AAA"]["splits"], [{"date": "2020-01-07", "value": 2.0}])
        self.assertEqual([(m["date"], m["ticker"]) for m in meta["large_moves"]], [("2020-01-06", "AAA")])
        self.assertEqual(meta["requested"]["tickers"], ["AAA.AX", "BBB.AX"])

    def test_timezone_aware_index_becomes_sydney_dates(self):
        """带时区的索引按悉尼时区取日期，避免 UTC 前一天的日期错位。"""
        frame = yahoo_frame(self.dates, adj={"AAA": [1.0, 1.1, 1.2, 1.3]}, close={"AAA": [1.0, 1.1, 1.2, 1.3]})
        frame.index = pd.DatetimeIndex(pd.to_datetime(self.dates)).tz_localize("Australia/Sydney").tz_convert("UTC")
        prices, _ = fetch.build_outputs(frame, ("AAA",), "2020-01-02", "2020-01-07")
        self.assertEqual(list(prices.index), list(pd.to_datetime(self.dates)))

    def test_missing_or_late_ticker_raises_instead_of_shortening_sample(self):
        """缺少某只股票，或其首个报价远晚于起始日，都应报错。"""
        frame = yahoo_frame(self.dates, adj={"AAA": [1.0, 1.1, 1.2, 1.3]}, close={"AAA": [1.0, 1.1, 1.2, 1.3]})
        with self.assertRaisesRegex(ValueError, "missing tickers: BBB"):
            fetch.build_outputs(frame, ("AAA", "BBB"), "2020-01-02", "2020-01-07")
        with self.assertRaisesRegex(ValueError, "near 2019-12-01"):
            fetch.build_outputs(frame, ("AAA",), "2019-12-01", "2020-01-07")

    def test_written_csv_round_trips_through_loader_with_matching_hash(self):
        """写出的 CSV 能被完整模式的读取函数读回，JSON 记录的哈希与文件一致。"""
        frame = yahoo_frame(
            self.dates,
            adj={"AAA": [1.0, 1.1, np.nan, 1.3], "BBB": [5.0, 5.1, 5.2, 5.3]},
            close={"AAA": [1.0, 1.1, np.nan, 1.3], "BBB": [5.0, 5.1, 5.2, 5.3]},
        )
        prices, meta = fetch.build_outputs(frame, ("AAA", "BBB"), "2020-01-02", "2020-01-07")
        with tempfile.TemporaryDirectory() as folder:
            csv_path, meta_path = fetch.write_outputs(prices, meta, Path(folder) / PRICE_FILE)
            loaded = load_price_table(csv_path)
            saved = json.loads(meta_path.read_text(encoding="utf-8"))
            import hashlib
            self.assertEqual(saved["csv_sha256"], hashlib.sha256(csv_path.read_bytes()).hexdigest())
            self.assertNotIn(b"\r\n", csv_path.read_bytes())
        self.assertEqual(saved["tickers"]["AAA"]["missing"], 1)
        pd.testing.assert_frame_equal(loaded, prices, check_freq=False)


class PriceTableTests(unittest.TestCase):
    """load_price_table 的格式检查与原演示读取保持一致。"""

    def write(self, folder, text):
        path = Path(folder) / "prices.csv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_rejects_missing_date_column_and_nonpositive_prices(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "date column"):
                load_price_table(self.write(folder, "day,A\n2020-01-01,1\n"))
            with self.assertRaisesRegex(ValueError, "strictly positive"):
                load_price_table(self.write(folder, "date,A\n2020-01-01,1\n2020-01-02,0\n"))
            with self.assertRaises(FileNotFoundError):
                load_price_table(Path(folder) / "absent.csv")

    def test_cash_rate_percent_becomes_sorted_decimal(self):
        """现金利率按百分数存储，读取后为小数并按生效日升序。"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rates.csv"
            path.write_text("effective_date,cash_rate_target\n2020-03-20,0.25\n2019-06-05,1.25\n", encoding="utf-8")
            rates = load_cash_rate(path)
            path.write_text("date,rate\n2020-01-01,1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cash_rate(path)
        self.assertEqual(list(rates.index), list(pd.to_datetime(["2019-06-05", "2020-03-20"])))
        np.testing.assert_allclose(rates.to_numpy(), [0.0125, 0.0025])

    def test_keeps_missing_values_and_sorts_dates(self):
        with tempfile.TemporaryDirectory() as folder:
            frame = load_price_table(self.write(folder, "date,A,B\n2020-01-02,2,\n2020-01-01,1,3\n"))
        self.assertEqual(list(frame.index), list(pd.to_datetime(["2020-01-01", "2020-01-02"])))
        self.assertTrue(np.isnan(frame.loc["2020-01-02", "B"]))


if __name__ == "__main__":
    unittest.main()

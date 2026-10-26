"""用 yfinance 下载 10 只股票含分红的复权收盘价，写入 data/。

输出两份文件：

- ``ASX200top10_yahoo.csv``：第一列为 date，其余为股票代码，取 Yahoo 的
  Adj Close（按现金分红和拆股向前调整），收益率因此为总收益口径；
- ``ASX200top10_yahoo.json``：下载时间、请求参数、yfinance 版本、各股票的
  起止日期、分红与拆股记录、单日涨跌超过阈值的日期，以及 CSV 的 SHA-256。

Yahoo 会随新的分红回溯调整整段历史，因此不同时间下载的价格水平可能不同，
但相邻日期的收益率不受影响。分拆等公司行动的处理因数据源而异，单日涨跌
超过阈值的日期记入 large_moves，需人工核对后再用于分析。仓库附带的数据
已在与早期 Bloomberg 价格重叠的 2015-03-30 至 2020-04-02 逐日核对：
2015 年 BHP 分拆 South32、2018 年 WES 分拆 Coles 两边处理一致，唯一超过
阈值的 2017-01-23 BXB 下跌 15.8% 为真实行情；两边收益率的差异集中在
除息日，等于当期股息率。

用法（项目根目录）::

    python -m pip install -r scripts/requirements.txt
    python scripts/fetch_yahoo_prices.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "ASX200top10_yahoo.csv"
# 与原彭博工作簿相同的 10 只股票；区间（含两端）在彭博价格的 2020-04-02 之后
# 延长到 2020-07-31，与新闻数据的覆盖期对齐。
TICKERS = ("BHP", "CSL", "RIO", "CBA", "WOW", "WES", "TLS", "AMC", "BXB", "FPH")
START = "2015-03-30"
END = "2020-07-31"
SUFFIX = ".AX"
# 单日复权收益绝对值超过此值时记入 large_moves，提示核对分拆等公司行动。
JUMP_THRESHOLD = 0.15
# Yahoo 限流（HTTP 429）时的重试次数与首次等待秒数，之后每次加倍。
RETRIES = 3
RETRY_WAIT_SECONDS = 20


def _failed(frame, symbols):
    """返回下载结果中缺列或 Adj Close 全为空的 Yahoo 代码。"""
    if frame is None or frame.empty or "Adj Close" not in frame.columns.get_level_values(0):
        return list(symbols)
    adjusted = frame["Adj Close"]
    return [s for s in symbols if s not in adjusted.columns or adjusted[s].isna().all()]


def download(tickers, start, end, retries=RETRIES, wait=RETRY_WAIT_SECONDS):
    """调用 yfinance 下载日线，返回 (价格字段, Yahoo 代码) 两层列的原始表格。

    yfinance 的 end 不含当日，因此传入 end 的下一天。auto_adjust=False
    同时保留 Close 与 Adj Close；actions=True 附带 Dividends 和 Stock Splits。
    Yahoo 限流时个别股票会返回空列：这些股票逐只重试，每次等待时间加倍。
    """
    import yfinance as yf

    symbols = [f"{ticker}{SUFFIX}" for ticker in tickers]
    stop = (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()

    def fetch(batch):
        return yf.download(
            batch, start=start, end=stop, interval="1d", auto_adjust=False, actions=True,
            group_by="column", progress=False, threads=False, multi_level_index=True,
        )

    frame = fetch(symbols)
    failed = _failed(frame, symbols)
    for attempt in range(retries):
        if not failed:
            break
        delay = wait * 2 ** attempt
        print(f"Retrying {', '.join(failed)} in {delay:.0f}s (attempt {attempt + 1}/{retries})", file=sys.stderr)
        time.sleep(delay)
        parts = []
        for symbol in failed:
            part = fetch([symbol])
            if not _failed(part, [symbol]):
                parts.append(part)
        if parts:
            keep = frame.loc[:, ~frame.columns.get_level_values(1).isin(failed)] if frame is not None else None
            frame = pd.concat([keep, *parts], axis=1).sort_index(axis=1)
        failed = _failed(frame, symbols)
    if failed:
        raise RuntimeError(f"Yahoo returned no prices for {', '.join(failed)}; wait a few minutes and rerun.")
    return frame


def _field(frame, name, tickers):
    """取出一个价格字段，列名去掉 .AX 后缀并按 tickers 顺序排列。"""
    if name not in frame.columns.get_level_values(0):
        raise ValueError(f"yfinance output has no {name!r} field.")
    table = frame[name].copy()
    table.columns = [str(column).removesuffix(SUFFIX) for column in table.columns]
    missing = [ticker for ticker in tickers if ticker not in table.columns]
    if missing:
        raise ValueError(f"yfinance output is missing tickers: {', '.join(missing)}")
    index = pd.DatetimeIndex(table.index)
    # 带时区时先换算到交易所时区，再只保留日期。
    if index.tz is not None:
        index = index.tz_convert("Australia/Sydney").tz_localize(None)
    table.index = pd.DatetimeIndex(index.normalize(), name="date")
    return table.loc[:, list(tickers)].astype(float)


def _events(table, ticker):
    values = table[ticker]
    return [{"date": str(day.date()), "value": float(value)} for day, value in values[values > 0].items()]


def build_outputs(frame, tickers=TICKERS, start=START, end=END, threshold=JUMP_THRESHOLD):
    """把 yfinance 原始表格整理为复权价格表和元数据，不访问网络。

    删除所有股票都无报价的日期，个别缺失保留为空值。任一股票没有
    报价，或首个报价晚于 start 之后 7 天，都直接报错，避免样本被静默截短。
    """
    prices = _field(frame, "Adj Close", tickers)
    close = _field(frame, "Close", tickers)
    dividends = _field(frame, "Dividends", tickers).fillna(0.0)
    splits = _field(frame, "Stock Splits", tickers).fillna(0.0)
    prices = prices.loc[prices.notna().any(axis=1)]
    if prices.empty:
        raise ValueError("No price observations were downloaded.")
    first = prices.apply(lambda column: column.first_valid_index())
    late = [t for t in tickers if pd.isna(first[t]) or first[t] > pd.Timestamp(start) + pd.Timedelta(days=7)]
    if late:
        raise ValueError(f"Tickers without prices near {start}: {', '.join(late)}")
    if (prices <= 0).any().any():
        raise ValueError("Downloaded prices must be strictly positive.")
    moves = prices.pct_change(fill_method=None)
    large = [
        {"date": str(day.date()), "ticker": ticker, "return": round(float(value), 6)}
        for (day, ticker), value in moves.stack().items() if abs(value) > threshold
    ]
    metadata = {
        "source": "Yahoo Finance via yfinance",
        "field": "Adj Close (adjusted for cash dividends and splits)",
        "requested": {"tickers": [f"{t}{SUFFIX}" for t in tickers], "start": start, "end_inclusive": end},
        "rows": int(len(prices)),
        "date_range": [str(prices.index.min().date()), str(prices.index.max().date())],
        "tickers": {
            ticker: {
                "first": str(prices[ticker].first_valid_index().date()),
                "last": str(prices[ticker].last_valid_index().date()),
                "missing": int(prices[ticker].isna().sum()),
                "close_rows": int(close[ticker].notna().sum()),
                "dividends": _events(dividends, ticker),
                "splits": _events(splits, ticker),
            }
            for ticker in tickers
        },
        "large_moves_threshold": threshold,
        "large_moves": large,
    }
    return prices, metadata


def write_outputs(prices, metadata, output):
    """写出 CSV（LF 换行、6 位小数）与同名 JSON，返回两个路径。"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = prices.to_csv(float_format="%.6f", lineterminator="\n")
    output.write_text(text, encoding="utf-8", newline="")
    meta_path = output.with_suffix(".json")
    meta = {**metadata, "csv_file": output.name, "csv_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="")
    return output, meta_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download dividend-adjusted ASX prices from Yahoo Finance.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV 输出路径，JSON 写在同名位置")
    parser.add_argument("--start", default=START, help="首个交易日（含）")
    parser.add_argument("--end", default=END, help="最后交易日（含）")
    args = parser.parse_args(argv)
    try:
        import yfinance
    except ImportError:
        parser.exit(1, "yfinance is not installed: python -m pip install -r scripts/requirements.txt\n")
    frame = download(TICKERS, args.start, args.end)
    prices, metadata = build_outputs(frame, TICKERS, args.start, args.end)
    metadata = {
        **metadata,
        "yfinance_version": yfinance.__version__,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    csv_path, meta_path = write_outputs(prices, metadata, args.output)
    print(f"Saved {len(prices)} rows x {len(prices.columns)} tickers to {csv_path}")
    print(f"Metadata: {meta_path}")
    if metadata["large_moves"]:
        print(f"Review {len(metadata['large_moves'])} daily moves above {JUMP_THRESHOLD:.0%} in the metadata (large_moves).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

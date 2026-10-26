"""下载澳洲储备银行（RBA）现金利率目标的历史记录，写入 data/。

输出两份文件：

- ``RBA_cash_rate.csv``：effective_date（生效日）与 cash_rate_target（年利率，
  百分数），每次议息会议一行（包括维持不变的决定），按日期升序；
- ``RBA_cash_rate.json``：来源网址、下载时间、行数与 CSV 的 SHA-256。

回测把它当作无风险利率：每个交易日取当日已生效的现金利率目标，用于计算
夏普比率。只使用 Python 标准库，从 RBA 现金利率页面的表格中解析数据。

用法（项目根目录）::

    python scripts/fetch_rba_cash_rate.py
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "RBA_cash_rate.csv"
SOURCE_URL = "https://www.rba.gov.au/statistics/cash-rate/"


def parse_table(page):
    """从页面表格提取 (生效日, 利率百分数) 列表，按日期升序，不访问网络。"""
    rows = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", page, flags=re.S):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", cell)).strip()
                 for cell in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, flags=re.S)]
        if len(cells) < 3:
            continue
        try:
            day = datetime.strptime(cells[0], "%d %b %Y").date()
            rate = float(cells[2].replace("to", "").split()[-1])
        except ValueError:
            continue  # 表头、说明行或区间写法以外的格式
        rows.append((day, rate))
    if not rows:
        raise ValueError("No cash rate rows found; the RBA page layout may have changed.")
    rows = sorted(set(rows))
    if len({day for day, _ in rows}) != len(rows):
        raise ValueError("Conflicting cash rates for the same effective date.")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download the RBA cash rate target history.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV 输出路径，JSON 写在同名位置")
    args = parser.parse_args(argv)
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        page = response.read().decode("utf-8", errors="replace")
    rows = parse_table(page)
    text = "effective_date,cash_rate_target\n" + "".join(f"{day.isoformat()},{rate:.2f}\n" for day, rate in rows)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8", newline="")
    meta = {
        "source": "Reserve Bank of Australia, cash rate target",
        "url": SOURCE_URL,
        "unit": "percent per annum",
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": len(rows),
        "date_range": [rows[0][0].isoformat(), rows[-1][0].isoformat()],
        "csv_file": output.name,
        "csv_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    output.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8", newline="")
    print(f"Saved {len(rows)} cash rate changes to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

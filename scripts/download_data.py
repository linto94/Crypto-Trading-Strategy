"""Download Binance USDT-M futures 1m klines + funding from data.binance.vision to data/parquet.

    python scripts/download_data.py --start 2021-01 --end 2026-08
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cts.data import download_funding, download_klines  # noqa: E402

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
                   "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]


def one(sym: str, a) -> str:
    download_klines(sym, a.start, a.end, a.data_dir)
    download_funding(sym, a.start, a.end, a.data_dir)
    return sym


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    ap.add_argument("--start", default="2021-01")
    ap.add_argument("--end", default="2026-08")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    with ThreadPoolExecutor(a.threads) as ex:
        for sym in ex.map(lambda s: one(s, a), a.symbols):
            print(f"done {sym}")


if __name__ == "__main__":
    main()

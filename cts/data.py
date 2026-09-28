"""Data download (Binance public archive), loading and resampling.

All frames are indexed by bar OPEN time (UTC) and carry a ``close_time`` column
(= open time + bar length). Higher-timeframe values are only ever joined onto
lower-timeframe bars by ``close_time`` so a bar can never see an unfinished HTF bar.
"""
from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

BASE_URL = "https://data.binance.vision/data/futures/um"
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
OHLCV = ["open", "high", "low", "close", "volume", "taker_buy_volume"]


# --------------------------------------------------------------------------- download
def _fetch(url: str, retries: int = 4) -> bytes | None:
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries:
                raise
        time.sleep(delay)
        delay *= 2
    return None


def _read_zip_csv(blob: bytes, names: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    first = raw[:64].decode(errors="ignore")
    header = 0 if first[:1].isalpha() else None
    df = pd.read_csv(io.BytesIO(raw), header=header)
    if header is None:
        df.columns = names[: df.shape[1]]
    return df


def _months(start: str, end: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def download_klines(symbol: str, start: str, end: str, data_dir: str | Path = "data",
                    interval: str = "1m") -> Path:
    """Download monthly kline archives for ``symbol`` (inclusive month range) to parquet."""
    out = Path(data_dir) / "parquet" / f"{symbol}_{interval}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for m in _months(start, end):
        url = f"{BASE_URL}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{m}.zip"
        blob = _fetch(url)
        if blob is None:
            print(f"  {symbol} {m}: not available, skipped")
            continue
        df = _read_zip_csv(blob, KLINE_COLS)
        df.columns = KLINE_COLS[: df.shape[1]]
        frames.append(df[["open_time", *OHLCV]])
        print(f"  {symbol} {m}: {len(df)} rows")
    if not frames:
        raise RuntimeError(f"no kline data downloaded for {symbol}")
    df = pd.concat(frames, ignore_index=True)
    unit = "us" if df["open_time"].iloc[0] > 1e14 else "ms"
    df["open_time"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)
    df = df.drop_duplicates("open_time").set_index("open_time").sort_index().astype("float64")
    df.to_parquet(out)
    return out


def download_funding(symbol: str, start: str, end: str, data_dir: str | Path = "data") -> Path:
    out = Path(data_dir) / "parquet" / f"{symbol}_funding.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for m in _months(start, end):
        url = f"{BASE_URL}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{m}.zip"
        blob = _fetch(url)
        if blob is None:
            continue
        frames.append(_read_zip_csv(blob, ["calc_time", "funding_interval_hours", "last_funding_rate"]))
    if not frames:
        raise RuntimeError(f"no funding data downloaded for {symbol}")
    df = pd.concat(frames, ignore_index=True)
    unit = "us" if df["calc_time"].iloc[0] > 1e14 else "ms"
    s = pd.Series(df["last_funding_rate"].astype(float).values,
                  index=pd.to_datetime(df["calc_time"], unit=unit, utc=True).dt.floor("min"),
                  name="funding_rate")
    s = s[~s.index.duplicated()].sort_index()
    s.to_frame().to_parquet(out)
    return out


# --------------------------------------------------------------------------- load
def load_1m(symbol: str, data_dir: str | Path = "data") -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / "parquet" / f"{symbol}_1m.parquet")


def load_funding(symbol: str, data_dir: str | Path = "data") -> pd.Series | None:
    p = Path(data_dir) / "parquet" / f"{symbol}_funding.parquet"
    return pd.read_parquet(p)["funding_rate"] if p.exists() else None


def available_symbols(data_dir: str | Path = "data") -> list[str]:
    d = Path(data_dir) / "parquet"
    return sorted(p.name[:-11] for p in d.glob("*_1m.parquet")) if d.exists() else []


# --------------------------------------------------------------------------- resample
def resample(df1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1m OHLCV to ``rule`` bars (left-labelled, left-closed) and add close_time.

    Bars with incomplete 1m coverage are kept but flagged via ``n_1m``.
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum", "taker_buy_volume": "sum"}
    if rule.upper() in ("1W", "W"):
        out = df1m.resample("W-MON", label="left", closed="left").agg(agg)
        n = df1m["close"].resample("W-MON", label="left", closed="left").count()
        length = pd.Timedelta(days=7)
    else:
        out = df1m.resample(rule, label="left", closed="left").agg(agg)
        n = df1m["close"].resample(rule, label="left", closed="left").count()
        length = pd.Timedelta(rule)
    out["n_1m"] = n
    out = out.dropna(subset=["close"])
    out["close_time"] = out.index + length
    return out


def bar_minutes(rule: str) -> int:
    return int(pd.Timedelta(rule).total_seconds() // 60)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


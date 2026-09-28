"""Causal indicator primitives. Every value at row t uses only rows <= t."""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def wilder_atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    atr = true_range(h, l, c).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return atr


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0)


def efficiency_ratio(c: pd.Series, n: int) -> pd.Series:
    num = (c - c.shift(n)).abs()
    den = c.diff().abs().rolling(n).sum()
    return num / den.replace(0, np.nan)


def rolling_vwap(h, l, c, v, n: int) -> pd.Series:
    tp = (h + l + c) / 3.0
    return (tp * v).rolling(n).sum() / v.rolling(n).sum().replace(0, np.nan)


def rolling_pct_rank(s: pd.Series, n: int, min_periods: int | None = None) -> pd.Series:
    """Percentile rank (0-100) of the current value within the trailing n values."""
    return s.rolling(n, min_periods=min_periods or n // 2).rank(pct=True) * 100.0


def rvol_time_of_day(v: pd.Series, days: int = 20) -> pd.Series:
    """Volume / median volume of the same time-of-day slot over the previous ``days`` days."""
    slot = v.index.hour * 60 + v.index.minute
    med = v.groupby(slot).transform(
        lambda s: s.shift(1).rolling(days, min_periods=max(5, days // 2)).median())
    return v / med.replace(0, np.nan)


def pivots(h: pd.Series, l: pd.Series, k: int) -> pd.DataFrame:
    """Fractal pivots, reported on the CONFIRMATION bar (pivot bar + k).

    Pivot high at bar i: H_i > max(H_{i-k..i-1}) and H_i >= max(H_{i+1..i+k}).
    Returns columns ph, ph_time, pl, pl_time (NaN/NaT where no pivot is confirmed).
    """
    prior_max = h.shift(1).rolling(k).max()
    next_max = h[::-1].shift(1).rolling(k).max()[::-1]
    prior_min = l.shift(1).rolling(k).min()
    next_min = l[::-1].shift(1).rolling(k).min()[::-1]
    is_ph = (h > prior_max) & (h >= next_max)
    is_pl = (l < prior_min) & (l <= next_min)
    idx = pd.Series(h.index, index=h.index)
    out = pd.DataFrame(index=h.index)
    out["ph"] = h.where(is_ph).shift(k)
    out["ph_time"] = idx.where(is_ph).shift(k)
    out["pl"] = l.where(is_pl).shift(k)
    out["pl_time"] = idx.where(is_pl).shift(k)
    return out


def variance_ratio(r: pd.Series, q: int, n: int) -> pd.Series:
    """Lo-MacKinlay variance ratio on a trailing window of n one-bar log returns."""
    rq = r.rolling(q).sum()
    var1 = r.rolling(n).var()
    varq = rq.rolling(n - q + 1).var()
    return varq / (q * var1)

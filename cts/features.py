"""Multi-timeframe feature set and regime classifier (docs/strategy_research.md §1).

``build_features`` returns a :class:`Features` bundle whose ``m15`` frame has one row per
15m bar. Every column on a row is known at that bar's CLOSE: 15m features are causal and
1h/4h values are joined via ``merge_asof`` on close_time, so only closed HTF bars are used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind
from .data import resample

REGIMES = ["SHOCK", "TREND_UP", "TREND_DOWN", "RANGE", "TRANSITION"]


@dataclass
class RegimeParams:
    shock_volpct: float = 95.0
    shock_tr_atr: float = 3.5
    shock_ret_sigma: float = 5.0
    trend_er4h: float = 0.30
    trend_slope4h: float = 0.75
    range_er1h: float = 0.20
    range_slope1h: float = 0.5
    range_volpct_lo: float = 20.0
    range_volpct_hi: float = 85.0
    dead_volpct: float = 10.0


@dataclass
class Features:
    m15: pd.DataFrame
    h1: pd.DataFrame
    d1: pd.DataFrame
    w1: pd.DataFrame


def _asof_join(left: pd.DataFrame, right: pd.DataFrame, cols: list[str], suffix: str) -> pd.DataFrame:
    r = right[["close_time", *cols]].rename(columns={c: f"{c}{suffix}" for c in cols})
    r = r.rename(columns={"close_time": f"_ct{suffix}"}).sort_values(f"_ct{suffix}")
    merged = pd.merge_asof(left.reset_index().sort_values("close_time"), r,
                           left_on="close_time", right_on=f"_ct{suffix}", direction="backward")
    merged = merged.drop(columns=[f"_ct{suffix}"]).set_index(left.index.name or "open_time")
    return merged


def build_features(df1m: pd.DataFrame, rp: RegimeParams | None = None) -> Features:
    rp = rp or RegimeParams()
    df1m = df1m.copy()
    df1m.index.name = "open_time"

    m15 = resample(df1m, "15min")
    h1 = resample(df1m, "1h")
    h4 = resample(df1m, "4h")
    d1 = resample(df1m, "1D")
    w1 = resample(df1m, "1W")
    for f in (m15, h1, h4, d1, w1):
        f.index.name = "open_time"

    # ---- 15m
    o, h, l, c, v = (m15[k] for k in ("open", "high", "low", "close", "volume"))
    m15["tr"] = ind.true_range(h, l, c)
    m15["atr"] = ind.wilder_atr(h, l, c, 14)
    m15["ema50"] = ind.ema(c, 50)
    m15["rsi14"] = ind.rsi(c, 14)
    m15["vwap96"] = ind.rolling_vwap(h, l, c, v, 96)
    m15["sd96"] = (c - m15["vwap96"]).rolling(96).std()
    m15["z"] = (c - m15["vwap96"]) / m15["sd96"]
    m15["rvol"] = ind.rvol_time_of_day(v, 20)
    m15["delta"] = 2 * m15["taker_buy_volume"] - v
    m15["delta_ratio"] = m15["delta"] / v.replace(0, np.nan)
    m15["logret"] = np.log(c).diff()
    m15["sigma_r"] = m15["logret"].rolling(672, min_periods=336).std()
    m15["ret4"] = np.log(c / c.shift(4))
    rng = (h - l).replace(0, np.nan)
    m15["close_pos"] = (c - l) / rng  # 0 = closed at low, 1 = closed at high

    # ---- 1h
    h1["atr"] = ind.wilder_atr(h1["high"], h1["low"], h1["close"], 14)
    h1["ema20"] = ind.ema(h1["close"], 20)
    h1["ema50"] = ind.ema(h1["close"], 50)
    h1["er48"] = ind.efficiency_ratio(h1["close"], 48)
    h1["er24"] = ind.efficiency_ratio(h1["close"], 24)
    h1["slope"] = (h1["ema50"] - h1["ema50"].shift(12)) / h1["atr"]
    h1["volpct"] = ind.rolling_pct_rank(h1["atr"] / h1["close"], 2160, min_periods=720)
    h1["rsi14"] = ind.rsi(h1["close"], 14)
    piv = ind.pivots(h1["high"], h1["low"], 2)
    h1 = h1.join(piv)

    # ---- 4h
    h4["atr"] = ind.wilder_atr(h4["high"], h4["low"], h4["close"], 14)
    h4["ema50"] = ind.ema(h4["close"], 50)
    h4["slope"] = (h4["ema50"] - h4["ema50"].shift(5)) / h4["atr"]
    h4["er20"] = ind.efficiency_ratio(h4["close"], 20)

    # ---- 1h regime (uses the last CLOSED 4h bar)
    h1 = _asof_join(h1, h4, ["close", "atr", "ema50", "slope", "er20"], "_4h")
    trend_up = ((h1["er20_4h"] >= rp.trend_er4h) & (h1["slope_4h"] >= rp.trend_slope4h)
                & (h1["close_4h"] > h1["ema50_4h"]) & (h1["ema20"] > h1["ema50"]))
    trend_dn = ((h1["er20_4h"] >= rp.trend_er4h) & (h1["slope_4h"] <= -rp.trend_slope4h)
                & (h1["close_4h"] < h1["ema50_4h"]) & (h1["ema20"] < h1["ema50"]))
    ranging = ((h1["er48"] <= rp.range_er1h) & (h1["slope"].abs() <= rp.range_slope1h)
               & h1["volpct"].between(rp.range_volpct_lo, rp.range_volpct_hi))
    reg = np.select([trend_up, trend_dn, ranging], ["TREND_UP", "TREND_DOWN", "RANGE"], "TRANSITION")
    warm = h1[["er20_4h", "slope_4h", "er48", "slope", "volpct"]].notna().all(axis=1)
    h1["regime"] = np.where(warm, reg, "WARMUP")

    # ---- join 1h (incl. its 4h context) onto 15m
    m15 = _asof_join(m15, h1, ["atr", "ema20", "ema50", "er48", "er24", "slope", "volpct", "rsi14",
                               "regime", "slope_4h", "er20_4h", "atr_4h", "ema50_4h"], "_1h")
    m15 = m15.rename(columns={"slope_4h_1h": "slope_4h", "er20_4h_1h": "er20_4h",
                              "atr_4h_1h": "atr_4h", "ema50_4h_1h": "ema50_4h"})

    # ---- 15m-level SHOCK override + DEAD overlay
    tr_shock = (m15["tr"] >= rp.shock_tr_atr * m15["atr"].shift(1)).rolling(4, min_periods=1).max() > 0
    ret_shock = m15["ret4"].abs() >= rp.shock_ret_sigma * m15["sigma_r"] * 2.0
    shock = tr_shock | ret_shock | (m15["volpct_1h"] >= rp.shock_volpct)
    m15["regime"] = np.where(m15["regime_1h"].isin(["WARMUP"]) | m15["regime_1h"].isna(), "WARMUP",
                             np.where(shock, "SHOCK", m15["regime_1h"]))
    m15["dead"] = m15["volpct_1h"] < rp.dead_volpct
    return Features(m15=m15, h1=h1, d1=d1, w1=w1)

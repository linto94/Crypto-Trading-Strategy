"""Synthetic 1m market data with NO predictable structure, for validating the engine.

Log-price is a driftless martingale with clustered (stochastic) volatility, an intraday
volatility/volume cycle, rare jumps, and a taker-buy share that is correlated with the
SAME bar's return (as in real data) but carries no information about future returns.
Any strategy that shows a net edge on this data after costs indicates a look-ahead bug.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_1m(days: int = 365, seed: int = 0, start: str = "2023-01-02",
                 price: float = 30000.0, vol_per_min: float = 0.0006, substeps: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 1440
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")

    # stochastic log-volatility, half-life ~2 days, stationary sd 0.45
    phi = np.exp(-np.log(2) / (2 * 1440))
    shocks = rng.standard_normal(n) * 0.45 * np.sqrt(1 - phi ** 2)
    lv = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc = phi * acc + shocks[i]
        lv[i] = acc
    mod = (idx.hour * 60 + idx.minute).to_numpy() / 1440.0
    season = 1.0 + 0.35 * np.sin(2 * np.pi * (mod - 0.35))
    sigma = vol_per_min * np.exp(lv) * season

    steps = rng.standard_normal((n, substeps)) * (sigma / np.sqrt(substeps))[:, None]
    jumps = (rng.random(n) < 1 / (3 * 1440)) * rng.standard_normal(n) * sigma * 12
    steps[:, -1] += jumps
    # martingale correction so E[price] has no drift
    steps -= 0.5 * (sigma[:, None] ** 2) / substeps
    path = np.log(price) + np.cumsum(steps.ravel())
    path = path.reshape(n, substeps)
    close = path[:, -1]
    open_ = np.concatenate([[np.log(price)], close[:-1]])
    high = np.maximum(path.max(axis=1), open_)
    low = np.minimum(path.min(axis=1), open_)

    ret = close - open_
    vol = 50.0 * season * np.exp(0.6 * rng.standard_normal(n)) * (1 + 1.5 * (ret / sigma) ** 2)
    share = np.clip(0.5 + 0.25 * np.tanh(ret / sigma) + 0.05 * rng.standard_normal(n), 0.02, 0.98)
    return pd.DataFrame({
        "open": np.exp(open_), "high": np.exp(high), "low": np.exp(low), "close": np.exp(close),
        "volume": vol, "taker_buy_volume": vol * share,
    }, index=pd.Index(idx, name="open_time"))

"""Null / benchmark entry generators (docs/strategy_research.md §9.6 test 3)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .features import Features
from .sim import Target, TradeSpec


def random_entries(feat: Features, symbol: str, n_trades: int, b: float, stop_atr: float = 1.5,
                   time_stop_bars: int = 10_000, seed: int = 0) -> list[TradeSpec]:
    """Random bar, random side, stop = stop_atr * ATR15 from the next open, single target at b R."""
    m = feat.m15
    rng = np.random.default_rng(seed)
    ok = np.flatnonzero(m["atr"].notna().to_numpy())
    ok = ok[ok < len(m) - 1]
    bars = np.sort(rng.choice(ok, size=min(n_trades, len(ok)), replace=False))
    o, atr, idx = m["open"].to_numpy(), m["atr"].to_numpy(), m.index
    specs = []
    for i in bars:
        side = 1 if rng.random() < 0.5 else -1
        stop = o[i + 1] - side * stop_atr * atr[i]
        specs.append(TradeSpec(symbol, "random", side, idx[i], idx[i + 1], stop,
                               [Target(b, 1.0)], time_stop_bars))
    return specs


def matched_random(feat: Features, symbol: str, trades: pd.DataFrame,
                   exit_plan: Callable[[int, int], tuple[list[Target], int, bool]],
                   seed: int = 0, window_days: int = 45) -> list[TradeSpec]:
    """For each real trade: a random bar within +/- ``window_days`` of it, with the same regime and
    side, the same stop distance in ATR15 units and the same exit rules. Tests whether the ENTRY
    adds edge beyond exit geometry + regime + period.

    ``exit_plan(i, side)`` returns (targets, time_stop_bars, be_after_tp1) for signal bar i.
    """
    m = feat.m15
    rng = np.random.default_rng(seed)
    o, atr, idx = m["open"].to_numpy(), m["atr"].to_numpy(), m.index
    regime = m["regime"].to_numpy(object)
    valid = m["atr"].notna().to_numpy() & ~m["dead"].to_numpy(bool)
    valid[-1] = False
    pools = {r: np.flatnonzero(valid & (regime == r)) for r in np.unique(regime)}
    w = window_days * 96
    specs = []
    for t in trades.itertuples():
        pool = pools.get(t.regime)
        if pool is None or len(pool) == 0:
            continue
        i_real = idx.get_loc(t.signal_time)
        a, b = np.searchsorted(pool, i_real - w), np.searchsorted(pool, i_real + w)
        if b <= a:
            continue
        i = int(pool[rng.integers(a, b)])
        side = int(t.side)
        stop = o[i + 1] - side * t.risk_atr * atr[i]
        targets, ts, be = exit_plan(i, side)
        specs.append(TradeSpec(symbol, "matched_random", side, idx[i], idx[i + 1], stop,
                               targets, ts, be, weight=float(t.weight), meta={"regime": t.regime}))
    return specs

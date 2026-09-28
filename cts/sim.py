"""Trade simulator with 1-minute fill resolution (docs/strategy_research.md §9.4).

Rules:
* Market entry at the OPEN of ``entry_time`` (the bar after the signal bar), plus spread/slippage.
* Stops are stop-market orders: filled at the stop price minus 2x slippage, or at the bar OPEN if
  the bar gaps through the stop.
* Targets are resting limits: filled at the target price (maker fee) only if price trades THROUGH
  it (1m high > target for a long).
* If one 1m bar touches both the stop and a target, the stop is assumed to fill first.
* After TP1 with ``be_after_tp1`` the stop moves to breakeven-after-costs; if that new stop is
  inside the same 1m bar's range it is assumed hit in that bar (pessimistic).
* Remaining size is closed at market at the open of the bar where the time stop expires.
* Historical funding is charged/credited for any settlement while the position is open.

P&L is reported in R, where 1R = |entry fill - initial stop|.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import CostModel


@dataclass
class Target:
    r: float                   # distance from entry fill in R
    fraction: float            # fraction of the initial position closed here
    cap: float | None = None   # optional absolute price: target is the NEARER of r-price and cap


@dataclass
class TradeSpec:
    symbol: str
    strategy: str
    side: int                  # +1 long, -1 short
    signal_time: pd.Timestamp  # open time of the signal bar (decision at its close)
    entry_time: pd.Timestamp   # open time of the bar where the market entry is filled
    stop: float
    targets: list[Target]
    time_stop_bars: int
    be_after_tp1: bool = False
    bar_minutes: int = 15
    weight: float = 1.0
    shock: bool = False
    meta: dict = field(default_factory=dict)


def to_np_time(ts) -> np.datetime64:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return np.datetime64(ts.value, "ns")


class Simulator:
    def __init__(self, df1m: pd.DataFrame, costs: CostModel, funding: pd.Series | None = None):
        self.t = df1m.index.values.astype("datetime64[ns]")
        self.o = df1m["open"].to_numpy(float)
        self.h = df1m["high"].to_numpy(float)
        self.l = df1m["low"].to_numpy(float)
        self.c = df1m["close"].to_numpy(float)
        self.costs = costs
        if funding is not None and len(funding):
            self.ft = funding.index.values.astype("datetime64[ns]")
            self.fr = funding.to_numpy(float)
        else:
            self.ft = np.array([], dtype="datetime64[ns]")
            self.fr = np.array([], dtype=float)

    # ------------------------------------------------------------------
    def run(self, spec: TradeSpec) -> dict:
        s, cm = spec.side, self.costs
        et = to_np_time(spec.entry_time)
        n = len(self.t)
        i0 = int(np.searchsorted(self.t, et))
        base = {"symbol": spec.symbol, "strategy": spec.strategy, "side": s,
                "signal_time": spec.signal_time, "entry_time": spec.entry_time,
                "weight": spec.weight, **spec.meta}
        if i0 >= n or self.t[i0] - et > np.timedelta64(spec.bar_minutes, "m"):
            return {**base, "status": "rejected", "reason": "no_data"}

        raw_open = self.o[i0]
        entry = raw_open * (1 + s * cm.market_impact(spec.shock))
        risk = s * (entry - spec.stop)
        if risk <= 0:
            return {**base, "status": "rejected", "reason": "stop_through_entry"}
        risk_raw = s * (raw_open - spec.stop)

        tps, prev = [], entry
        for tg in spec.targets:
            p = entry + s * tg.r * risk
            if tg.cap is not None:
                p = min(p, tg.cap) if s > 0 else max(p, tg.cap)
            if s * (p - prev) < 0:
                p = prev
            tps.append(p)
            prev = p
        if s * (tps[0] - entry) <= 0:
            return {**base, "status": "rejected", "reason": "target_behind_entry"}

        end_t = et + np.timedelta64(spec.time_stop_bars * spec.bar_minutes, "m")
        i_end = int(np.searchsorted(self.t, end_t))

        pnl = -cm.taker() * entry          # per unit of initial size
        gross = 0.0
        remaining = 1.0
        cur_stop = spec.stop
        moved = False
        ti = 0
        tp1 = False
        fills: list[tuple[np.datetime64, float]] = []  # (time, remaining after fill)
        mfe = mae = 0.0
        reason = None
        exit_t = None

        def close_rest(price_fill, price_raw, fee, j):
            nonlocal pnl, gross, remaining
            pnl += remaining * (s * (price_fill - entry) - fee * price_fill)
            gross += remaining * s * (price_raw - raw_open)
            remaining = 0.0
            fills.append((self.t[j], 0.0))

        j = i0
        last = min(i_end, n)
        while j < last:
            oj, hj, lj = self.o[j], self.h[j], self.l[j]
            fav, adv = (hj, lj) if s > 0 else (lj, hj)
            # gap through stop at the open
            if s * (oj - cur_stop) <= 0:
                close_rest(oj * (1 - s * cm.stop_impact(spec.shock)), oj, cm.taker(), j)
                reason, exit_t = ("be_stop" if moved else "stop"), self.t[j]
                break
            stop_touch = (lj <= cur_stop) if s > 0 else (hj >= cur_stop)
            if stop_touch:
                mae = max(mae, s * (entry - adv))
                close_rest(cur_stop * (1 - s * cm.stop_impact(spec.shock)), cur_stop, cm.taker(), j)
                reason, exit_t = ("be_stop" if moved else "stop"), self.t[j]
                break
            hit_this_bar = False
            while ti < len(tps) and s * (fav - tps[ti]) > 0:
                f = min(spec.targets[ti].fraction, remaining)
                pnl += f * (s * (tps[ti] - entry) - cm.maker() * tps[ti])
                gross += f * s * (tps[ti] - raw_open)
                remaining -= f
                fills.append((self.t[j], remaining))
                if ti == 0:
                    tp1 = True
                    if spec.be_after_tp1:
                        cur_stop = entry * (1 + s * cm.round_trip())
                        moved = True
                hit_this_bar = True
                ti += 1
            mfe = max(mfe, s * (fav - entry))
            mae = max(mae, s * (entry - adv))
            if remaining <= 1e-12:
                reason, exit_t = "target", self.t[j]
                break
            if hit_this_bar and moved and ((lj <= cur_stop) if s > 0 else (hj >= cur_stop)):
                close_rest(cur_stop * (1 - s * cm.stop_impact(spec.shock)), cur_stop, cm.taker(), j)
                reason, exit_t = "be_stop", self.t[j]
                break
            j += 1
        else:
            if i_end < n:
                p = self.o[i_end]
                close_rest(p * (1 - s * cm.market_impact(spec.shock)), p, cm.taker(), i_end)
                reason, exit_t = "time", self.t[i_end]
            else:
                p = self.c[n - 1]
                close_rest(p * (1 - s * cm.market_impact(spec.shock)), p, cm.taker(), n - 1)
                reason, exit_t = "eod", self.t[n - 1]

        # funding: charged on the size open at each settlement in (entry, exit]
        if len(self.ft):
            k0 = int(np.searchsorted(self.ft, et, side="right"))
            k1 = int(np.searchsorted(self.ft, exit_t, side="right"))
            for k in range(k0, k1):
                size = 1.0
                for ft_, rem in fills:
                    if ft_ <= self.ft[k]:
                        size = rem
                pnl -= s * self.fr[k] * entry * size

        held = (exit_t - et) / np.timedelta64(spec.bar_minutes, "m")
        return {
            **base, "status": "filled", "exit_time": pd.Timestamp(exit_t, tz="UTC"),
            "entry": entry, "stop": spec.stop, "tp1_price": tps[0], "risk": risk,
            "risk_pct": risk / entry, "R": pnl / risk, "gross_R": gross / risk_raw,
            "reason": reason, "tp1_hit": tp1, "bars_held": float(held),
            "mfe_R": mfe / risk, "mae_R": mae / risk,
        }

    def run_many(self, specs: list[TradeSpec]) -> pd.DataFrame:
        return pd.DataFrame([self.run(sp) for sp in specs])

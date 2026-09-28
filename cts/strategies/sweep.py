"""Strategy A — Range-Extreme Liquidity Sweep Reversal (docs/strategy_research.md §2).

Signal generation walks the 15m bars in time order. The level map is updated at every 1h
close using only closed 1h / 1D / 1W bars; the sweep test on a 15m bar uses only that bar and
earlier ones; the risk/room checks use the NEXT bar's open, which is the moment the market
order would be sent. So every decision uses only information available when it is made.

Simplifications vs. the spec (documented in the report):
* A level is consumed when a signal passes all filters, even if the book-level constraints
  later drop the trade (the generator does not know about other symbols' positions).
* The macro-event blackout is not applied (no event calendar yet).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ..costs import CostModel
from ..features import Features
from ..sim import Target, TradeSpec


@dataclass
class SweepParams:
    pierce_min_atr: float = 0.10
    pierce_max_atr: float = 1.50
    reclaim_atr: float = 0.05
    max_window: int = 3
    close_pos_min: float = 0.60
    rvol_min: float = 1.5
    require_absorption: bool = True
    room_R: float = 2.0
    room_R_transition: float = 2.5
    stop_buffer_atr: float = 0.25
    risk_min_atr: float = 0.8
    risk_max_atr: float = 3.0
    cost_filter_mult: float = 8.0
    max_bar_range_atr: float = 3.5
    slope4h_block: float = 0.75
    allowed_regimes: tuple[str, ...] = ("RANGE", "TRANSITION")
    with_trend_in_trend: bool = False
    management: str = "fixed"          # "fixed" | "scale"
    tp_R: float = 1.0
    time_stop_fixed: int = 16
    tp1_R: float = 0.8
    tp2_R: float = 1.6
    time_stop_scale: int = 24
    level_merge_atr1h: float = 0.25
    pivot_lookback_h: int = 72
    pivot_min_age_h: int = 8
    transition_weight: float = 0.5

    def label(self) -> str:
        return (f"pen{self.pierce_max_atr}_rv{self.rvol_min}_{self.management}"
                f"{'' if self.management == 'scale' else self.tp_R}")


@dataclass
class _Level:
    lid: tuple
    price: float
    kind: str          # "S" support (low) | "R" resistance (high)
    born: pd.Timestamp
    is_pivot: bool
    alive: bool = True


@dataclass
class _Merged:
    price: float
    members: list = field(default_factory=list)


def _merge(levels: list[_Level], kind: str, tol: float) -> list[_Merged]:
    lv = sorted((x for x in levels if x.kind == kind), key=lambda x: x.price)
    out: list[_Merged] = []
    for x in lv:
        if out and x.price - out[-1].members[-1].price <= tol:
            out[-1].members.append(x)
        else:
            out.append(_Merged(price=x.price, members=[x]))
    for m in out:
        m.price = (min if kind == "S" else max)(x.price for x in m.members)
    return out


def exit_plan(p: SweepParams, side: int, vwap_i: float, sd_i: float) -> tuple[list[Target], int, bool]:
    """(targets, time_stop_bars, be_after_tp1) for a signal bar with the given VWAP96/SD96."""
    if p.management == "scale":
        cap = vwap_i + side * sd_i if not np.isnan(vwap_i) else None
        return [Target(p.tp1_R, 0.5), Target(p.tp2_R, 0.5, cap)], p.time_stop_scale, True
    return [Target(p.tp_R, 1.0)], p.time_stop_fixed, False


def with_management(specs: list[TradeSpec], p: SweepParams) -> list[TradeSpec]:
    """Re-plan the exits of already generated specs (entries are independent of management)."""
    out = []
    for sp in specs:
        targets, ts, be = exit_plan(p, sp.side, sp.meta["vwap96"], sp.meta["sd96"])
        out.append(replace(sp, targets=targets, time_stop_bars=ts, be_after_tp1=be))
    return out


def generate(feat: Features, symbol: str, p: SweepParams, costs: CostModel,
             funnel: dict | None = None) -> list[TradeSpec]:
    """Return trade specs; if ``funnel`` is given, count how many candidates each filter removed."""
    fn = funnel if funnel is not None else {}

    def drop(key):
        fn[key] = fn.get(key, 0) + 1

    m = feat.m15
    n = len(m)
    idx = m.index
    o = m["open"].to_numpy(float); h = m["high"].to_numpy(float)
    l = m["low"].to_numpy(float); c = m["close"].to_numpy(float)
    atr = m["atr"].to_numpy(float); rvol = m["rvol"].to_numpy(float)
    delta = m["delta"].to_numpy(float); cpos = m["close_pos"].to_numpy(float)
    vwap = m["vwap96"].to_numpy(float); sd = m["sd96"].to_numpy(float)
    slope4h = m["slope_4h"].to_numpy(float)
    regime = m["regime"].to_numpy(object); dead = m["dead"].to_numpy(bool)
    close_t = m["close_time"]

    h1 = feat.h1
    h1_by_close = {ct: row for ct, row in zip(h1["close_time"], h1.itertuples())}
    d1_by_close = dict(zip(feat.d1["close_time"], zip(feat.d1.index, feat.d1["high"], feat.d1["low"])))
    w1_by_close = dict(zip(feat.w1["close_time"], zip(feat.w1.index, feat.w1["high"], feat.w1["low"])))

    levels: dict[tuple, _Level] = {}
    supports: list[_Merged] = []
    resists: list[_Merged] = []
    episodes: dict[tuple, list[int]] = {}   # level id -> bar indices that pierced it
    rtc = costs.round_trip()
    specs: list[TradeSpec] = []
    lookback = pd.Timedelta(hours=p.pivot_lookback_h)
    min_age = pd.Timedelta(hours=p.pivot_min_age_h)

    for i in range(n - 1):
        ct = close_t.iat[i]
        # ------------------------------------------------ level map update at 1h close
        hrow = h1_by_close.get(ct)
        if hrow is not None:
            hc = hrow.close
            for lv in levels.values():
                if lv.alive and ((lv.kind == "S" and hc < lv.price) or (lv.kind == "R" and hc > lv.price)):
                    lv.alive = False
            if not np.isnan(hrow.ph):
                lid = ("PH", hrow.ph_time)
                levels[lid] = _Level(lid, hrow.ph, "R", hrow.ph_time, True)
            if not np.isnan(hrow.pl):
                lid = ("PL", hrow.pl_time)
                levels[lid] = _Level(lid, hrow.pl, "S", hrow.pl_time, True)
            drow = d1_by_close.get(ct)
            if drow is not None:
                for k in [k for k in levels if k[0] in ("PDH", "PDL")]:
                    del levels[k]
                levels[("PDH", drow[0])] = _Level(("PDH", drow[0]), drow[1], "R", ct, False)
                levels[("PDL", drow[0])] = _Level(("PDL", drow[0]), drow[2], "S", ct, False)
            wrow = w1_by_close.get(ct)
            if wrow is not None:
                for k in [k for k in levels if k[0] in ("PWH", "PWL")]:
                    del levels[k]
                levels[("PWH", wrow[0])] = _Level(("PWH", wrow[0]), wrow[1], "R", ct, False)
                levels[("PWL", wrow[0])] = _Level(("PWL", wrow[0]), wrow[2], "S", ct, False)
            for k in [k for k, lv in levels.items() if lv.is_pivot and ct - lv.born > lookback]:
                del levels[k]
            usable = [lv for lv in levels.values()
                      if lv.alive and (not lv.is_pivot or ct - lv.born >= min_age)]
            tol = p.level_merge_atr1h * hrow.atr if not np.isnan(hrow.atr) else 0.0
            supports = _merge(usable, "S", tol)
            resists = _merge(usable, "R", tol)

        a = atr[i]
        if np.isnan(a) or a <= 0 or regime[i] in ("WARMUP", "SHOCK") or dead[i]:
            continue

        # ------------------------------------------------ sweep test
        done = False
        for side, book in ((1, supports), (-1, resists)):
            if done:
                break
            reg = regime[i]
            reg_ok = reg in p.allowed_regimes or (
                p.with_trend_in_trend and reg == ("TREND_UP" if side > 0 else "TREND_DOWN"))
            if not reg_ok:
                continue
            if side > 0 and not slope4h[i] > -p.slope4h_block:
                continue
            if side < 0 and not slope4h[i] < p.slope4h_block:
                continue
            for lvl in book:
                L = lvl.price
                if not any(x.alive for x in lvl.members):
                    continue
                if side * (L - (l[i] if side > 0 else h[i])) >= p.pierce_min_atr * a:
                    for x in lvl.members:
                        episodes.setdefault(x.lid, []).append(i)
                recl = L + side * p.reclaim_atr * a
                if side * (c[i] - recl) <= 0:      # no reclaim on this bar
                    continue
                # window: bars after the last close beyond the reclaim threshold
                t0 = i
                while t0 - 1 >= 0 and side * (c[t0 - 1] - recl) <= 0 and i - t0 + 1 <= p.max_window:
                    t0 -= 1
                if i - t0 + 1 > p.max_window or t0 - 1 < 0:
                    continue
                ext = l[t0:i + 1].min() if side > 0 else h[t0:i + 1].max()
                pen = side * (L - ext) / a
                if pen < p.pierce_min_atr:
                    continue
                drop("0_candidates")
                if pen > p.pierce_max_atr:
                    drop("pierce_too_deep"); continue
                crit_pos = cpos[i] if side > 0 else 1 - cpos[i]
                if not crit_pos >= p.close_pos_min:
                    drop("close_position"); continue
                if not np.nanmax(rvol[t0:i + 1]) >= p.rvol_min:
                    drop("rvol"); continue
                if p.require_absorption and not side * delta[t0:i + 1].sum() < 0:
                    drop("absorption"); continue
                if (h[t0:i + 1] - l[t0:i + 1]).max() > p.max_bar_range_atr * a:
                    drop("news_bar"); continue
                # a pierce BEFORE this window on the same UTC day = level already swept today
                day = idx[t0].normalize()
                if any(e < t0 and idx[e].normalize() == day
                       for x in lvl.members for e in episodes.get(x.lid, [])):
                    drop("swept_earlier_today"); continue
                # ---------------------------------------- next-open checks
                entry = o[i + 1]
                stop = ext - side * p.stop_buffer_atr * a
                risk = side * (entry - stop)
                if not (p.risk_min_atr * a <= risk <= p.risk_max_atr * a):
                    drop("risk_atr_bounds"); continue
                if risk < p.cost_filter_mult * rtc * entry:
                    drop("cost_filter"); continue
                opp = [x.price for x in (resists if side > 0 else supports)
                       if side * (x.price - entry) > 0]
                room_req = p.room_R_transition if reg == "TRANSITION" else p.room_R
                if opp and min(side * (x - entry) for x in opp) < room_req * risk:
                    drop("room"); continue
                drop("zz_emitted")
                # ---------------------------------------- emit
                targets, ts, be = exit_plan(p, side, vwap[i], sd[i])
                specs.append(TradeSpec(
                    symbol=symbol, strategy="A_sweep", side=side, signal_time=idx[i],
                    entry_time=idx[i + 1], stop=stop, targets=targets, time_stop_bars=ts,
                    be_after_tp1=be, weight=p.transition_weight if reg == "TRANSITION" else 1.0,
                    meta={"regime": reg, "level_kinds": "+".join(sorted({x.lid[0] for x in lvl.members})),
                          "pen_atr": pen, "risk_atr": risk / a, "rvol_max": float(np.nanmax(rvol[t0:i + 1])),
                          "window": i - t0 + 1, "vwap96": vwap[i], "sd96": sd[i]}))
                for x in lvl.members:
                    x.alive = False
                done = True
                break
    return specs


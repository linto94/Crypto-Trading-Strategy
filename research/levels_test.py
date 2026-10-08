"""Tests several ways of marking levels: does price bounce at them more often than
at the SAME levels shifted 1.5 / 3 ATR away (the control)?

Each level is a thin zone: level +/- 0.1 x ATR(24, 1H). Event = 1H bar enters the
zone after the previous bar was fully outside. Outcome = BOUNCE if price moves
1 x ATR away from the near edge before moving 1 x ATR past the far edge.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, sys.path[0])
from lkz_zone_test import load, rma_atr, outcome, run_zones, PAIRS

HALF = 0.1
SHIFTS = (-3.0, -1.5, 1.5, 3.0)
ROUND = {"XAUUSD": 50, "EURUSD": 0.005, "GBPUSD": 0.005, "GER40": 200, "US500": 50, "US100": 200, "BTCUSD": 2000}


def daily_tables(df):
    d = df.copy()
    d["day"] = d.Datetime.dt.floor("D")
    d["week"] = d.Datetime.dt.to_period("W").dt.start_time
    day = d.groupby("day").agg(o=("Open", "first"), h=("High", "max"), l=("Low", "min"), c=("Close", "last"))
    wk = d.groupby("week").agg(h=("High", "max"), l=("Low", "min"))
    return d, day, wk


def levels_by_method(df, nice):
    """dict method -> list (per bar t) of level prices known before bar t."""
    n = len(df)
    d, day, wk = daily_tables(df)
    days = list(day.index)
    di = {x: i for i, x in enumerate(days)}
    weeks = list(wk.index)
    wi = {x: i for i, x in enumerate(weeks)}
    h, l, c = df.High.values, df.Low.values, df.Close.values
    hours = df.Datetime.dt.hour.values
    out = {k: [None] * n for k in ["PDH/PDL", "Prev week H/L", "Asian range H/L", "Round numbers",
                                    "Daily open", "Daily pivot P/R1/S1", "Fresh swing (untouched)"]}
    # Asian range per day (00:00-06:59 UTC), usable from 07:00
    asia = d[d.Datetime.dt.hour < 7].groupby("day").agg(h=("High", "max"), l=("Low", "min"))
    # fresh swings: 1H pivots L=10, alive until price trades through them
    L = 10
    fresh = []
    step = ROUND[nice]
    for t in range(1, n):
        dy = d.day.iat[t]
        k = di[dy]
        if k >= 1:
            p = day.iloc[k - 1]
            out["PDH/PDL"][t] = [p.h, p.l]
            P = (p.h + p.l + p.c) / 3
            out["Daily pivot P/R1/S1"][t] = [P, 2 * P - p.l, 2 * P - p.h]
        out["Daily open"][t] = [day.o.iat[k]] if d.Datetime.iat[t] != dy else None
        w = wi[d.week.iat[t]]
        if w >= 1:
            out["Prev week H/L"][t] = [wk.h.iat[w - 1], wk.l.iat[w - 1]]
        if hours[t] >= 7 and dy in asia.index:
            out["Asian range H/L"][t] = [asia.at[dy, "h"], asia.at[dy, "l"]]
        base = np.floor(c[t - 1] / step) * step
        out["Round numbers"][t] = [base, base + step]
        # fresh swings known before bar t (pivot confirmed at bar t-1)
        j = t - 1 - L
        if j - L >= 0:
            if h[j] == h[j - L:t].max() and h[j] > h[j - L:j].max():
                fresh.append(h[j])
            if l[j] == l[j - L:t].min() and l[j] < l[j - L:j].min():
                fresh.append(l[j])
        out["Fresh swing (untouched)"][t] = list(fresh[-6:])
        # a swing stops being fresh once a bar trades through it (it is touched at t -> event, then removed)
        fresh = [x for x in fresh if not (l[t] <= x <= h[t])]
    return out


def events(df, levels_per_bar, atr, shift=0.0):
    h, l = df.High.values, df.Low.values
    res, last = [], {}
    for t in range(1, len(df)):
        lv = levels_per_bar[t]
        a = atr[t - 1]
        if not lv or np.isnan(a):
            continue
        for x in lv:
            x = x + shift * a
            top, bot = x + HALF * a, x - HALF * a
            if not (h[t] >= bot and l[t] <= top):
                continue
            if l[t - 1] > top:
                side, near, far = 1, top, bot
            elif h[t - 1] < bot:
                side, near, far = -1, bot, top
            else:
                continue
            key = round(x / a * 2)
            if t - last.get(key, -99) < 5:
                continue
            last[key] = t
            o = outcome(h, l, t, side, near, far, a)
            if o is not None:
                res.append(o)
    return res


if __name__ == "__main__":
    DATA = sys.argv[1]
    import lkz_zone_test
    lkz_zone_test.DATA = DATA
    agg = {}
    per_pair = {}
    for nice, f in PAIRS:
        df = load(f"{f}_H1")
        atr = rma_atr(df.High.values, df.Low.values, df.Close.values, 24)
        meths = levels_by_method(df, nice)
        shown, _ = run_zones(df, L=5, mult=0.5, atr_len=24)
        meths["LKZ zones (mid)"] = [None if s is None else [(a + b) / 2 for a, b, _ in s] for s in shown]
        for m, lv in meths.items():
            real = events(df, lv, atr)
            ctrl = []
            for s in SHIFTS:
                ctrl += events(df, lv, atr, s)
            a = agg.setdefault(m, [0, 0, 0, 0])
            a[0] += sum(real); a[1] += len(real); a[2] += sum(ctrl); a[3] += len(ctrl)
            per_pair.setdefault(m, {})[nice] = (np.mean(real) if real else np.nan, np.mean(ctrl) if ctrl else np.nan, len(real))
    print(f"{'Method':26s} {'touches':>8s} {'bounce':>7s} {'control':>8s} {'diff':>7s} {'z':>5s}   pairs better than control")
    rows = []
    for m, (rs, rn, cs, cn) in agg.items():
        pr, pc = rs / rn, cs / cn
        se = np.sqrt(pr * (1 - pr) / rn + pc * (1 - pc) / cn)
        better = sum(1 for v in per_pair[m].values() if v[0] > v[1])
        rows.append((pr - pc, f"{m:26s} {rn:8d} {pr:7.1%} {pc:8.1%} {pr - pc:+7.1%} {(pr - pc) / se:5.1f}   {better}/7"))
    for _, r in sorted(rows, reverse=True):
        print(r)

"""15M scalping setups, after spread costs, with an honest split:
TRAIN = first 8 months (choose), TEST = last 4 months (confirm, never used for choosing).
Setups: FVG limit, FVG + rejection close, CHoCH at LKZ zone, session opening-range breakout.
Targets 1.5R / 2R / 3R. Same-bar stop & target = stop. Time stop TS bars (exit at close).
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, sys.path[0])
from lkz_zone_test import load, rma_atr, run_zones, PAIRS
from trade_sim import minor_swing, POINT

SPLIT = pd.Timestamp("2026-06-08")
TS = 16
TARGETS = (1.5, 2.0, 3.0)


def trade(h, l, c, t, side, entry, stop, k_tgt, start_next=True):
    risk = (entry - stop) * side
    if risk <= 0:
        return None
    tgt = entry + side * k_tgt * risk
    first = t + 1 if start_next else t
    for k in range(first, min(t + 1 + TS, len(c))):
        if (l[k] <= stop) if side > 0 else (h[k] >= stop):
            return -1.0, risk
        if k == t:
            continue  # entry bar: a target touch may have happened before the fill, so ignore it
        if (h[k] >= tgt) if side > 0 else (l[k] <= tgt):
            return k_tgt, risk
    k = min(t + TS, len(c) - 1)
    return (c[k] - entry) * side / risk, risk


def setups(df, nice):
    o, h, l, c = df.Open.values, df.High.values, df.Low.values, df.Close.values
    sp = df.Spread.values * POINT[nice]
    atr = rma_atr(h, l, c, 96)
    when = df.Datetime.values
    rows = []

    def add(name, t, side, entry, stop, start_next=True):
        for k in TARGETS:
            r = trade(h, l, c, t, side, entry, stop, k, start_next)
            if r:
                R, risk = r
                rows.append(dict(setup=name, target=k, when=when[t], R=R - sp[t] / risk, Rg=R))

    # --- FVG setups (first return)
    for i in range(2, len(c) - 2):
        a = atr[i]
        if np.isnan(a):
            continue
        for d, top, bot in ((1, l[i], h[i - 2]), (-1, l[i - 2], h[i])):
            if top - bot < 0.1 * a:
                continue
            for t in range(i + 1, min(i + 200, len(c) - 1)):
                if h[t] >= bot and l[t] <= top:
                    if (d == 1 and l[t - 1] > top) or (d == -1 and h[t - 1] < bot):
                        near = top if d == 1 else bot
                        far = bot if d == 1 else top
                        add("FVG limit", t, d, near, far - d * 0.25 * a, start_next=False)
                        if (d == 1 and c[t] > top) or (d == -1 and c[t] < bot):
                            ext = min(l[t], bot) if d == 1 else max(h[t], top)
                            add("FVG + rejection", t, d, c[t], ext - d * 0.1 * a)
                    break
    # --- CHoCH at LKZ zone
    shown, _ = run_zones(df, L=10, mult=1.0, atr_len=96)
    last = {}
    for t in range(12, len(c) - 2):
        zs = shown[t]; a = atr[t - 1]
        if not zs or np.isnan(a):
            continue
        for top, bot, cnt in zs:
            if not (h[t] >= bot and l[t] <= top):
                continue
            side = 1 if l[t - 1] > top else -1 if h[t - 1] < bot else 0
            if side == 0:
                continue
            key = round((top + bot) / 2 / a)
            if t - last.get(key, -99) < 5:
                continue
            last[key] = t
            lvl = minor_swing(h, l, t, side)
            if lvl is None:
                continue
            ext = l[t] if side == 1 else h[t]
            for u in range(t, min(t + 8, len(c) - 1)):
                ext = min(ext, l[u]) if side == 1 else max(ext, h[u])
                if (side == 1 and ext < bot - a) or (side == -1 and ext > top + a):
                    break
                if (side == 1 and c[u] > lvl) or (side == -1 and c[u] < lvl):
                    add("CHoCH at zone", u, side, c[u], ext - side * 0.1 * a)
                    break
    # --- session opening-range breakout (London 08:00, New York 09:30 local), first 30 minutes
    ts = pd.Series(pd.to_datetime(df.Datetime)).dt.tz_localize("UTC")
    for tz, hh, mm, label in (("Europe/London", 8, 0, "London ORB"), ("America/New_York", 9, 30, "NY ORB")):
        loc = ts.dt.tz_convert(tz)
        starts = np.where((loc.dt.hour == hh) & (loc.dt.minute == mm) & (loc.dt.dayofweek < 5))[0]
        for s in starts:
            if s + 14 >= len(c):
                continue
            hi, lo = h[s:s + 2].max(), l[s:s + 2].min()
            for t in range(s + 2, s + 14):
                if c[t] > hi:
                    add(label, t, 1, c[t], lo); break
                if c[t] < lo:
                    add(label, t, -1, c[t], hi); break
    return rows


def summary(d):
    if len(d) < 20:
        return None
    m = d.R.mean()
    return dict(n=len(d), win=np.mean(d.R > 0), avg=m, gross=d.Rg.mean(), t=m / (d.R.std() / np.sqrt(len(d))))


if __name__ == "__main__":
    import lkz_zone_test
    lkz_zone_test.DATA = sys.argv[1]
    rows = []
    for nice, f in PAIRS:
        rows += [dict(r, pair=nice) for r in setups(load(f"{f}_M15"), nice)]
    d = pd.DataFrame(rows)
    d["when"] = pd.to_datetime(d.when)
    d.to_csv("scalp_trades.csv", index=False)
    print(f"{'setup':16s} {'tgt':>4s} | TRAIN (Oct-Jun): trades  win   avg R after costs (before)  t     | TEST (Jun-Oct): trades  win   avg R  t     | pairs>0 in TEST")
    for (s, k), g in d.groupby(["setup", "target"]):
        tr, te = summary(g[g.when < SPLIT]), summary(g[g.when >= SPLIT])
        if not tr or not te:
            continue
        pp = sum(x.R.mean() > 0 for _, x in g[g.when >= SPLIT].groupby("pair"))
        print(f"{s:16s} {k:4.1f} | {tr['n']:6d} {tr['win']:5.1%} {tr['avg']:+.3f} ({tr['gross']:+.3f}) {tr['t']:+5.1f} | "
              f"{te['n']:6d} {te['win']:5.1%} {te['avg']:+.3f} {te['t']:+5.1f} | {pp}/7")
    print("\nBest pair+setup combos chosen on TRAIN (avg R > 0, 40+ trades), checked on TEST:")
    for (s, k, p), g in d.groupby(["setup", "target", "pair"]):
        tr, te = summary(g[g.when < SPLIT]), summary(g[g.when >= SPLIT])
        if tr and te and tr["n"] >= 40 and tr["avg"] > 0.1:
            print(f"  {s:16s} {k:.1f}R {p:7s} TRAIN {tr['n']:4d} trades avg {tr['avg']:+.3f}R  ->  TEST {te['n']:4d} trades avg {te['avg']:+.3f}R")

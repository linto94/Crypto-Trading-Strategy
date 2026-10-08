"""Creator 1: daily-timeframe levels.
S/R  = daily zones with 3+ swing touches.  Key level = daily swings touched only 1-2 times.
Daily bars built from 1H data; zones known before each day; touches/outcomes measured on 1H
exactly like the other tests. Control = same zones shifted +/-1 and +/-2 daily ATR.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, sys.path[0])
from lkz_zone_test import load, rma_atr, outcome, run_zones, PAIRS


def events(h, l, atr1h, zones_per_bar, shifts_d, dAtr_per_bar):
    res, last = [], {}
    for t in range(1, len(h)):
        zs = zones_per_bar[t]
        a = atr1h[t - 1]
        if not zs or np.isnan(a) or np.isnan(dAtr_per_bar[t]):
            continue
        off = shifts_d * dAtr_per_bar[t]
        for top, bot in zs:
            top, bot = top + off, bot + off
            if not (h[t] >= bot and l[t] <= top):
                continue
            if l[t - 1] > top:
                side, near, far = 1, top, bot
            elif h[t - 1] < bot:
                side, near, far = -1, bot, top
            else:
                continue
            key = round((top + bot) / 2 / a)
            if t - last.get(key, -99) < 5:
                continue
            last[key] = t
            o = outcome(h, l, t, side, near, far, a)
            if o is not None:
                res.append(o)
    return res


def nearest(zs, px, k=2):
    up = sorted([z for z in zs if (z[0] + z[1]) / 2 >= px], key=lambda z: max(z[1] - px, 0))[:k]
    dn = sorted([z for z in zs if (z[0] + z[1]) / 2 < px], key=lambda z: max(px - z[0], 0))[:k]
    return up + dn


if __name__ == "__main__":
    import lkz_zone_test
    lkz_zone_test.DATA = sys.argv[1]
    agg = {}
    for nice, f in PAIRS:
        df = load(f"{f}_H1")
        h, l, c = df.High.values, df.Low.values, df.Close.values
        atr1h = rma_atr(h, l, c, 24)
        d = df.assign(day=df.Datetime.dt.floor("D")).groupby("day").agg(
            Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"), Close=("Close", "last")).reset_index()
        d = d.rename(columns={"day": "Datetime"})
        shown, datr = run_zones(d, L=2, mult=0.5, atr_len=14, min_piv=1, max_piv=40, per_side=50)
        day_idx = {x: i for i, x in enumerate(d.Datetime)}
        bar_day = df.Datetime.dt.floor("D").map(day_idx).values
        sets = {"Daily S/R (3+ touches)": [], "Daily key level (1-2 touches)": []}
        dAtr = np.full(len(df), np.nan)
        for t in range(len(df)):
            k = bar_day[t]
            zs = shown[k] if k is not None and k < len(shown) else None
            dAtr[t] = datr[k - 1] if k >= 1 else np.nan
            if not zs or t == 0:
                sets["Daily S/R (3+ touches)"].append(None); sets["Daily key level (1-2 touches)"].append(None)
                continue
            sr = [(a, b) for a, b, n in zs if n >= 3]
            kl = [(a, b) for a, b, n in zs if n <= 2]
            sets["Daily S/R (3+ touches)"].append(nearest(sr, c[t - 1]))
            sets["Daily key level (1-2 touches)"].append(nearest(kl, c[t - 1]))
        for m, zpb in sets.items():
            real = events(h, l, atr1h, zpb, 0.0, dAtr)
            ctrl = sum((events(h, l, atr1h, zpb, s, dAtr) for s in (-2.0, -1.0, 1.0, 2.0)), [])
            a = agg.setdefault(m, [0, 0, 0, 0, 0])
            a[0] += sum(real); a[1] += len(real); a[2] += sum(ctrl); a[3] += len(ctrl)
            a[4] += int(bool(real) and bool(ctrl) and np.mean(real) > np.mean(ctrl))
    for m, (rs, rn, cs, cn, better) in agg.items():
        pr, pc = rs / rn, cs / cn
        se = np.sqrt(pr * (1 - pr) / rn + pc * (1 - pc) / cn)
        print(f"{m:32s} touches {rn:5d}  bounce {pr:.1%}  control {pc:.1%}  diff {pr - pc:+.1%}  z={(pr - pc) / se:.1f}  pairs better {better}/7")

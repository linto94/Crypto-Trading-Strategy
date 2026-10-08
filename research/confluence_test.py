"""Confluence filters on LKZ zone touches (real zones vs the same zones shifted +/-1.5, +/-3 ATR).
For every touch we record features known at the close of the touch bar; the outcome starts
on the next bar (same bounce/break rule as the other tests).
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, sys.path[0])
from lkz_zone_test import load, rma_atr, outcome, run_zones, PAIRS

ROUND = {"XAUUSD": 50, "EURUSD": 0.005, "GBPUSD": 0.005, "GER40": 200, "US500": 50, "US100": 200, "BTCUSD": 2000}


def ema(x, n):
    out = np.empty(len(x)); a = 2 / (n + 1); out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0])
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    ru, rd = rma(up, n), rma(dn, n)
    return 100 - 100 / (1 + ru / np.where(rd == 0, 1e-12, rd))


def rma(x, n):
    out = np.full(len(x), np.nan); out[n - 1] = x[:n].mean()
    for i in range(n, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def collect(df, nice, cfg, per_day):
    o, h, l, c, v = df.Open.values, df.High.values, df.Low.values, df.Close.values, df.Volume.values.astype(float)
    shown, atr = run_zones(df, **cfg)
    e200 = ema(c, 200)
    r = rsi(c)
    vma = pd.Series(v).rolling(20).mean().values
    rng = h - l
    hours = df.Datetime.dt.hour.values
    day = df.Datetime.dt.floor("D")
    dd = df.assign(day=day).groupby("day").agg(h=("High", "max"), l=("Low", "min"))
    prev = dd.shift(1)
    pdh = day.map(prev.h).values; pdl = day.map(prev.l).values
    step = ROUND[nice]
    rows = []
    for shift in (0.0, -3.0, -1.5, 1.5, 3.0):
        last = {}
        for t in range(10, len(df) - 1):
            zs = shown[t]; a = atr[t - 1]
            if not zs or np.isnan(a):
                continue
            for top, bot, cnt in zs:
                top, bot = top + shift * a, bot + shift * a
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
                res = outcome(h, l, t, side, near, far, a)
                if res is None:
                    continue
                mid = (top + bot) / 2
                approach = (c[t - 7] - c[t - 1]) * side / a          # how far price travelled into the zone
                shrink = rng[t - 3:t].mean() < rng[t - 6:t - 3].mean()
                hr = hours[t]
                sess = "London open" if 7 <= hr < 10 else "NY open" if 13 <= hr < 16 else "Asia" if hr < 7 else "Other"
                rnd = abs(mid - round(mid / step) * step) <= max(0.3 * a, (top - bot) / 2)
                pdx = any(not np.isnan(x) and bot - 0.3 * a <= x <= top + 0.3 * a for x in (pdh[t], pdl[t]))
                rows.append(dict(
                    real=shift == 0.0, win=res,
                    trend="with trend" if (c[t - 1] - e200[t - 1]) * side > 0 else "against trend",
                    approach="fast (>3 ATR)" if approach > 3 else "medium (1.5-3 ATR)" if approach > 1.5 else "slow (<1.5 ATR)",
                    exhaustion="shrinking candles" if shrink else "growing candles",
                    session=sess,
                    rsi="RSI extreme" if (side == 1 and r[t] < 30) or (side == -1 and r[t] > 70) else "RSI normal",
                    volume="volume spike" if vma[t - 1] > 0 and v[t] > 1.5 * vma[t - 1] else "normal volume",
                    rejection="rejection close" if (side == 1 and c[t] > top) or (side == -1 and c[t] < bot) else "closed in/through",
                    sweep="sweep (poked through, closed back)" if ((side == 1 and l[t] < bot and c[t] > bot) or (side == -1 and h[t] > top and c[t] < top)) else "no sweep",
                    stacked="stacked with PDH/PDL or round no." if (rnd or pdx) else "zone alone",
                    strength="4+ pivots" if cnt >= 4 else "3 pivots",
                ))
    return rows


if __name__ == "__main__":
    import lkz_zone_test
    lkz_zone_test.DATA = sys.argv[1]
    for name, tf, cfg, per_day in [("15M", "M15", dict(L=10, mult=1.0, atr_len=96), 96), ("1H", "H1", dict(L=5, mult=0.5, atr_len=24), 24)]:
        rows = []
        for nice, f in PAIRS:
            rows += [dict(r, pair=nice) for r in collect(load(f"{f}_{tf}"), nice, cfg, per_day)]
        d = pd.DataFrame(rows)
        d.to_csv(f"confluence_{tf}.csv", index=False)
        base_r = d[d.real].win.mean(); base_c = d[~d.real].win.mean()
        print(f"\n===== {name}: all touches  real {base_r:.1%} (n={d.real.sum()})  made-up {base_c:.1%} (n={(~d.real).sum()})")
        for feat in ["trend", "approach", "exhaustion", "session", "rsi", "volume", "rejection", "sweep", "stacked", "strength"]:
            for val, g in d.groupby(feat):
                rr, cc = g[g.real], g[~g.real]
                if len(rr) < 30:
                    continue
                pr, pc = rr.win.mean(), cc.win.mean()
                se = np.sqrt(pr * (1 - pr) / len(rr) + pc * (1 - pc) / max(len(cc), 1))
                print(f"  {feat:10s} {val:36s} real {pr:6.1%} (n={len(rr):5d}) | made-up {pc:6.1%} | real-vs-made-up {pr - pc:+5.1%} (z={(pr - pc) / se:4.1f}) | vs all real {pr - base_r:+5.1%}")

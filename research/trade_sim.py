"""Full trade simulation in R, after spread costs.
A  Rejection at LKZ zone: touch bar closes back outside the zone -> enter at that close.
B  Same at made-up zones (LKZ zones shifted +/-1.5, +/-3 ATR).
C  A but skip when RSI is extreme or the touch bar has a volume spike.   (C-made-up too)
D  CHoCH at LKZ zone: after the touch, enter when a bar closes beyond the last minor swing
   of the run (2-bar pivot) within M bars.                                (D-made-up too)
Stop = run/wick extreme beyond the zone - 0.1 ATR. Target = 1.5R. Time stop after TS bars.
Same-bar stop & target = stop. Cost = spread at entry, in R.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, sys.path[0])
from lkz_zone_test import load, rma_atr, run_zones, PAIRS
from confluence_test import rsi

POINT = {"XAUUSD": 1e-3, "EURUSD": 1e-5, "GBPUSD": 1e-5, "GER40": 1e-3, "US500": 1e-3, "US100": 1e-3, "BTCUSD": 0.1}
TGT = 1.5


def run_trade(h, l, c, t, side, entry, stop, ts):
    risk = (entry - stop) * side
    if risk <= 0:
        return None
    tgt = entry + side * TGT * risk
    for k in range(t + 1, min(t + 1 + ts, len(c))):
        hit_s = l[k] <= stop if side > 0 else h[k] >= stop
        hit_t = h[k] >= tgt if side > 0 else l[k] <= tgt
        if hit_s:
            return -1.0, risk
        if hit_t:
            return TGT, risk
    k = min(t + ts, len(c) - 1)
    return (c[k] - entry) * side / risk, risk


def minor_swing(h, l, t, side, lookback=12):
    """last 2-bar pivot high (side=+1, price fell into support) / low before or at t."""
    for k in range(t - 2, max(t - lookback, 2), -1):
        if side > 0 and h[k] > max(h[k - 2:k].max(), h[k + 1:k + 3].max()):
            return h[k]
        if side < 0 and l[k] < min(l[k - 2:k].min(), l[k + 1:k + 3].min()):
            return l[k]
    return None


def simulate(df, nice, cfg, ts, m_choch):
    h, l, c, v = df.High.values, df.Low.values, df.Close.values, df.Volume.values.astype(float)
    spread = df.Spread.values * POINT[nice]
    shown, atr = run_zones(df, **cfg)
    r = rsi(c)
    vma = pd.Series(v).rolling(20).mean().values
    out = []
    for shift in (0.0, -3.0, -1.5, 1.5, 3.0):
        last = {}
        busy_until = -1
        for t in range(12, len(df) - 2):
            zs = shown[t]; a = atr[t - 1]
            if not zs or np.isnan(a):
                continue
            for top, bot, cnt in zs:
                top, bot = top + shift * a, bot + shift * a
                if not (h[t] >= bot and l[t] <= top):
                    continue
                if l[t - 1] > top:
                    side = 1
                elif h[t - 1] < bot:
                    side = -1
                else:
                    continue
                key = round((top + bot) / 2 / a)
                if t - last.get(key, -99) < 5:
                    continue
                last[key] = t
                stretched = (side == 1 and r[t] < 30) or (side == -1 and r[t] > 70) or (vma[t - 1] > 0 and v[t] > 1.5 * vma[t - 1])
                # --- rejection entry
                if (side == 1 and c[t] > top) or (side == -1 and c[t] < bot):
                    stop = (min(l[t], bot) - 0.1 * a) if side == 1 else (max(h[t], top) + 0.1 * a)
                    res = run_trade(h, l, c, t, side, c[t], stop, ts)
                    if res:
                        R, risk = res
                        cost = spread[t] / risk
                        out.append(dict(kind="rejection", real=shift == 0, stretched=stretched, R=R - cost, Rgross=R))
                # --- CHoCH entry
                lvl = minor_swing(h, l, t, side)
                if lvl is None:
                    continue
                ext = l[t] if side == 1 else h[t]
                for u in range(t, min(t + m_choch, len(c) - 1)):
                    ext = min(ext, l[u]) if side == 1 else max(ext, h[u])
                    far_break = (ext < bot - 1.0 * a) if side == 1 else (ext > top + 1.0 * a)
                    if far_break:
                        break
                    if (side == 1 and c[u] > lvl) or (side == -1 and c[u] < lvl):
                        stop = ext - 0.1 * a if side == 1 else ext + 0.1 * a
                        res = run_trade(h, l, c, u, side, c[u], stop, ts)
                        if res:
                            R, risk = res
                            out.append(dict(kind="CHoCH", real=shift == 0, stretched=stretched, R=R - spread[u] / risk, Rgross=R))
                        break
    return out


def report(d, label):
    if len(d) < 20:
        print(f"  {label:44s} too few trades ({len(d)})")
        return
    m, s = d.R.mean(), d.R.std()
    print(f"  {label:44s} trades {len(d):5d}  win {np.mean(d.R > 0):5.1%}  avg {m:+.3f}R  (before costs {d.Rgross.mean():+.3f}R)  "
          f"total {d.R.sum():+7.1f}R  t={m / (s / np.sqrt(len(d))):+.1f}  pairs>0 {sum(g.R.mean() > 0 for _, g in d.groupby('pair'))}/7")


if __name__ == "__main__":
    import lkz_zone_test
    lkz_zone_test.DATA = sys.argv[1]
    for name, tf, cfg, ts, mc in [("15M", "M15", dict(L=10, mult=1.0, atr_len=96), 16, 8), ("1H", "H1", dict(L=5, mult=0.5, atr_len=24), 12, 6)]:
        rows = []
        for nice, f in PAIRS:
            rows += [dict(x, pair=nice) for x in simulate(load(f"{f}_{tf}"), nice, cfg, ts, mc)]
        d = pd.DataFrame(rows)
        d.to_csv(f"trades_{tf}.csv", index=False)
        print(f"\n===== {name} (target 1.5R, costs = data spread)")
        for kind in ("rejection", "CHoCH"):
            report(d[(d.kind == kind) & d.real], f"{kind} at LKZ zone")
            report(d[(d.kind == kind) & ~d.real], f"{kind} at made-up zone")
            report(d[(d.kind == kind) & d.real & ~d.stretched], f"{kind} at LKZ zone, skip RSI/volume extremes")
            report(d[(d.kind == kind) & ~d.real & ~d.stretched], f"{kind} at made-up zone, skip extremes")

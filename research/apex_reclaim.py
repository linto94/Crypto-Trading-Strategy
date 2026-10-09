"""Linto-Apex zones (pivot 10) - RECLAIM entry.

Drawn box = zone mid +/- 0.3 x ATR14 (what you see on the chart).
BUY  : a candle CLOSES below the box bottom (previous close was at/above it), then within
       10 candles a candle CLOSES back above the box bottom -> buy at that close.
       Stop = lowest low since the break - 0.1 ATR.
SELL : mirror at the box top.
Target 1.5R or 2R. SL checked first each bar. One trade at a time per pair/TF.
Cost = spread at entry, in R.
Control: the same rules on made-up zones (real zones shifted +/-1.5 and +/-3 ATR).
"""
import sys
import numpy as np
import pandas as pd
from apex_test import rma_atr, pivots, build_zones, PAIRS, POINT, stats, SPLIT

WIN = 10


def zone_series(df, P=10):
    h, l = df.High.values, df.Low.values
    ph, pl = pivots(h, P, True), pivots(l, P, False)
    rng = pd.Series(h).rolling(300).max().values - pd.Series(l).rolling(300).min().values
    pv, zones, out = [], [], []
    for t in range(len(df)):
        new = False
        if not np.isnan(ph[t]):
            pv.insert(0, ph[t]); pv = pv[:20]; new = True
        if not np.isnan(pl[t]):
            pv.insert(0, pl[t]); pv = pv[:20]; new = True
        if new:
            zones = build_zones(pv, 0.10 * rng[t]) if not np.isnan(rng[t]) and len(pv) >= 2 else []
        out.append([z[0] for z in zones])
    return out


def simulate(df, nice, mids, shift, rr):
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    when = df.Datetime.values
    sp = df.Spread.values * POINT[nice]
    atr = rma_atr(h, l, c, 14)
    pend = {}          # key -> [side, level, extreme, start]
    pos = None; trades = []
    for t in range(301, len(c)):
        a = atr[t]
        if np.isnan(a):
            continue
        if pos is not None:
            side, ent, stop, tgt, risk, k0, cost = pos
            if (l[t] <= stop) if side > 0 else (h[t] >= stop):
                trades.append(dict(pair=nice, side="BUY" if side > 0 else "SELL", when=when[k0], R=-1 - cost, Rg=-1.0)); pos = None
            elif (h[t] >= tgt) if side > 0 else (l[t] <= tgt):
                trades.append(dict(pair=nice, side="BUY" if side > 0 else "SELL", when=when[k0], R=rr - cost, Rg=rr)); pos = None
        # update pending setups / look for reclaim
        signal = None
        for key in list(pend):
            side, lvl, ext, st = pend[key]
            ext = min(ext, l[t]) if side > 0 else max(ext, h[t])
            pend[key][2] = ext
            if t > st and ((c[t] > lvl) if side > 0 else (c[t] < lvl)):
                if signal is None:
                    signal = (side, ext)
                del pend[key]
            elif t - st >= WIN:
                del pend[key]
        # new breaks on this bar (zones known at this bar's close)
        for m in mids[t]:
            m = m + shift * a
            bot, top = m - 0.3 * a, m + 0.3 * a
            key = round(m / a)
            if c[t] < bot <= c[t - 1]:
                pend[(key, 1)] = [1, bot, l[t], t]
            if c[t] > top >= c[t - 1]:
                pend[(key, -1)] = [-1, top, h[t], t]
        if pos is None and signal is not None:
            side, ext = signal
            stop = ext - 0.1 * a if side > 0 else ext + 0.1 * a
            risk = (c[t] - stop) * side
            if risk > 0:
                pos = (side, c[t], stop, c[t] + side * rr * risk, risk, t, sp[t] / risk)
    return trades


if __name__ == "__main__":
    DATA = sys.argv[1]; P = int(sys.argv[2]) if len(sys.argv) > 2 else 10; rows = []
    for tf in ("H4", "H1"):
        for nice, f in PAIRS:
            df = pd.read_csv(f"{DATA}/{f}_{tf}.csv", parse_dates=["Datetime"])
            mids = zone_series(df, P)
            for shift in (0.0, -3.0, -1.5, 1.5, 3.0):
                for rr in (1.5, 2.0):
                    d = pd.DataFrame(simulate(df, nice, mids, shift, rr))
                    if len(d):
                        d["tf"] = tf; d["rr"] = rr; d["real"] = shift == 0.0; rows.append(d)
    d = pd.concat(rows); d.to_csv(f"apex_reclaim_trades_P{P}.csv", index=False)
    order = ["XAUUSD", "EURUSD", "GBPUSD", "GER40", "US500", "US100", "BTCUSD"]
    d["when"] = pd.to_datetime(d.when)
    for tf in ("H4", "H1"):
        for rr in (1.5, 2.0):
            for real in (True, False):
                g = d[(d.tf == tf) & (d.rr == rr) & (d.real == real)]
                a, e, u = stats(g), stats(g[g.when < SPLIT]), stats(g[g.when >= SPLIT])
                print(f"ALL {tf} {rr}R {'REAL' if real else 'made-up'}: n {a['n']} win {a['win']:.1%} avg {a['avg']:+.3f} (before {a['avg_g']:+.3f}) early {e['avg']:+.3f} unseen {u['avg']:+.3f}")
            print(f"\n{tf} {rr}R REAL zones: pair side n win avg tot streak dd | early | unseen")
            for p in order:
                for s in ("BUY", "SELL"):
                    g = d[(d.tf == tf) & (d.rr == rr) & d.real & (d.pair == p) & (d.side == s)].sort_values("when")
                    a, e, u = stats(g), stats(g[g.when < SPLIT]), stats(g[g.when >= SPLIT])
                    print(f"{p} {s} {a['n']} {a['win']:.0%} {a['avg']:+.2f} {a['tot']:+.0f} {a['streak']} {a['dd']:.0f} | {e['avg']:+.2f} ({e['n']}) | {u['avg']:+.2f} ({u['n']})")
            print()

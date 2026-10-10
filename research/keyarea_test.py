"""APN 'key area' method (two videos: significant turning points + key areas).

Structure (real time, no hindsight):
  Downtrend has: B = the broken significant low, P = the last significant lower high above it.
  After B breaks, the new low only becomes SIGNIFICANT once price pushes back up to retest B
  (high >= B), or - if the move was impulsive - retraces 38.2% of the move from P (video 1 Fib
  rule), whichever comes first. Smaller bounces are ignored.
  Continuation: after that retest, a close below the new significant low -> new B = that low,
  new P = highest high of the retest pullback.
  Trend change: a close above P (the last significant lower high). New uptrend: B = old P,
  P = lowest low of the down leg. Uptrend is the mirror.
  CONTROL ("every swing"): the new low counts as soon as any 3-candle swing low forms (no retest
  needed) - this is how ordinary indicators treat swings.
Key area = between B and P. Trades only with the trend, one trade per structure leg.
Entry A: limit order at B (what the video says NOT to do - for comparison).
Entry B: price enters the key area, then a candle CLOSES back out of it (below B for sells,
         above B for buys) -> enter at that close.
Stop: beyond P (the whole structure) + 0.1 ATR. Targets 1.3R / 1.5R / 2R. Stop checked first.
One trade at a time. Cost = data spread at entry.
"""
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
from apex_test import rma_atr, PAIRS, POINT, stats, SPLIT

FIB = 0.382   # impulsive move: a 38.2% pullback also makes the turning point significant
K = 3   # candles each side for the control's "every swing"


def run(args):
    DATA, nice, f, tf, mode, entry, rr = args
    df = pd.read_csv(f"{DATA}/{f}_{tf}.csv", parse_dates=["Datetime"])
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    when = df.Datetime.values
    sp = df.Spread.values * POINT[nice]
    atr = rma_atr(h, l, c, 14)
    n = len(c)
    # init: first close outside the first 20 candles' range
    hi0, lo0 = h[:20].max(), l[:20].min()
    d = 0; t0 = 20
    while t0 < n and d == 0:
        if c[t0] > hi0: d, B, P, X = 1, hi0, lo0, h[t0]
        elif c[t0] < lo0: d, B, P, X = -1, lo0, hi0, l[t0]
        t0 += 1
    sig = False; S = np.nan; R = np.nan; leg = 0; leg_start = t0
    traded_leg = -1; inside = False; order = None; pos = None
    trades = []
    for t in range(t0, n):
        a = atr[t - 1] if not np.isnan(atr[t - 1]) else atr[t]
        # ---- open trade
        if pos is not None:
            side, ent, stop, tgt, risk, k0, cost = pos
            if t > k0:
                if (l[t] <= stop) if side > 0 else (h[t] >= stop):
                    trades.append(dict(pair=nice, tf=tf, side="BUY" if side > 0 else "SELL", when=when[k0], R=-1 - cost, Rg=-1.0)); pos = None
                elif (h[t] >= tgt) if side > 0 else (l[t] <= tgt):
                    trades.append(dict(pair=nice, tf=tf, side="BUY" if side > 0 else "SELL", when=when[k0], R=rr - cost, Rg=rr)); pos = None
        # ---- entry A: limit at B placed at previous close
        if entry == "A" and order is not None and pos is None:
            side, lim, stop, oleg = order
            if oleg == leg and ((h[t] >= lim) if side < 0 else (l[t] <= lim)):
                ent = max(o[t], lim) if side < 0 else min(o[t], lim)
                risk = (stop - ent) * (-side) if side < 0 else (ent - stop)
                if risk > 0:
                    cost = sp[t] / risk
                    traded_leg = leg
                    if (h[t] >= stop) if side < 0 else (l[t] <= stop):
                        trades.append(dict(pair=nice, tf=tf, side="BUY" if side > 0 else "SELL", when=when[t], R=-1 - cost, Rg=-1.0))
                    else:
                        pos = (side, ent, stop, ent + side * rr * risk, risk, t, cost)
            order = None
        # ---- key-area touch on this candle (before structure update)
        if (h[t] >= B) if d < 0 else (l[t] <= B):
            inside = True
        # ---- structure update with this candle
        old_leg = leg
        if d < 0:
            X = min(X, l[t])
            if not sig:
                if mode == "sig" and h[t] >= min(B, X + FIB * (P - X)):
                    sig, S, R = True, X, h[t]
                elif mode == "all" and t - K - 1 >= leg_start:
                    k = t - K
                    if l[k] == X and l[k] < l[k - K:k].min() and l[k] <= l[k + 1:t + 1].min():
                        sig, S, R = True, l[k], h[k:t + 1].max()
            else:
                R = max(R, h[t])
            if c[t] > P:
                d, B, P, X = 1, P, X, h[t]; sig = False; leg += 1
            elif sig and c[t] < S:
                B, P, X = S, R, l[t]; sig = False; leg += 1
        else:
            X = max(X, h[t])
            if not sig:
                if mode == "sig" and l[t] <= max(B, X - FIB * (X - P)):
                    sig, S, R = True, X, l[t]
                elif mode == "all" and t - K - 1 >= leg_start:
                    k = t - K
                    if h[k] == X and h[k] > h[k - K:k].max() and h[k] >= h[k + 1:t + 1].max():
                        sig, S, R = True, h[k], l[k:t + 1].min()
            else:
                R = min(R, l[t])
            if c[t] < P:
                d, B, P, X = -1, P, X, l[t]; sig = False; leg += 1
            elif sig and c[t] > S:
                B, P, X = S, R, h[t]; sig = False; leg += 1
        if leg != old_leg:
            inside = False; leg_start = t
            continue
        if pos is not None or traded_leg == leg or np.isnan(a):
            continue
        side = 1 if d > 0 else -1
        stop = P - 0.1 * a if side > 0 else P + 0.1 * a
        if entry == "A":
            if (c[t] < B) if side < 0 else (c[t] > B):
                order = (side, B, stop, leg)
        elif inside and ((c[t] < B) if side < 0 else (c[t] > B)):
            risk = (c[t] - stop) * side
            if risk > 0:
                traded_leg = leg
                pos = (side, c[t], stop, c[t] + side * rr * risk, risk, t, sp[t] / risk)
    out = pd.DataFrame(trades)
    if len(out):
        out["mode"], out["entry"], out["rr"] = mode, entry, rr
    return out


if __name__ == "__main__":
    DATA = sys.argv[1]
    jobs = [(DATA, nice, f, tf, mode, e, rr) for nice, f in PAIRS for tf in ("H4", "H1")
            for mode in ("sig", "all") for e in ("A", "B") for rr in (1.3, 1.5, 2.0)]
    with Pool(4) as p:
        d = pd.concat([x for x in p.map(run, jobs) if len(x)])
    d["when"] = pd.to_datetime(d.when)
    d.to_csv("keyarea_trades.csv", index=False)
    for tf in ("H4", "H1"):
        for e in ("A", "B"):
            for rr in (1.3, 1.5, 2.0):
                line = []
                for mode in ("sig", "all"):
                    g = d[(d.tf == tf) & (d.entry == e) & (d.rr == rr) & (d["mode"] == mode)]
                    a, x, u = stats(g), stats(g[g.when < SPLIT]), stats(g[g.when >= SPLIT])
                    line.append(f"{mode}: n {a['n']:5d} win {a['win']:.0%} avg {a['avg']:+.3f} (pre {a['avg_g']:+.3f}) {x['avg']:+.3f}/{u['avg']:+.3f}")
                print(f"{tf} entry {e} {rr}R | " + " | ".join(line))

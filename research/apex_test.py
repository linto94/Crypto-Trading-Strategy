"""Linto-Apex v8.1 trade-setup backtest (indicator #1).

Zones (exact copy of the Pine logic): pivots (High/Low, period P) -> last 20 pivots ->
greedy clusters with width = 10% of the 300-bar range, min 2 pivots, overlap -> stronger
wins, top 5 by strength. Rebuilt only when a new pivot confirms (P bars later).

Setup at every bar close while flat: nearest zone midpoint to close. Mid below close = BUY,
above = SELL. Limit order at the mid for the next bar (re-placed every bar).
SL = cluster low - 0.15% (buy) / cluster high + 0.15% (sell).
TP1/TP2 = mids of the next zones on the far side of price, nearest first.
Variants: EDGE = limit at the near edge of the drawn box (mid +/- 0.3 x ATR14) instead of the mid.
Fill: if the bar trades through the limit (gap -> open). SL on the fill bar = loss; TP on the
fill bar ignored. Then SL checked before TP each bar. No time stop. One trade at a time.
Cost = data spread at the fill bar, in R.
"""
import sys
import numpy as np
import pandas as pd

POINT = {"XAUUSD": 1e-3, "EURUSD": 1e-5, "GBPUSD": 1e-5, "GER40": 1e-3, "US500": 1e-3, "US100": 1e-3, "BTCUSD": 0.1}
PAIRS = [("XAUUSD", "XAUUSD"), ("EURUSD", "EURUSD"), ("GBPUSD", "GBPUSD"), ("GER40", "DEUIDXEUR"),
         ("US500", "USA500IDXUSD"), ("US100", "USATECHIDXUSD"), ("BTCUSD", "BTCUSD")]
SPLIT = pd.Timestamp("2025-01-01")


def rma_atr(h, l, c, n):
    tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
    tr[0] = h[0] - l[0]
    out = np.full(len(tr), np.nan)
    out[n - 1] = tr[:n].mean()
    for i in range(n, len(tr)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def pivots(x, p, high):
    """Pine-style pivot at bar t (confirmed) for bar t-p."""
    n = len(x); out = np.full(n, np.nan)
    for t in range(2 * p, n):
        c = t - p; v = x[c]
        left, right = x[c - p:c], x[c + 1:t + 1]
        if high and v > left.max() and v >= right.max():
            out[t] = v
        if not high and v < left.min() and v <= right.min():
            out[t] = v
    return out


def build_zones(piv, width, min_str=2, max_zones=5):
    n = len(piv); used = [False] * n
    cm, cs, ch, cl = [], [], [], []
    for i in range(n):
        if used[i]:
            continue
        hi = lo = piv[i]; s = 1; used[i] = True
        for j in range(n):
            if j != i and not used[j]:
                nh, nl = max(hi, piv[j]), min(lo, piv[j])
                if nh - nl <= width:
                    hi, lo, s = nh, nl, s + 1; used[j] = True
        if s >= min_str:
            mid = (hi + lo) / 2
            k = next((k for k in range(len(cm)) if abs(mid - cm[k]) <= width), -1)
            if k >= 0:
                if s > cs[k]:
                    cm[k], cs[k], ch[k], cl[k] = mid, s, hi, lo
            else:
                cm.append(mid); cs.append(s); ch.append(hi); cl.append(lo)
    order = sorted(range(len(cm)), key=lambda k: -cs[k])  # stable, like the bubble sort
    return [(cm[k], ch[k], cl[k]) for k in order[:max_zones]]


def simulate(df, nice, P=10, entry_mode="mid", tp_n=1, sl_pct=0.15, rr=None):
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    when = df.Datetime.values
    sp = df.Spread.values * POINT[nice]
    atr = rma_atr(h, l, c, 14)
    ph, pl = pivots(h, P, True), pivots(l, P, False)
    rng = pd.Series(h).rolling(300).max().values - pd.Series(l).rolling(300).min().values
    pv, zones = [], []
    trades, misses = [], []
    pos = None; order = None
    n = len(df)
    for t in range(n):
        new = False
        if not np.isnan(ph[t]):
            pv.insert(0, ph[t]); pv = pv[:20]; new = True
        if not np.isnan(pl[t]):
            pv.insert(0, pl[t]); pv = pv[:20]; new = True
        if new:
            zones = build_zones(pv, 0.10 * rng[t]) if not np.isnan(rng[t]) and len(pv) >= 2 else []
        # ---- manage an open trade
        if pos is not None:
            side, ent, stop, tgt, risk, k0, cost = pos
            if t > k0:
                if (l[t] <= stop) if side > 0 else (h[t] >= stop):
                    R = -1.0
                elif (h[t] >= tgt) if side > 0 else (l[t] <= tgt):
                    R = (tgt - ent) * side / risk
                else:
                    R = None
                if R is not None:
                    trades.append(dict(pair=nice, side="BUY" if side > 0 else "SELL", when=when[k0], R=R - cost, Rg=R,
                                       exit_bar=t, tgt=tgt, stop=stop))
                    pos = None
            continue_after = pos is not None
            if continue_after:
                continue
        # ---- pending limit from previous close tries to fill on this bar
        if order is not None and pos is None:
            side, lim, stop, tgt, mid = order
            filled = (l[t] <= lim) if side > 0 else (h[t] >= lim)
            if filled:
                ent = min(o[t], lim) if side > 0 else max(o[t], lim)
                risk = (ent - stop) * side
                if rr is not None and risk > 0:
                    tgt = ent + side * rr * risk
                if risk > 0 and (tgt - ent) * side > 0:
                    cost = sp[t] / risk
                    if (l[t] <= stop) if side > 0 else (h[t] >= stop):
                        trades.append(dict(pair=nice, side="BUY" if side > 0 else "SELL", when=when[t], R=-1 - cost, Rg=-1.0,
                                           exit_bar=t, tgt=tgt, stop=stop))
                    else:
                        pos = (side, ent, stop, tgt, risk, t, cost)
            elif entry_mode == "mid":
                # touched the drawn box but never reached the mid?
                a = atr[t - 1]
                edge = mid + side * 0.3 * a
                if (l[t] <= edge) if side > 0 else (h[t] >= edge):
                    misses.append((t, side, mid, stop, tgt))
            order = None
        if pos is not None or not zones or np.isnan(atr[t]):
            continue
        # ---- place the next order at this bar's close
        k = min(range(len(zones)), key=lambda i: abs(c[t] - zones[i][0]))
        mid, zh, zl = zones[k]
        side = 1 if mid <= c[t] else -1
        far = sorted([z[0] for z in zones if z[0] > c[t]]) if side > 0 else sorted([z[0] for z in zones if z[0] <= c[t]], reverse=True)
        far = [m for m in far if (m - mid) * side > 0]
        if not far and rr is None:
            continue
        tgt = far[min(tp_n, len(far)) - 1] if far else np.nan
        stop = zl * (1 - sl_pct / 100) if side > 0 else zh * (1 + sl_pct / 100)
        lim = mid if entry_mode == "mid" else mid + side * 0.3 * atr[t]
        order = (side, lim, stop, tgt, mid)
    return trades, misses, (h, l)


def stats(d):
    if len(d) == 0:
        return dict(n=0)
    r = d.R.values
    eq = np.cumsum(r); dd = (np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()
    streak = cur = 0
    for x in r:
        cur = cur + 1 if x < 0 else 0; streak = max(streak, cur)
    return dict(n=len(r), win=np.mean(r > 0), avg=r.mean(), avg_g=d.Rg.mean(), tot=r.sum(), streak=streak, dd=dd)


if __name__ == "__main__":
    DATA = sys.argv[1]
    P = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    allrows = []
    for tf in ("H4", "H1"):
        for nice, f in PAIRS:
            df = pd.read_csv(f"{DATA}/{f}_{tf}.csv", parse_dates=["Datetime"])
            for variant, em, tp in (("MID_TP1", "mid", 1), ("MID_TP2", "mid", 2), ("EDGE_TP1", "edge", 1)):
                tr, misses, (h, l) = simulate(df, nice, P, em, tp)
                d = pd.DataFrame(tr)
                if len(d):
                    d["tf"] = tf; d["variant"] = variant; allrows.append(d)
                if variant == "MID_TP1":
                    # missed tip touches: would price have reached TP1 before the SL from the touch bar?
                    win = 0
                    for t, side, mid, stop, tgt in misses:
                        for k in range(t + 1, len(h)):
                            if (l[k] <= stop) if side > 0 else (h[k] >= stop):
                                break
                            if (h[k] >= tgt) if side > 0 else (l[k] <= tgt):
                                win += 1; break
                    # stopped trades that later reached TP1 within 50 bars ("stop hunt then continue")
                    lost = d[d.Rg < 0] if len(d) else d
                    rec = 0
                    for _, x in lost.iterrows():
                        s = 1 if x.side == "BUY" else -1
                        seg = slice(int(x.exit_bar) + 1, min(int(x.exit_bar) + 51, len(h)))
                        if (h[seg] >= x.tgt).any() if s > 0 else (l[seg] <= x.tgt).any():
                            rec += 1
                    print(f"DIAG {tf} {nice}: tip-touch-no-fill {len(misses)} (went on to TP1 {win}), "
                          f"stopped {len(lost)} (later hit TP1 within 50 bars {rec})", flush=True)
    out = pd.concat(allrows)
    out.to_csv(f"apex_trades_P{P}.csv", index=False)

"""LPSRZ / Linto-Apex v10 (ZoneMaster mode, V2 plan layer) backtest, default settings.

Zones: candle-BODY swing highs/lows (period 10) from the last 300 bars, sorted by price; swings
closer than 0.55 x ATR14 merged (most recent kept). 3 nearest above + 3 below close.
Zone = level +/- 0.275 ATR.
Plan direction = 1D trend (fast swings: 5-candle wick pivots on daily bars, flips on a daily
close beyond the last swing high/low), value of the last CLOSED day (no repaint).
Plan: entry = nearest zone level below close (buy) / above (sell), limit order.
Stop = beyond the NEXT zone behind (its far edge) + 0.3% of price.
TP1 = near edge of the first zone beyond price (else 2R). Skip if TP1 < 0.7R.
TP1 or a missed trade (TP reached before entry) freezes the plan until price pulls back halfway
from TP1 to entry (re-arm). Trade managed to TP1 or stop (stop first in the same candle).
Difference from the indicator's own bookkeeping: a resting limit order fills the moment price
touches entry - the indicator can replace the plan on that same candle if it closes beyond
the entry, which a real limit order would not allow.
Daily bars built from the 1H file (UTC days).
"""
import sys
import numpy as np
import pandas as pd
from apex_test import rma_atr, pivots, PAIRS, POINT, stats, SPLIT

HW = 0.55 / 2


def daily_trend(h1, P=5):
    d = h1.set_index("Datetime").resample("1D").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
    h, l, c = d.High.values, d.Low.values, d.Close.values
    ph, pl = pivots(h, P, True), pivots(l, P, False)
    lh = ll = np.nan; tr = 0; out = []
    for t in range(len(d)):
        if not np.isnan(ph[t]): lh = ph[t]
        if not np.isnan(pl[t]): ll = pl[t]
        if not np.isnan(lh) and c[t] > lh and tr != 1:
            tr = 1
        elif not np.isnan(ll) and c[t] < ll and tr != -1:
            tr = -1
        out.append(tr)
    # value usable on day D = trend at the close of day D-1
    return pd.Series(out, index=d.index).shift(1)


def zones_at(pv, pb, t, c, a):
    keep = [(v, b) for v, b in zip(pv, pb) if t - b <= 300 + 10]
    if not keep or np.isnan(a):
        return []
    keep.sort(key=lambda x: x[0])
    reps = []; last = None; rep = None; repb = -1
    for v, b in keep:
        if last is not None and v - last > 0.55 * a:
            reps.append(rep); repb = -1
        if b > repb:
            rep, repb = v, b
        last = v
    reps.append(rep)
    up0 = next((k for k, v in enumerate(reps) if v > c), len(reps))
    return reps[max(0, up0 - 3): up0 + 3]


def simulate(df, nice, tr_series, mode="with"):
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    when = df.Datetime.values
    sp = df.Spread.values * POINT[nice]
    atr = rma_atr(h, l, c, 14)
    bh, bl = np.maximum(o, c), np.minimum(o, c)
    ph, pl = pivots(bh, 10, True), pivots(bl, 10, False)
    day = df.Datetime.dt.floor("D")
    trend = day.map(tr_series).fillna(0).astype(int).values
    if mode == "against":
        trend = -trend
    pv, pb = [], []
    st = 0; pDir = 0; pEntry = np.nan; pSL = pTP = np.nan; lastExit = ""; live = None
    trades = []
    for t in range(len(c)):
        if not np.isnan(ph[t]): pv.append(ph[t]); pb.append(t - 10)
        if not np.isnan(pl[t]): pv.append(pl[t]); pb.append(t - 10)
        a = atr[t]
        # ---- live trade
        if st == 2:
            side, ent, risk, k0, cost = live
            if t > k0:
                if (l[t] <= pSL) if side > 0 else (h[t] >= pSL):
                    trades.append(dict(pair=nice, side="BUY" if side > 0 else "SELL", when=when[k0], R=-1 - cost, Rg=-1.0)); st = 3; lastExit = "SL"
                elif (h[t] >= pTP) if side > 0 else (l[t] <= pTP):
                    g = (pTP - ent) * side / risk
                    trades.append(dict(pair=nice, side="BUY" if side > 0 else "SELL", when=when[k0], R=g - cost, Rg=g)); st = 3; lastExit = "TP1"
            if st == 2:
                continue
        # ---- resting limit order from the previous close
        elif st == 1:
            touched = (l[t] <= pEntry) if pDir > 0 else (h[t] >= pEntry)
            if touched:
                ent = min(o[t], pEntry) if pDir > 0 else max(o[t], pEntry)
                risk = (ent - pSL) * pDir
                if risk <= 0:
                    trades.append(dict(pair=nice, side="BUY" if pDir > 0 else "SELL", when=when[t], R=-1.0, Rg=-1.0)); st = 3; lastExit = "SL"
                else:
                    cost = sp[t] / risk
                    if (l[t] <= pSL) if pDir > 0 else (h[t] >= pSL):
                        trades.append(dict(pair=nice, side="BUY" if pDir > 0 else "SELL", when=when[t], R=-1 - cost, Rg=-1.0)); st = 3; lastExit = "SL"
                    else:
                        live = (pDir, ent, risk, t, cost); st = 2
                        continue
            elif (h[t] >= pTP) if pDir > 0 else (l[t] <= pTP):
                st = 4
        if np.isnan(a):
            continue
        # ---- plan logic at this close (indicator rules)
        zs = zones_at(pv, pb, t, c[t], a)
        planDir = trend[t]
        cEntry = np.nan
        for m in zs:
            if planDir == 1 and m <= c[t] and (np.isnan(cEntry) or m > cEntry): cEntry = m
            if planDir == -1 and m >= c[t] and (np.isnan(cEntry) or m < cEntry): cEntry = m
        hwp = max(HW * a, 1e-9)
        frozen = (st == 3 and lastExit == "TP1") or st == 4
        alive = not np.isnan(pEntry) and any(abs(m - pEntry) <= hwp for m in zs)
        same = not np.isnan(pEntry) and pDir == planDir and alive
        want = (not np.isnan(cEntry)) and (pDir != planDir or np.isnan(pEntry) or abs(cEntry - pEntry) > hwp)
        if planDir == 0 or (np.isnan(cEntry) and not (frozen and same)):
            st = 0; pDir = planDir; pEntry = np.nan
        elif want and not (frozen and same):
            pDir = planDir; pEntry = cEntry
            edge = pEntry - hwp if pDir == 1 else pEntry + hwp
            nxt = np.nan
            for m in zs:
                if pDir == 1 and m < pEntry - hwp and (np.isnan(nxt) or m > nxt): nxt = m; edge = m - hwp
                if pDir == -1 and m > pEntry + hwp and (np.isnan(nxt) or m < nxt): nxt = m; edge = m + hwp
            pSL = edge * (1 - 0.003) if pDir == 1 else edge * (1 + 0.003)
            cands = sorted([m for m in zs if (m > c[t] and m > pEntry) if pDir == 1] if pDir == 1 else
                           [m for m in zs if m < c[t] and m < pEntry], reverse=(pDir == -1))
            pTP = (cands[0] - pDir * hwp) if cands else pEntry + pDir * 2 * abs(pEntry - pSL)
            rr = abs(pTP - pEntry) / max(abs(pEntry - pSL), 1e-12)
            if rr < 0.7:
                st = 5; pEntry = np.nan; lastExit = ""
            else:
                st = 1; lastExit = ""
        elif frozen:
            lvl = pEntry + (pTP - pEntry) * 0.5
            if (c[t] <= lvl) if pDir == 1 else (c[t] >= lvl):
                st = 1; lastExit = ""
    return trades


if __name__ == "__main__":
    DATA = sys.argv[1]; rows = []
    for nice, f in PAIRS:
        h1 = pd.read_csv(f"{DATA}/{f}_H1.csv", parse_dates=["Datetime"])
        trs = daily_trend(h1)
        for tf in ("H4", "H1"):
            df = h1 if tf == "H1" else pd.read_csv(f"{DATA}/{f}_H4.csv", parse_dates=["Datetime"])
            for mode in ("with", "against"):
                d = pd.DataFrame(simulate(df, nice, trs, mode))
                if len(d):
                    d["tf"] = tf; d["mode"] = mode; rows.append(d)
    d = pd.concat(rows); d["when"] = pd.to_datetime(d.when); d.to_csv("lpsrz_trades.csv", index=False)
    order = ["XAUUSD", "EURUSD", "GBPUSD", "GER40", "US500", "US100", "BTCUSD"]
    for tf in ("H4", "H1"):
        for mode in ("with", "against"):
            g = d[(d.tf == tf) & (d["mode"] == mode)]
            a, e, u = stats(g), stats(g[g.when < SPLIT]), stats(g[g.when >= SPLIT])
            print(f"ALL {tf} {mode} 1D trend: n {a['n']} win {a['win']:.1%} avg {a['avg']:+.3f} (before {a['avg_g']:+.3f}) early {e['avg']:+.3f} unseen {u['avg']:+.3f}  avg win size {g.Rg[g.Rg > 0].mean():.2f}R")
        print(f"\n{tf} (with 1D trend, default): pair side n win avg tot streak dd | early | unseen")
        for p in order:
            for s in ("BUY", "SELL"):
                g = d[(d.tf == tf) & (d["mode"] == "with") & (d.pair == p) & (d.side == s)].sort_values("when")
                a, e, u = stats(g), stats(g[g.when < SPLIT]), stats(g[g.when >= SPLIT])
                if a["n"] == 0: print(p, s, "none"); continue
                fe = lambda x: f"{x['avg']:+.2f} ({x['n']})" if x["n"] else "- (0)"
                print(f"{p} {s} {a['n']} {a['win']:.0%} {a['avg']:+.2f} {a['tot']:+.0f} {a['streak']} {a['dd']:.0f} | {fe(e)} | {fe(u)}")
        print()

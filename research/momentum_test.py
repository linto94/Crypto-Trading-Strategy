"""Two evidence-backed ideas, tested on the user's data (UTC timestamps), after spread costs.

A) Intraday momentum (Gao, Han, Li, Zhou 2018): the return from yesterday's 16:00 ET close to
   10:00 ET predicts the 15:30-16:00 ET return. Trade: at 15:30 go long if positive, short if
   negative, exit 16:00. Variant: only trade when 15:00-15:30 agrees.
B) Daily trend-following, equal risk across the 7 instruments:
   TSMOM-20 / TSMOM-60: position = sign of last 20 / 60 day return, held daily.
   Donchian 20/10: long on close above prior 20-day high until close below prior 10-day low (short mirror).
   Positions scaled to 1% daily volatility (20-day). Cost = spread on every position change.
   Control: 2000 random shuffles of each signal (same number of long/short days).
"""
import sys
import numpy as np
import pandas as pd

D = sys.argv[1]
POINT = {"XAUUSD": 1e-3, "EURUSD": 1e-5, "GBPUSD": 1e-5, "GER40": 1e-3, "US500": 1e-3, "US100": 1e-3, "BTCUSD": 0.1}
FILES = {"XAUUSD": "XAUUSD", "EURUSD": "EURUSD", "GBPUSD": "GBPUSD", "GER40": "DEUIDXEUR",
         "US500": "USA500IDXUSD", "US100": "USATECHIDXUSD", "BTCUSD": "BTCUSD"}


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan


def intraday(nice):
    df = pd.read_csv(f"{D}/{FILES[nice]}_M15.csv", parse_dates=["Datetime"])
    df["et"] = df.Datetime.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    df["d"] = df.et.dt.date
    df["hm"] = df.et.dt.strftime("%H:%M")
    piv = df.pivot_table(index="d", columns="hm", values="Close", aggfunc="last")
    spr = df.pivot_table(index="d", columns="hm", values="Spread", aggfunc="last") * POINT[nice]
    need = ["09:45", "14:45", "15:15", "15:45"]
    piv = piv[[c for c in need if c in piv.columns]].dropna()
    prev_close = piv["15:45"].shift(1)
    r1 = piv["09:45"] / prev_close - 1
    r12 = piv["15:15"] / piv["14:45"] - 1
    last = piv["15:45"] / piv["15:15"] - 1
    cost = (spr.reindex(piv.index)["15:15"] / piv["15:15"]).fillna(0)
    ok = r1.notna()
    s1 = np.sign(r1[ok])
    a = (s1 * last[ok] - cost[ok])
    both = ok & (np.sign(r1) == np.sign(r12))
    b = (np.sign(r1[both]) * last[both] - cost[both])
    return a, b, last[ok], (s1 * last[ok])


def daily(nice):
    df = pd.read_csv(f"{D}/{FILES[nice]}_H1.csv", parse_dates=["Datetime"])
    df["day"] = df.Datetime.dt.floor("D")
    g = df.groupby("day").agg(o=("Open", "first"), h=("High", "max"), l=("Low", "min"), c=("Close", "last"),
                              n=("Close", "size"), s=("Spread", "median"))
    g = g[g.n >= 12]  # drop thin weekend/holiday days
    g["s"] = g.s * POINT[nice] / g.c
    return g


def signals(g):
    c = g.c
    ret = c.pct_change()
    sig = {}
    sig["TSMOM-20"] = np.sign(c / c.shift(20) - 1)
    sig["TSMOM-60"] = np.sign(c / c.shift(60) - 1)
    hi20, lo20 = g.h.rolling(20).max().shift(1), g.l.rolling(20).min().shift(1)
    lo10, hi10 = g.l.rolling(10).min().shift(1), g.h.rolling(10).max().shift(1)
    pos, p = [], 0
    for i in range(len(g)):
        if p <= 0 and c.iat[i] > hi20.iat[i]:
            p = 1
        elif p >= 0 and c.iat[i] < lo20.iat[i]:
            p = -1
        elif p == 1 and c.iat[i] < lo10.iat[i]:
            p = 0
        elif p == -1 and c.iat[i] > hi10.iat[i]:
            p = 0
        pos.append(p)
    sig["Donchian 20/10"] = pd.Series(pos, index=g.index, dtype=float)
    vol = ret.rolling(20).std()
    return ret, sig, vol


def pnl(ret, pos, vol, spread):
    w = (pos / (vol / 0.01)).shift(1)            # decided at yesterday's close
    gross = w * ret
    turnover = w.diff().abs().fillna(w.abs())
    return (gross - turnover * spread).dropna(), gross.dropna()


if __name__ == "__main__":
    print("A) INTRADAY MOMENTUM (trade 15:30-16:00 New York, direction of the move to 10:00)")
    for nice in ("US500", "US100"):
        a, b, last, gross = intraday(nice)
        print(f"  {nice}: {len(a)} days | win {np.mean(a > 0):.1%} | avg {a.mean() * 1e4:+.2f} bp/day after costs "
              f"(before {gross.mean() * 1e4:+.2f}) | t={tstat(a):+.2f} | total {a.sum():+.2%}")
        print(f"         only when 15:00-15:30 agrees: {len(b)} days | win {np.mean(b > 0):.1%} | avg {b.mean() * 1e4:+.2f} bp | t={tstat(b):+.2f}")
        print(f"         (always-long baseline for that half hour: {last.mean() * 1e4:+.2f} bp/day)")

    print("\nB) DAILY TREND-FOLLOWING, 7 instruments, equal risk (1% daily vol each), after spread costs")
    rng = np.random.default_rng(7)
    data = {n: daily(n) for n in FILES}
    for name in ("TSMOM-20", "TSMOM-60", "Donchian 20/10"):
        port, port_g, per = [], [], {}
        rand_ports = np.zeros((2000,))
        rand_series = []
        for n, g in data.items():
            ret, sig, vol = signals(g)
            net, gross = pnl(ret, sig[name], vol, g.s)
            port.append(net.rename(n)); port_g.append(gross.rename(n))
            per[n] = net.sum()
        P = pd.concat(port, axis=1).fillna(0).mean(axis=1)
        PG = pd.concat(port_g, axis=1).fillna(0).mean(axis=1)
        # permutation control: shuffle each instrument's signal days, recompute portfolio mean
        sims = []
        for k in range(500):
            parts = []
            for n, g in data.items():
                ret, sig, vol = signals(g)
                s = sig[name].copy()
                valid = s.dropna()
                s.loc[valid.index] = rng.permutation(valid.values)
                parts.append(pnl(ret, s, vol, g.s)[0].rename(n))
            sims.append(pd.concat(parts, axis=1).fillna(0).mean(axis=1).mean())
        p_value = np.mean(np.array(sims) >= P.mean())
        eq = P.cumsum()
        dd = (eq - eq.cummax()).min()
        months = P.groupby(P.index.to_period("M")).sum()
        ann = P.mean() * 252
        sharpe = P.mean() / P.std() * np.sqrt(252)
        print(f"  {name:15s} days {len(P)} | yearly return {ann:+.1%} (before costs {PG.mean() * 252:+.1%}) | Sharpe {sharpe:+.2f} | "
              f"worst drawdown {dd:.1%} | winning months {np.mean(months > 0):.0%} | beats random shuffles: p={p_value:.2f}")
        print(f"                  per instrument (year total): " + ", ".join(f"{n} {v:+.1%}" for n, v in per.items()))

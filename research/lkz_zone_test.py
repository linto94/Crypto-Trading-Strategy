"""Replicates Linto Key Zones (LKZ) zone logic and tests whether price reacts at
its zones more often than at random levels of the same width.

Event: a bar enters a displayed zone after the previous bar was fully outside it.
Outcome (from the next bar, up to MAXF bars): BOUNCE if price moves K x ATR away
from the near edge in the reversal direction before it moves K x ATR beyond the far
edge (BREAK). Same-bar ties count as BREAK (conservative).
"""
import sys
import numpy as np
import pandas as pd

DATA = sys.argv[1]
K = 1.0
MAXF = 100


def load(name):
    df = pd.read_csv(f"{DATA}/{name}.csv", parse_dates=["Datetime"])
    return df.reset_index(drop=True)


def rma_atr(h, l, c, n):
    tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
    tr[0] = h[0] - l[0]
    out = np.full(len(tr), np.nan)
    if len(tr) < n:
        return out
    out[n - 1] = tr[:n].mean()
    for i in range(n, len(tr)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def run_zones(df, L, mult, atr_len, min_piv=3, max_piv=40, per_side=2, use_broken=True):
    """Returns list per bar t of displayed zones [(top, bot, cnt)] known BEFORE bar t."""
    h, l, c = df.High.values, df.Low.values, df.Close.values
    n = len(df)
    atr = rma_atr(h, l, c, atr_len)
    p_price, p_bar = [], []
    zones = []  # dicts: top, bot, cnt, above, below
    shown = [None] * n
    for i in range(n):
        new = False
        j = i - L
        if j - L >= 0:
            win_h = h[j - L:i + 1]
            if h[j] == win_h.max() and h[j] > h[j - L:j].max():
                p_price.insert(0, h[j]); p_bar.insert(0, j); new = True
            win_l = l[j - L:i + 1]
            if l[j] == win_l.min() and l[j] < l[j - L:j].min():
                p_price.insert(0, l[j]); p_bar.insert(0, j); new = True
        del p_price[max_piv:]; del p_bar[max_piv:]
        if new and not np.isnan(atr[i]):
            width = mult * atr[i]
            m = len(p_price)
            zones = []
            if m >= min_piv:
                used = [False] * m
                cand = []
                for a in range(m):
                    if used[a]:
                        continue
                    hi = lo = p_price[a]; cnt = 1; newest = p_bar[a]; used[a] = True
                    for b in range(m):
                        if b != a and not used[b]:
                            v = p_price[b]
                            nh, nl = max(hi, v), min(lo, v)
                            if nh - nl <= width:
                                hi, lo, cnt = nh, nl, cnt + 1
                                newest = max(newest, p_bar[b]); used[b] = True
                    if cnt >= min_piv:
                        mid = (hi + lo) / 2
                        ov = next((k for k, z in enumerate(cand) if abs(mid - (z[0] + z[1]) / 2) <= width), -1)
                        if ov >= 0:
                            if cnt > cand[ov][2]:
                                cand[ov] = (hi, lo, cnt, newest)
                        else:
                            cand.append((hi, lo, cnt, newest))
                minT = 0.1 * atr[i]
                for hi, lo, cnt, newest in cand:
                    if hi - lo < minT:
                        md = (hi + lo) / 2; hi, lo = md + minT / 2, md - minT / 2
                    seg = c[max(newest, i - 999):i + 1]
                    zones.append(dict(top=hi, bot=lo, cnt=cnt, above=bool((seg > hi).any()), below=bool((seg < lo).any())))
        else:
            for z in zones:
                if c[i] > z["top"]: z["above"] = True
                if c[i] < z["bot"]: z["below"] = True
        # zones displayed for the NEXT bar, relative to this bar's close
        if i + 1 < n:
            live = [z for z in zones if not (use_broken and z["above"] and z["below"])]
            up = sorted([z for z in live if (z["top"] + z["bot"]) / 2 >= c[i]], key=lambda z: (max(z["bot"] - c[i], 0), -z["cnt"]))[:per_side]
            dn = sorted([z for z in live if (z["top"] + z["bot"]) / 2 < c[i]], key=lambda z: (max(c[i] - z["top"], 0), -z["cnt"]))[:per_side]
            shown[i + 1] = [(z["top"], z["bot"], z["cnt"]) for z in up + dn]
    return shown, atr


def outcome(h, l, t, side, near, far, a):
    """side=+1 support (expect up), -1 resistance (expect down)."""
    up_t = near + K * a if side > 0 else near - K * a
    br_t = far - K * a if side > 0 else far + K * a
    for k in range(t + 1, min(t + 1 + MAXF, len(h))):
        hit_b = h[k] >= up_t if side > 0 else l[k] <= up_t
        hit_x = l[k] <= br_t if side > 0 else h[k] >= br_t
        if hit_x:
            return 0
        if hit_b:
            return 1
    return None


def events_for(df, shown, atr):
    h, l = df.High.values, df.Low.values
    res, widths, last = [], [], {}
    for t in range(1, len(df)):
        if shown[t] is None or np.isnan(atr[t - 1]):
            continue
        for top, bot, cnt in shown[t]:
            if not (h[t] >= bot and l[t] <= top):
                continue
            if l[t - 1] > top:
                side, near, far = 1, top, bot
            elif h[t - 1] < bot:
                side, near, far = -1, bot, top
            else:
                continue
            key = round((top + bot) / 2 / max(atr[t - 1], 1e-12))
            if t - last.get(key, -99) < 5:
                continue
            last[key] = t
            o = outcome(h, l, t, side, near, far, atr[t - 1])
            if o is not None:
                res.append(o); widths.append((top - bot) / atr[t - 1])
    return res, widths


def random_events(df, atr, width_atr, n_tries=4000, seed=1):
    rng = np.random.default_rng(seed)
    h, l, c = df.High.values, df.Low.values, df.Close.values
    res = []
    starts = rng.integers(200, len(df) - 200, n_tries)
    for s in starts:
        a = atr[s]
        if np.isnan(a):
            continue
        dist = rng.uniform(0.5, 4.0) * a * rng.choice([-1, 1])
        w = width_atr * a
        if dist > 0:
            bot, top = c[s] + dist, c[s] + dist + w
        else:
            top, bot = c[s] + dist, c[s] + dist - w
        for t in range(s + 1, min(s + 300, len(df))):
            if h[t] >= bot and l[t] <= top:
                if l[t - 1] > top:
                    o = outcome(h, l, t, 1, top, bot, atr[t - 1])
                elif h[t - 1] < bot:
                    o = outcome(h, l, t, -1, bot, top, atr[t - 1])
                else:
                    o = None
                if o is not None:
                    res.append(o)
                break
    return res


PAIRS = [("XAUUSD", "XAUUSD"), ("EURUSD", "EURUSD"), ("GBPUSD", "GBPUSD"), ("GER40", "DEUIDXEUR"),
         ("US500", "USA500IDXUSD"), ("US100", "USATECHIDXUSD"), ("BTCUSD", "BTCUSD")]

if __name__ == "__main__":
    mode = sys.argv[2] if len(sys.argv) > 2 else "base"
    tf = sys.argv[3] if len(sys.argv) > 3 else "H1"
    configs = {
        "H1": dict(L=5, mult=0.5, atr_len=24),
        "M15": dict(L=10, mult=1.0, atr_len=96),
    }
    variants = {"base": [dict()],
                "sens": [dict(min_piv=2), dict(min_piv=3), dict(min_piv=4),
                         dict(mult_scale=0.6), dict(mult_scale=1.6), dict(use_broken=False)]}[mode]
    for v in variants:
        cfg = dict(configs[tf])
        if "mult_scale" in v:
            cfg["mult"] *= v.pop("mult_scale"); label = f"width x{cfg['mult']}"
        else:
            label = ", ".join(f"{k}={x}" for k, x in v.items()) or "default"
        cfg.update(v)
        tz = tr = 0
        rows = []
        for nice, f in PAIRS:
            df = load(f"{f}_{tf}")
            shown, atr = run_zones(df, **cfg)
            ev, w = events_for(df, shown, atr)
            rnd = random_events(df, atr, np.median(w) if w else 0.5)
            rows.append((nice, len(ev), np.mean(ev) if ev else float("nan"), len(rnd), np.mean(rnd) if rnd else float("nan"),
                         len(ev) / (len(df) / (24 if tf == "H1" else 96))))
            tz += sum(ev); tr += sum(rnd)
        ne = sum(r[1] for r in rows); nr = sum(r[3] for r in rows)
        print(f"\n[{tf}] {label}")
        for r in rows:
            print(f"  {r[0]:7s} zone touches {r[1]:4d}  bounce {r[2]:.0%}   | random {r[3]:4d}  bounce {r[4]:.0%}   | touches/day {r[5]:.2f}")
        pz, pr = tz / ne, tr / nr
        se = np.sqrt(pz * (1 - pz) / ne + pr * (1 - pr) / nr)
        print(f"  ALL     zone touches {ne:4d}  bounce {pz:.1%}   | random {nr:4d}  bounce {pr:.1%}   | diff {pz - pr:+.1%} (z={(pz - pr) / se:.1f})")

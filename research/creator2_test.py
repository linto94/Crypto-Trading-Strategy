"""Creator 2: untested (virgin) swing levels.
Test A (magnet): at sample times, take the nearest untested swing above and below price.
  Race: does price reach the swing before a mirror level the same distance on the other side?
  No effect = 50%.
Test B (reaction): first touch of an untested swing -> bounce vs the same level shifted
  +/-1.5, +/-3 ATR (same method as the other level tests).
Timeframes: 15M (L=10), 1H (L=10), 4H (L=5, built from 1H).
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, sys.path[0])
from lkz_zone_test import load, rma_atr, outcome, PAIRS

HALF = 0.1


def resample_h4(df):
    g = df.set_index("Datetime").resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    return g.reset_index()


def fresh_levels(h, l, L):
    """per bar t: list of untested swing prices known before t."""
    n = len(h)
    out = [None] * n
    fresh = []
    for t in range(1, n):
        j = t - 1 - L
        if j - L >= 0:
            if h[j] == h[j - L:t].max() and h[j] > h[j - L:j].max():
                fresh.append(h[j])
            if l[j] == l[j - L:t].min() and l[j] < l[j - L:j].min():
                fresh.append(l[j])
        out[t] = list(fresh)
        fresh = [x for x in fresh if not (l[t] <= x <= h[t])]
        fresh = fresh[-40:]
    return out


def magnet(h, l, c, atr, lv, every, horizon):
    wins = tot = 0
    for t in range(1, len(h) - 1, every):
        if not lv[t] or np.isnan(atr[t - 1]):
            continue
        px, a = c[t - 1], atr[t - 1]
        ups = [x for x in lv[t] if x > px + 0.5 * a]
        dns = [x for x in lv[t] if x < px - 0.5 * a]
        for tgt in ([min(ups)] if ups else []) + ([max(dns)] if dns else []):
            d = tgt - px
            mirror = px - d
            for k in range(t, min(t + horizon, len(h))):
                hit_t = (h[k] >= tgt) if d > 0 else (l[k] <= tgt)
                hit_m = (l[k] <= mirror) if d > 0 else (h[k] >= mirror)
                if hit_t and hit_m:
                    break
                if hit_t:
                    wins += 1; tot += 1; break
                if hit_m:
                    tot += 1; break
    return wins, tot


def reaction(h, l, atr, lv, shift):
    res, last = [], {}
    for t in range(1, len(h)):
        a = atr[t - 1]
        if not lv[t] or np.isnan(a):
            continue
        for x in lv[t]:
            x = x + shift * a
            top, bot = x + HALF * a, x - HALF * a
            if not (h[t] >= bot and l[t] <= top):
                continue
            if l[t - 1] > top:
                side, near, far = 1, top, bot
            elif h[t - 1] < bot:
                side, near, far = -1, bot, top
            else:
                continue
            key = round(x / a * 2)
            if t - last.get(key, -99) < 5:
                continue
            last[key] = t
            o = outcome(h, l, t, side, near, far, a)
            if o is not None:
                res.append(o)
    return res


if __name__ == "__main__":
    import lkz_zone_test
    lkz_zone_test.DATA = sys.argv[1]
    specs = [("15M", "M15", 10, 96, 16, 192), ("1H", "H1", 10, 24, 4, 96), ("4H", "H4", 5, 6, 1, 30)]
    for name, tf, L, atrn, every, horizon in specs:
        mw = mt = 0
        rr = rc = []
        r_s = r_n = c_s = c_n = 0
        for nice, f in PAIRS:
            df = load(f"{f}_H1") if tf == "H4" else load(f"{f}_{tf}")
            if tf == "H4":
                df = resample_h4(df)
            h, l, c = df.High.values, df.Low.values, df.Close.values
            atr = rma_atr(h, l, c, atrn)
            lv = fresh_levels(h, l, L)
            w, t = magnet(h, l, c, atr, lv, every, horizon)
            mw += w; mt += t
            real = reaction(h, l, atr, lv, 0.0)
            ctrl = sum((reaction(h, l, atr, lv, s) for s in (-3, -1.5, 1.5, 3)), [])
            r_s += sum(real); r_n += len(real); c_s += sum(ctrl); c_n += len(ctrl)
        p = mw / mt
        print(f"{name}: MAGNET  untested level reached first {p:.1%} of {mt} races (50% = no effect, z={(p - 0.5) / np.sqrt(0.25 / mt):.1f})")
        pr, pc = r_s / r_n, c_s / c_n
        se = np.sqrt(pr * (1 - pr) / r_n + pc * (1 - pc) / c_n)
        print(f"{name}: REACTION first touch bounce {pr:.1%} of {r_n}  vs shifted {pc:.1%}  diff {pr - pc:+.1%} (z={(pr - pc) / se:.1f})")

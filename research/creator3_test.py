"""Creator 3 (OTZ): break-and-retest of a level tested as BOTH support and resistance.
Zones: clusters of swing highs AND lows (width 0.5 x ATR), >= MINP pivots, both types present.
Event: price was on side O of the zone, closed through to side S (break), and now comes back
to the zone for the FIRST time from side S. Expected: bounce back into side S (the flip).
Control: identical break-and-retest logic on the same zones shifted +/-1.5 and +/-3 ATR.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, sys.path[0])
from lkz_zone_test import load, rma_atr, outcome, PAIRS
from creator2_test import resample_h4

LOOKBACK = 120


def mixed_zones(df, L, mult, atr_len, minp, max_piv=40):
    h, l, c = df.High.values, df.Low.values, df.Close.values
    atr = rma_atr(h, l, c, atr_len)
    piv = []  # (price, type) newest first
    zones = []
    out = [None] * len(df)
    for i in range(len(df)):
        new = False
        j = i - L
        if j - L >= 0:
            if h[j] == h[j - L:i + 1].max() and h[j] > h[j - L:j].max():
                piv.insert(0, (h[j], "H")); new = True
            if l[j] == l[j - L:i + 1].min() and l[j] < l[j - L:j].min():
                piv.insert(0, (l[j], "L")); new = True
        del piv[max_piv:]
        if new and not np.isnan(atr[i]):
            w = mult * atr[i]
            used = [False] * len(piv)
            cand = []
            for a in range(len(piv)):
                if used[a]:
                    continue
                hi = lo = piv[a][0]; types = {piv[a][1]}; cnt = 1; used[a] = True
                for b in range(len(piv)):
                    if b != a and not used[b]:
                        nh, nl = max(hi, piv[b][0]), min(lo, piv[b][0])
                        if nh - nl <= w:
                            hi, lo, cnt = nh, nl, cnt + 1; types.add(piv[b][1]); used[b] = True
                if cnt >= minp and len(types) == 2:
                    mid = (hi + lo) / 2
                    ov = next((k for k, z in enumerate(cand) if abs(mid - (z[0] + z[1]) / 2) <= w), -1)
                    if ov >= 0:
                        if cnt > cand[ov][2]:
                            cand[ov] = (hi, lo, cnt)
                    else:
                        cand.append((hi, lo, cnt))
            mt = 0.1 * atr[i]
            zones = [((hi + lo) / 2 + mt / 2, (hi + lo) / 2 - mt / 2) if hi - lo < mt else (hi, lo) for hi, lo, _ in cand]
        if i + 1 < len(df):
            out[i + 1] = list(zones)
    return out, atr


def retest_events(h, l, c, atr, zpb, shift):
    res, last = [], {}
    n = len(h)
    for t in range(2, n):
        a = atr[t - 1]
        if not zpb[t] or np.isnan(a):
            continue
        for top, bot in zpb[t]:
            top, bot = top + shift * a, bot + shift * a
            if not (h[t] >= bot and l[t] <= top):
                continue
            if l[t - 1] > top:
                side = 1      # came back from above -> broken resistance, now support
            elif h[t - 1] < bot:
                side = -1     # came back from below -> broken support, now resistance
            else:
                continue
            # walk back: bars t-1 .. b must not touch the zone; before the break, a close on the other side
            k = t - 1
            while k > t - LOOKBACK and k > 0 and not (h[k] >= bot and l[k] <= top):
                k -= 1
            if k <= t - LOOKBACK or k <= 0:
                continue
            # bar k touched the zone: it must be the break bar (closes on side `side`) coming from the other side
            if not ((side == 1 and c[k] > top) or (side == -1 and c[k] < bot)):
                continue
            m = k - 1
            while m > t - LOOKBACK and m > 0 and bot <= c[m] <= top:
                m -= 1
            if not ((side == 1 and c[m] < bot) or (side == -1 and c[m] > top)):
                continue
            key = round((top + bot) / 2 / a)
            if t - last.get(key, -99) < 5:
                continue
            last[key] = t
            near, far = (top, bot) if side == 1 else (bot, top)
            o = outcome(h, l, t, side, near, far, a)
            if o is not None:
                res.append(o)
    return res


if __name__ == "__main__":
    import lkz_zone_test
    lkz_zone_test.DATA = sys.argv[1]
    for name, tf, L, atrn in [("1H", "H1", 5, 24), ("4H", "H4", 5, 6), ("15M", "M15", 10, 96)]:
        for minp in (2, 3):
            rs = rn = cs = cn = 0
            better = 0
            for nice, f in PAIRS:
                df = load(f"{f}_H1") if tf == "H4" else load(f"{f}_{tf}")
                if tf == "H4":
                    df = resample_h4(df)
                h, l, c = df.High.values, df.Low.values, df.Close.values
                zpb, atr = mixed_zones(df, L, 0.5 if tf != "M15" else 1.0, atrn, minp)
                real = retest_events(h, l, c, atr, zpb, 0.0)
                ctrl = sum((retest_events(h, l, c, atr, zpb, s) for s in (-3, -1.5, 1.5, 3)), [])
                rs += sum(real); rn += len(real); cs += sum(ctrl); cn += len(ctrl)
                if real and ctrl and np.mean(real) > np.mean(ctrl):
                    better += 1
            pr, pc = rs / rn, cs / cn
            se = np.sqrt(pr * (1 - pr) / rn + pc * (1 - pc) / cn)
            print(f"{name} min {minp} pivots: first retest after break, flip held {pr:.1%} of {rn}  | made-up levels {pc:.1%} of {cn}  | diff {pr - pc:+.1%} (z={(pr - pc) / se:.1f})  pairs better {better}/7")

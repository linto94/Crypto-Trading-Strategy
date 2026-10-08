"""FVG, order blocks, supply/demand: first return to the zone -> bounce in the zone's direction?
Control: identical zones shifted +/-1.5 and +/-3 ATR, same first-touch rule and expected side.
Outcome identical to the other tests (1 ATR away from near edge before 1 ATR past far edge).
"""
import sys
import numpy as np
sys.path.insert(0, sys.path[0])
from lkz_zone_test import load, rma_atr, outcome, PAIRS

EXPIRE = 200


def make_zones(o, h, l, c, atr, L):
    """list of (created_bar, top, bot, dir, kind); dir +1 bullish (expect up on return from above)."""
    n = len(c)
    zs = []
    last_sh = last_sl = None
    broke_h = broke_l = True
    for i in range(2, n):
        a = atr[i]
        if np.isnan(a):
            continue
        # FVG
        if l[i] > h[i - 2] and l[i] - h[i - 2] >= 0.1 * a:
            zs.append((i, l[i], h[i - 2], 1, "FVG"))
        if h[i] < l[i - 2] and l[i - 2] - h[i] >= 0.1 * a:
            zs.append((i, l[i - 2], h[i], -1, "FVG"))
        # swings for order blocks (confirmed L bars later)
        j = i - L
        if j - L >= 0:
            if h[j] == h[j - L:i + 1].max() and h[j] > h[j - L:j].max():
                last_sh, broke_h = h[j], False
            if l[j] == l[j - L:i + 1].min() and l[j] < l[j - L:j].min():
                last_sl, broke_l = l[j], False
        if last_sh is not None and not broke_h and c[i] > last_sh:
            broke_h = True
            for k in range(i - 1, max(i - 11, 0), -1):
                if c[k] < o[k]:
                    zs.append((i, h[k], l[k], 1, "Order block")); break
        if last_sl is not None and not broke_l and c[i] < last_sl:
            broke_l = True
            for k in range(i - 1, max(i - 11, 0), -1):
                if c[k] > o[k]:
                    zs.append((i, h[k], l[k], -1, "Order block")); break
        # supply / demand: big departure candle after 1-3 small base candles
        rng, body = h[i] - l[i], abs(c[i] - o[i])
        if rng >= 2 * a and body >= 0.7 * rng:
            base = []
            for k in range(i - 1, i - 4, -1):
                if h[k] - l[k] < 0.7 * a:
                    base.append(k)
                else:
                    break
            if base:
                top, bot = max(h[k] for k in base), min(l[k] for k in base)
                zs.append((i, top, bot, 1 if c[i] > o[i] else -1, "Supply/demand"))
    return zs


def first_return(h, l, atr, zs, shift):
    out = {}
    for (i, top, bot, d, kind) in zs:
        a0 = atr[i]
        top, bot = top + shift * a0, bot + shift * a0
        if top - bot < 0.1 * a0:
            m = (top + bot) / 2; top, bot = m + 0.05 * a0, m - 0.05 * a0
        for t in range(i + 1, min(i + EXPIRE, len(h))):
            if h[t] >= bot and l[t] <= top:
                ok = (d == 1 and l[t - 1] > top) or (d == -1 and h[t - 1] < bot)
                if ok:
                    near, far = (top, bot) if d == 1 else (bot, top)
                    r = outcome(h, l, t, d, near, far, atr[t - 1])
                    if r is not None:
                        out.setdefault(kind, []).append(r)
                break
    return out


if __name__ == "__main__":
    import lkz_zone_test
    lkz_zone_test.DATA = sys.argv[1]
    for name, tf, atrn, L in [("15M", "M15", 96, 5), ("1H", "H1", 24, 5)]:
        agg = {}
        for nice, f in PAIRS:
            df = load(f"{f}_{tf}")
            o, h, l, c = df.Open.values, df.High.values, df.Low.values, df.Close.values
            atr = rma_atr(h, l, c, atrn)
            zs = make_zones(o, h, l, c, atr, L)
            real = first_return(h, l, atr, zs, 0.0)
            ctrl = {}
            for s in (-3, -1.5, 1.5, 3):
                for k, v in first_return(h, l, atr, zs, s).items():
                    ctrl.setdefault(k, []).extend(v)
            for k in real:
                a = agg.setdefault(k, [0, 0, 0, 0, 0])
                a[0] += sum(real[k]); a[1] += len(real[k]); a[2] += sum(ctrl.get(k, [])); a[3] += len(ctrl.get(k, []))
                if ctrl.get(k) and np.mean(real[k]) > np.mean(ctrl[k]):
                    a[4] += 1
        for k, (rs, rn, cs, cn, b) in agg.items():
            pr, pc = rs / rn, cs / cn
            se = np.sqrt(pr * (1 - pr) / rn + pc * (1 - pc) / cn)
            print(f"{name} {k:14s} held {pr:.1%} of {rn:5d} | made-up {pc:.1%} of {cn:6d} | diff {pr - pc:+.1%} (z={(pr - pc) / se:.1f}) pairs better {b}/7")

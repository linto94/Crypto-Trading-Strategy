"""Tests two ideas from the price-action course on LKZ zone touches.

TEST 1 - approach type (bounce vs break, same outcome rule as the other tests):
  power move : price travelled > 2.5 ATR in the last 6 bars into the zone, with no
               counter-move of more than 0.5 ATR on the way
  stair-step : 2+ rising swing lows into resistance (falling swing highs into support)
               in the last 20 bars, and a slow approach (< 1.5 ATR in 6 bars)
  repeated   : the zone was already touched 2+ times in the last 24 bars
  build-up   : the last 4 bars were small (< 0.6 ATR) and closed within 0.5 ATR of the zone
  Course claim: power move -> bounce; stair-step / repeated / build-up -> break.

TEST 2 - pullback in trend (full trade simulation, after spread costs):
  trend      : close and EMA50 both above EMA200 (up) / both below (down)
  setup      : price pulls back into a support zone in an uptrend (resistance zone in a downtrend)
  trigger    : hammer or engulfing candle within 3 bars of the touch
  entry      : trigger close.  stop: pullback extreme beyond the zone - 1 ATR.  target 1.5R.
  compared   : with-trend vs against-trend, real zones vs made-up (shifted) zones,
               train (Oct-Jun) vs unseen test (Jun-Oct).
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, sys.path[0])
from lkz_zone_test import load, rma_atr, outcome, run_zones, PAIRS
from trade_sim import run_trade, POINT
from confluence_test import ema

SPLIT = pd.Timestamp("2026-06-08")


def swing_lows(l, a, b):
    return [l[k] for k in range(a + 2, b - 2) if l[k] < min(l[k - 2:k].min(), l[k + 1:k + 3].min())]


def swing_highs(h, a, b):
    return [h[k] for k in range(a + 2, b - 2) if h[k] > max(h[k - 2:k].max(), h[k + 1:k + 3].max())]


def hammer(o, h, l, c, k, side):
    body = abs(c[k] - o[k]); rng = h[k] - l[k]
    if rng <= 0:
        return False
    if side == 1:
        return (min(o[k], c[k]) - l[k]) >= 2 * max(body, 1e-12) and c[k] >= l[k] + 0.5 * rng
    return (h[k] - max(o[k], c[k])) >= 2 * max(body, 1e-12) and c[k] <= h[k] - 0.5 * rng


def engulf(o, c, k, side):
    if side == 1:
        return c[k] > o[k] and c[k - 1] < o[k - 1] and c[k] >= o[k - 1] and o[k] <= c[k - 1]
    return c[k] < o[k] and c[k - 1] > o[k - 1] and c[k] <= o[k - 1] and o[k] >= c[k - 1]


def run(df, nice, cfg):
    o, h, l, c = df.Open.values, df.High.values, df.Low.values, df.Close.values
    sp = df.Spread.values * POINT[nice]
    when = df.Datetime.values
    shown, atr = run_zones(df, **cfg)
    e50, e200 = ema(c, 50), ema(c, 200)
    ev, tr = [], []
    for shift in (0.0, -3.0, -1.5, 1.5, 3.0):
        last = {}
        touches = {}
        for t in range(30, len(c) - 2):
            zs = shown[t]; a = atr[t - 1]
            if not zs or np.isnan(a):
                continue
            for top, bot, cnt in zs:
                top, bot = top + shift * a, bot + shift * a
                if not (h[t] >= bot and l[t] <= top):
                    continue
                side = 1 if l[t - 1] > top else -1 if h[t - 1] < bot else 0   # 1 = support (came from above)
                if side == 0:
                    continue
                key = round((top + bot) / 2 / a)
                prev = [x for x in touches.get(key, []) if 5 <= t - x <= 48]
                touches.setdefault(key, []).append(t)
                if t - last.get(key, -99) < 5:
                    continue
                last[key] = t
                real = shift == 0.0
                # ---------- TEST 1 features
                travel = (c[t - 7] - c[t - 1]) * side / a            # distance moved into the zone
                if side == 1:   # falling into support: counter-move = biggest bounce on the way down
                    counter = max(h[k] - l[t - 6:k].min() for k in range(t - 5, t))   # bounce above an earlier low
                else:           # rising into resistance: biggest dip on the way up
                    counter = max(h[t - 6:k].max() - l[k] for k in range(t - 5, t))   # dip below an earlier high
                power = travel > 2.0 and counter < 0.75 * a
                if side == -1:
                    sw = swing_lows(l, t - 20, t)
                    stair = len(sw) >= 2 and all(sw[i] < sw[i + 1] for i in range(len(sw) - 1)) and travel < 1.5
                else:
                    sw = swing_highs(h, t - 20, t)
                    stair = len(sw) >= 2 and all(sw[i] > sw[i + 1] for i in range(len(sw) - 1)) and travel < 1.5
                near = top if side == 1 else bot
                build = all(h[k] - l[k] < 0.6 * a for k in range(t - 4, t)) and abs(c[t - 1] - near) < 0.5 * a
                res = outcome(h, l, t, side, top if side == 1 else bot, bot if side == 1 else top, a)
                if res is not None:
                    ev.append(dict(real=real, win=res, power=power, stair=stair, repeated=len(prev) >= 1, build=build))
                # ---------- TEST 2 pullback trade
                up = c[t - 1] > e200[t - 1] and e50[t - 1] > e200[t - 1]
                dn = c[t - 1] < e200[t - 1] and e50[t - 1] < e200[t - 1]
                if not (up or dn):
                    continue
                with_trend = (side == 1 and up) or (side == -1 and dn)
                ext = l[t] if side == 1 else h[t]
                for u in range(t, min(t + 3, len(c) - 1)):
                    ext = min(ext, l[u]) if side == 1 else max(ext, h[u])
                    if (side == 1 and ext < bot - a) or (side == -1 and ext > top + a):
                        break
                    if hammer(o, h, l, c, u, side) or engulf(o, c, u, side):
                        stop = (min(ext, bot) - a) if side == 1 else (max(ext, top) + a)
                        r = run_trade(h, l, c, u, side, c[u], stop, 24)
                        if r:
                            R, risk = r
                            tr.append(dict(real=real, with_trend=with_trend, when=when[u], R=R - sp[u] / risk, Rg=R))
                        break
    return ev, tr


def line(label, d):
    if len(d) < 20:
        print(f"  {label:42s} too few ({len(d)})"); return
    m = d.R.mean()
    print(f"  {label:42s} trades {len(d):5d}  win {np.mean(d.R > 0):5.1%}  avg {m:+.3f}R after costs ({d.Rg.mean():+.3f} before)  t={m / (d.R.std() / np.sqrt(len(d))):+.1f}")


if __name__ == "__main__":
    import lkz_zone_test
    lkz_zone_test.DATA = sys.argv[1]
    for name, tf, cfg in [("15M", "M15", dict(L=10, mult=1.0, atr_len=96)), ("1H", "H1", dict(L=5, mult=0.5, atr_len=24))]:
        E, T = [], []
        for nice, f in PAIRS:
            e, t = run(load(f"{f}_{tf}"), nice, cfg)
            E += e; T += [dict(x, pair=nice) for x in t]
        E, T = pd.DataFrame(E), pd.DataFrame(T)
        T["when"] = pd.to_datetime(T.when)
        print(f"\n===== {name}  TEST 1: bounce rate by approach type (real zones | made-up zones)")
        base = E[E.real].win.mean()
        print(f"  all touches                          real {base:5.1%} (n={E.real.sum()})  | made-up {E[~E.real].win.mean():5.1%}")
        for f_, lab in [("power", "power move in (course: bounce)"), ("stair", "stair-step in (course: break)"),
                        ("repeated", "re-tested within 48 bars (course: break)"), ("build", "build-up at zone (course: break)")]:
            for val in (True, False):
                g = E[E[f_] == val]
                r, cc = g[g.real], g[~g.real]
                if len(r) < 20:
                    continue
                print(f"  {lab if val else '  not ' + f_:44s} real {r.win.mean():5.1%} (n={len(r):5d}) | made-up {cc.win.mean():5.1%} (n={len(cc):5d})")
        print(f"\n===== {name}  TEST 2: pullback-in-trend trades at zones (1.5R, after costs)")
        for wt in (True, False):
            for real in (True, False):
                g = T[(T.with_trend == wt) & (T.real == real)]
                lab = ("WITH trend" if wt else "against trend") + (", real zone" if real else ", made-up zone")
                line(lab + "  [train]", g[g.when < SPLIT])
                line(lab + "  [unseen test]", g[g.when >= SPLIT])

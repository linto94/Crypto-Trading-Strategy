"""Linto-Apex variant: SL buffer 0.3% beyond the cluster edge, fixed 1.5R / 2R target."""
import sys, pandas as pd
from apex_test import simulate, PAIRS
DATA = sys.argv[1]; rows = []
for tf in ("H4", "H1"):
    for nice, f in PAIRS:
        df = pd.read_csv(f"{DATA}/{f}_{tf}.csv", parse_dates=["Datetime"])
        for em in ("mid", "edge"):
            for sl in (0.15, 0.3):
                for rr in (1.5, 2.0):
                    tr, _, _ = simulate(df, nice, 10, em, 1, sl, rr)
                    d = pd.DataFrame(tr); d["tf"] = tf; d["variant"] = f"{em.upper()}_SL{sl}_RR{rr}"; rows.append(d)
pd.concat(rows).to_csv("apex_rr_trades.csv", index=False)

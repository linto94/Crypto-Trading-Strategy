import sys, numpy as np, pandas as pd
from apex_test import stats, SPLIT
d = pd.read_csv(sys.argv[1], parse_dates=["when"])
order = ["XAUUSD","EURUSD","GBPUSD","GER40","US500","US100","BTCUSD"]
for var in ["MID_TP1","MID_TP2","EDGE_TP1"]:
    v = d[d.variant == var]
    print(f"\n######## {var}")
    for tf in ["H4","H1"]:
        print(f"--- {tf}   [all 5y]  n win avgR(after) avgR(before) totR streak maxDD || early(21-24) n avg tot || unseen(25-26) n avg tot")
        for p in order:
            for s in ["BUY","SELL"]:
                g = v[(v.tf==tf)&(v.pair==p)&(v.side==s)].sort_values("when")
                a, e, u = stats(g), stats(g[g.when<SPLIT]), stats(g[g.when>=SPLIT])
                if a["n"]==0: print(p,s,"none"); continue
                f = lambda x: f"{x['n']:4d} {x['avg']:+.2f} {x['tot']:+7.1f}" if x["n"] else "   0"
                print(f"{p:7s} {s:4s} {a['n']:4d} {a['win']:5.1%} {a['avg']:+.3f} {a['avg_g']:+.3f} {a['tot']:+7.1f} {a['streak']:3d} {a['dd']:6.1f} || {f(e)} || {f(u)}")
        g = v[v.tf==tf]; a=stats(g)
        print(f"ALL {tf}: n {a['n']} win {a['win']:.1%} avg {a['avg']:+.3f} (before {a['avg_g']:+.3f}) tot {a['tot']:+.1f}; avg RR to target {((g.Rg[g.Rg>0]).mean()):.2f}")

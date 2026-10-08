import sys, numpy as np
sys.argv=[sys.argv[0], sys.argv[1]]
from lkz_zone_test import *
tf=sys.argv[0] and "H1"
for tf,cfg in [("H1",dict(L=5,mult=0.5,atr_len=24)),("M15",dict(L=10,mult=1.0,atr_len=96))]:
    tot={}
    for nice,f in PAIRS:
        df=load(f"{f}_{tf}"); shown,atr=run_zones(df,**cfg)
        for d in (0,-1.5,1.5,-3,3):
            sh=[None if s is None else [(a+d*atr[i-1],b+d*atr[i-1],c) for a,b,c in s] for i,s in enumerate(shown)]
            ev,_=events_for(df,sh,atr); x=tot.setdefault(d,[0,0]); x[0]+=sum(ev); x[1]+=len(ev)
    print(tf, {d:(f"{v[0]/v[1]:.1%}", v[1]) for d,v in tot.items()})

"""Strategy A (SWEEP) research run: parameter grid, walk-forward OOS, cost stress, matched-random
null, breakdowns, and (only with --lockbox) the one-shot lockbox evaluation.

    python scripts/run_strategy_a.py                 # walk-forward on pre-lockbox data
    python scripts/run_strategy_a.py --lockbox       # ALSO evaluate the lockbox (run once!)
    python scripts/run_strategy_a.py --synthetic     # pipeline check on synthetic no-edge data
"""
from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cts.costs import CostModel  # noqa: E402
from cts.data import available_symbols, load_1m, load_funding  # noqa: E402
from cts.features import build_features  # noqa: E402
from cts.metrics import breakdown, monte_carlo, summarize  # noqa: E402
from cts.nulls import matched_random  # noqa: E402
from cts.portfolio import apply_constraints  # noqa: E402
from cts.sim import Simulator  # noqa: E402
from cts.strategies.sweep import SweepParams, exit_plan, generate, with_management  # noqa: E402
from cts.synthetic import synthetic_1m  # noqa: E402

DEV = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
HOLDOUT = ["ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
PEN = [1.0, 1.5, 2.0]
RVOL = [1.2, 1.5, 2.0]
COSTF = [4.0, 8.0]
MGMT = ["fixed", "scale"]
BASELINE = "pen1.5|rv1.5|cf8.0|{}"


def label(pen, rv, cf, mgmt):
    return f"pen{pen}|rv{rv}|cf{cf}|{mgmt}"


# --------------------------------------------------------------------------- per-symbol work
def work(job):
    sym, data_dir, n_null, synthetic = job
    if synthetic:
        df, funding = synthetic_1m(days=900, seed=int(sym[-1])), None
    else:
        df, funding = load_1m(sym, data_dir), load_funding(sym, data_dir)
    feat = build_features(df)
    cm, cm2 = CostModel.for_symbol(sym), CostModel.for_symbol(sym, 2.0)
    sim, sim2 = Simulator(df, cm, funding), Simulator(df, cm2, funding)
    vwap, sd = feat.m15["vwap96"].to_numpy(), feat.m15["sd96"].to_numpy()
    trades, nulls, funnels = [], [], {}
    for pen, rv, cf in itertools.product(PEN, RVOL, COSTF):
        base = SweepParams(pierce_max_atr=pen, rvol_min=rv, cost_filter_mult=cf)
        fn: dict = {}
        specs = generate(feat, sym, base, cm, fn)
        funnels[f"pen{pen}|rv{rv}|cf{cf}"] = fn
        for mgmt in MGMT:
            p = replace(base, management=mgmt)
            sp = with_management(specs, p)
            if not sp:
                continue
            t = sim.run_many(sp)
            t["R_2x"] = sim2.run_many(sp)["R"].to_numpy()
            t["combo"] = label(pen, rv, cf, mgmt)
            trades.append(t)
            filled = t[t["status"] == "filled"]
            if n_null and len(filled):
                def plan(i, side, p=p):
                    return exit_plan(p, side, vwap[i], sd[i])
                for seed in range(n_null):
                    ns = matched_random(feat, sym, filled, plan, seed=seed)
                    for s_, (_, real) in zip(ns, filled.iterrows()):
                        s_.meta["orig_signal_time"] = real["signal_time"]
                    nt = sim.run_many(ns)
                    nt["combo"], nt["seed"] = label(pen, rv, cf, mgmt), seed
                    nulls.append(nt)
    T = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    N = pd.concat(nulls, ignore_index=True) if nulls else pd.DataFrame()
    print(f"  {sym}: {len(T)} trade rows across combos, {len(N)} null rows", flush=True)
    return sym, T, N, funnels


# --------------------------------------------------------------------------- selection
def book(df: pd.DataFrame) -> pd.DataFrame:
    return apply_constraints(df) if len(df) else df


def score_grid(train: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    rows = []
    for (pen, rv, cf, mgmt) in itertools.product(PEN, RVOL, COSTF, MGMT):
        g = book(train[train["combo"] == label(pen, rv, cf, mgmt)])
        s = summarize(g) if len(g) else {"trades": 0}
        rows.append({"pen": pen, "rv": rv, "cf": cf, "mgmt": mgmt, "trades": s["trades"],
                     "E": s.get("expectancy_R", np.nan)})
    g = pd.DataFrame(rows)
    # plateau score: mean expectancy of the cell and its pen/rvol neighbours (same cf & mgmt)
    smooth = []
    for _, r in g.iterrows():
        ip, ir = PEN.index(r.pen), RVOL.index(r.rv)
        nb = g[(g.cf == r.cf) & (g.mgmt == r.mgmt)
               & g.pen.isin(PEN[max(ip - 1, 0): ip + 2]) & g.rv.isin(RVOL[max(ir - 1, 0): ir + 2])]
        smooth.append(nb["E"].mean())
    g["score"] = smooth
    g["eligible"] = g["trades"] >= min_trades
    return g


def choose(train: pd.DataFrame, min_trades: int):
    g = score_grid(train, min_trades)
    e = g[g.eligible & g.score.notna()]
    if e.empty:
        return None, g
    r = e.sort_values("score", ascending=False).iloc[0]
    return label(r.pen, r.rv, r.cf, r.mgmt), g


# --------------------------------------------------------------------------- report helpers
def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    df = df.reset_index() if df.index.name or not isinstance(df.index, pd.RangeIndex) else df
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


KEYS = ["trades", "win_rate", "wr_ci_lo", "wr_ci_hi", "breakeven_wr", "avg_win_R", "avg_loss_R",
        "expectancy_R", "expectancy_se", "profit_factor", "total_R", "max_dd_R", "max_dd_pct",
        "max_consec_losses", "tp1_rate", "trades_per_week"]


def summary_row(name: str, t: pd.DataFrame, r_col: str = "R") -> dict:
    tt = t.assign(R=t[r_col]) if r_col != "R" else t
    s = summarize(tt) if len(tt) else {"trades": 0}
    return {"set": name, **{k: s.get(k, np.nan) for k in KEYS}}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="reports/strategy_a")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nulls", type=int, default=20, help="matched-random seeds per combo")
    ap.add_argument("--min-train-trades", type=int, default=60)
    ap.add_argument("--lockbox-months", type=int, default=12)
    ap.add_argument("--lockbox", action="store_true", help="evaluate the lockbox period (once!)")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    if a.synthetic:
        dev, hold = ["SYN1", "SYN2", "SYN3"], ["SYN4"]
    else:
        have = set(available_symbols(a.data_dir))
        dev, hold = [s for s in DEV if s in have], [s for s in HOLDOUT if s in have]
        if not dev:
            sys.exit(f"no data in {a.data_dir}/parquet — run scripts/download_data.py first")
    syms = dev + hold
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"symbols dev={dev} holdout={hold}", flush=True)
    with Pool(min(a.workers, len(syms))) as pool:
        res = pool.map(work, [(s, a.data_dir, a.nulls, a.synthetic) for s in syms])
    T = pd.concat([r[1] for r in res if len(r[1])], ignore_index=True)
    N = pd.concat([r[2] for r in res if len(r[2])], ignore_index=True) if any(len(r[2]) for r in res) \
        else pd.DataFrame()
    funnels = {r[0]: r[3] for r in res}
    T = T[T["status"] == "filled"].copy()

    data_start = T["entry_time"].min().normalize()
    data_end = T["entry_time"].max()
    lock_start = (data_end.tz_convert(None).to_period("M").start_time.tz_localize("UTC")
                  - pd.DateOffset(months=a.lockbox_months - 1))
    pre = T[T["entry_time"] < lock_start]

    # ---------------- walk-forward
    s0 = (data_start + pd.DateOffset(days=35)).tz_convert(None).to_period("M").start_time.tz_localize("UTC")
    wins, oos, oos_dev_hold = [], [], []
    k = 0
    while True:
        tr0, tr1 = s0 + pd.DateOffset(months=3 * k), s0 + pd.DateOffset(months=3 * k + 12)
        te1 = tr1 + pd.DateOffset(months=3)
        if te1 > lock_start:
            break
        train = pre[(pre.entry_time >= tr0) & (pre.entry_time < tr1) & pre.symbol.isin(dev)]
        combo, _ = choose(train, a.min_train_trades)
        test_all = pre[(pre.entry_time >= tr1) & (pre.entry_time < te1)]
        row = {"window": k, "train": f"{tr0:%Y-%m}..{tr1:%Y-%m}", "test": f"{tr1:%Y-%m}..{te1:%Y-%m}",
               "chosen": combo or "none (too few trades)"}
        if combo:
            t = book(test_all[test_all.combo == combo]).assign(window=k)
            oos.append(t)
            raw = test_all[test_all.combo == combo].assign(window=k)
            oos_dev_hold.append(raw)
            s = summarize(t) if len(t) else {"trades": 0}
            row.update({"oos_trades": s["trades"], "oos_wr": s.get("win_rate"),
                        "oos_E": s.get("expectancy_R"), "oos_pf": s.get("profit_factor")})
        wins.append(row)
        k += 1
    W = pd.DataFrame(wins)
    OOS = pd.concat(oos, ignore_index=True) if oos else pd.DataFrame(columns=T.columns)
    RAW = pd.concat(oos_dev_hold, ignore_index=True) if oos_dev_hold else pd.DataFrame(columns=T.columns)

    # ---------------- null comparison on the OOS selection (unconstrained trade sets)
    null_txt = "not run"
    if len(N) and len(RAW):
        null_E = []
        for seed in sorted(N["seed"].unique()):
            parts = []
            for w_ in W.itertuples():
                if not isinstance(w_.chosen, str) or w_.chosen.startswith("none"):
                    continue
                tr1 = pd.Timestamp(w_.test.split("..")[0] + "-01", tz="UTC")
                te1 = tr1 + pd.DateOffset(months=3)
                q = N[(N.seed == seed) & (N.combo == w_.chosen) & (N.status == "filled")]
                parts.append(q[(q.orig_signal_time >= tr1) & (q.orig_signal_time < te1)])
            nn = pd.concat(parts) if parts else pd.DataFrame()
            if len(nn):
                null_E.append(nn["R"].mean())
        real_E = RAW["R"].mean()
        if null_E:
            pct = (np.array(null_E) < real_E).mean() * 100
            null_txt = (f"real OOS expectancy {real_E:+.3f}R vs matched-random nulls: mean "
                        f"{np.mean(null_E):+.3f}R, 95th pct {np.percentile(null_E, 95):+.3f}R "
                        f"-> real beats {pct:.0f}% of {len(null_E)} null runs")

    # ---------------- baseline (spec defaults), full pre-lockbox, descriptive (in-sample!)
    base_rows = []
    for mgmt in MGMT:
        t = book(pre[pre.combo == BASELINE.format(mgmt)])
        base_rows.append(summary_row(f"baseline {mgmt}", t))
    grid_rows = []
    for (pen, rv, cf, mgmt) in itertools.product(PEN, RVOL, COSTF, MGMT):
        t = book(pre[pre.combo == label(pen, rv, cf, mgmt)])
        r = summary_row(label(pen, rv, cf, mgmt), t)
        grid_rows.append({k_: r[k_] for k_ in ["set", "trades", "win_rate", "expectancy_R",
                                                "profit_factor", "max_dd_R"]})
    G = pd.DataFrame(grid_rows)

    # ---------------- OOS headline + breakdowns
    head = [summary_row("WF-OOS all symbols", OOS),
            summary_row("WF-OOS dev symbols", OOS[OOS.symbol.isin(dev)]),
            summary_row("WF-OOS holdout symbols", OOS[OOS.symbol.isin(hold)]),
            summary_row("WF-OOS at 2x costs", OOS, "R_2x")]
    H = pd.DataFrame(head)
    mc = monte_carlo(OOS.sort_values("exit_time")["R"].to_numpy()) if len(OOS) > 10 else {}

    lock_txt = "Lockbox NOT evaluated (run with --lockbox exactly once, when the design is frozen)."
    if a.lockbox:
        full_dev = pre[pre.symbol.isin(dev)]
        combo, _ = choose(full_dev, a.min_train_trades)
        box = T[(T.entry_time >= lock_start) & (T.combo == (combo or ""))]
        lb = book(box)
        lock_txt = (f"Lockbox {lock_start:%Y-%m} → end, combo chosen on all pre-lockbox dev data: "
                    f"`{combo}`\n\n" + md_table(pd.DataFrame([summary_row('lockbox', lb),
                                                              summary_row('lockbox 2x costs', lb, 'R_2x')])))

    # ---------------- write
    OOS.to_csv(out / "oos_trades.csv", index=False)
    W.to_csv(out / "wf_windows.csv", index=False)
    G.to_csv(out / "grid_prelockbox.csv", index=False)
    pd.DataFrame({(s_, c): f for s_, d in funnels.items() for c, f in d.items()}).T.fillna(0).astype(int)\
        .to_csv(out / "funnel.csv")

    def bd(by):
        return md_table(breakdown(OOS, by)) if len(OOS) else "(no OOS trades)"

    fsum = pd.DataFrame({s: f.get("pen1.5|rv1.5|cf8.0", {}) for s, f in funnels.items()}).fillna(0).astype(int)
    lines = [
        "# Strategy A (SWEEP) — backtest report", "",
        f"Data: {data_start:%Y-%m-%d} → {data_end:%Y-%m-%d}; dev symbols {dev}; holdout {hold}. "
        f"{'**SYNTHETIC no-edge data — pipeline check only.**' if a.synthetic else ''}",
        f"Lockbox starts {lock_start:%Y-%m-%d}; everything below except the lockbox section uses "
        "only earlier data.", "",
        "## Walk-forward out-of-sample (12m train / 3m test, dev-symbol selection, applied to all symbols)", "",
        md_table(H), "",
        f"Monte Carlo (10k bootstrap, 0.5% risk): {mc}", "",
        f"Matched-random null: {null_txt}", "",
        "### Windows", "", md_table(W), "",
        "### OOS by symbol", "", bd("symbol"), "",
        "### OOS by regime", "", bd("regime"), "",
        "### OOS by side", "", bd("side"), "",
        "### OOS by year", "", bd(OOS["entry_time"].dt.year.rename("year")) if len(OOS) else "", "",
        "### OOS by exit reason", "", bd("reason"), "",
        "## Lockbox", "", lock_txt, "",
        "## Descriptive (in-sample, pre-lockbox) — spec baseline pen1.5 / rvol1.5 / cost-filter 8x", "",
        md_table(pd.DataFrame(base_rows)), "",
        "### Full grid, pre-lockbox, all symbols (in-sample view, for plateau inspection only)", "",
        md_table(G), "",
        "### Filter funnel, baseline generation (counts of candidates removed per filter)", "",
        md_table(fsum), "",
    ]
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines[:12]))
    print(f"\nreport written to {out / 'report.md'}")


if __name__ == "__main__":
    main()

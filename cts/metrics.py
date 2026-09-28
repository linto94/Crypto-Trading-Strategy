"""Performance metrics (docs/strategy_research.md §9.5). Trades are rows with an ``R`` column."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def max_consecutive(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def drawdown_R(r: np.ndarray) -> float:
    eq = np.concatenate([[0.0], np.cumsum(r)])
    return float((np.maximum.accumulate(eq) - eq).max())


def drawdown_pct(r: np.ndarray, risk: float = 0.005) -> float:
    eq = np.cumprod(np.concatenate([[1.0], 1 + risk * r]))
    return float(((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max())


def summarize(trades: pd.DataFrame, risk: float = 0.005) -> dict:
    t = trades[trades["status"] == "filled"] if "status" in trades.columns else trades
    t = t.sort_values("exit_time") if "exit_time" in t.columns else t
    n = len(t)
    if n == 0:
        return {"trades": 0}
    r = t["R"].to_numpy(float)
    w = t["weight"].to_numpy(float) if "weight" in t else np.ones(n)
    wins = r > 0
    k = int(wins.sum())
    lo, hi = wilson(k, n)
    avg_win = float(r[wins].mean()) if k else 0.0
    avg_loss = float(-r[~wins].mean()) if k < n else 0.0
    b_net = avg_win / avg_loss if avg_loss > 0 else float("inf")
    be_wr = 1 / (1 + b_net) if np.isfinite(b_net) else 0.0
    gp, gl = r[wins].sum(), -r[~wins].sum()
    out = {
        "trades": n,
        "win_rate": k / n,
        "wr_ci_lo": lo,
        "wr_ci_hi": hi,
        "avg_win_R": avg_win,
        "avg_loss_R": avg_loss,
        "breakeven_wr": be_wr,
        "wr_edge_vs_breakeven": k / n - be_wr,
        "expectancy_R": float(r.mean()),
        "expectancy_se": float(r.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "profit_factor": float(gp / gl) if gl > 0 else float("inf"),
        "total_R": float(r.sum()),
        "max_dd_R": drawdown_R(r * w),
        "max_dd_pct": drawdown_pct(r * w, risk),
        "max_consec_losses": max_consecutive(~wins),
    }
    if "gross_R" in t:
        out["gross_expectancy_R"] = float(t["gross_R"].mean())
    if "tp1_hit" in t:
        out["tp1_rate"] = float(t["tp1_hit"].mean())
    if "entry_time" in t and n > 1:
        span_weeks = (t["entry_time"].max() - t["entry_time"].min()).total_seconds() / (7 * 86400)
        out["trades_per_week"] = n / span_weeks if span_weeks > 0 else float("nan")
    return out


def monte_carlo(r: np.ndarray, n_sims: int = 10000, risk: float = 0.005, seed: int = 0) -> dict:
    """Bootstrap the trade sequence; percentiles of max DD and longest losing streak."""
    rng = np.random.default_rng(seed)
    n = len(r)
    if n < 2:
        return {}
    dd, streak = np.empty(n_sims), np.empty(n_sims)
    for i in range(n_sims):
        s = r[rng.integers(0, n, n)]
        dd[i] = drawdown_pct(s, risk)
        streak[i] = max_consecutive(s <= 0)
    return {
        "mc_dd_pct_p50": float(np.percentile(dd, 50)),
        "mc_dd_pct_p95": float(np.percentile(dd, 95)),
        "mc_streak_p50": float(np.percentile(streak, 50)),
        "mc_streak_p95": float(np.percentile(streak, 95)),
    }


def breakdown(trades: pd.DataFrame, by: str | list[str], cols=("trades", "win_rate", "expectancy_R",
                                                              "profit_factor", "total_R")) -> pd.DataFrame:
    rows = {}
    for key, g in trades.groupby(by):
        s = summarize(g)
        rows[key] = {c: s.get(c, np.nan) for c in cols}
    return pd.DataFrame(rows).T

"""Book-level constraints (docs/strategy_research.md §1.2)."""
from __future__ import annotations

import pandas as pd


def apply_constraints(trades: pd.DataFrame, max_open: int = 3, max_same_dir: int = 2,
                      one_per_symbol: bool = True) -> pd.DataFrame:
    """Walk filled trades in entry order and drop those that would breach the book limits.

    Exit times come from each trade's own simulation, so no future information is used to
    decide acceptance: at a trade's entry only trades that entered earlier and are still open
    are considered.
    """
    t = trades[trades["status"] == "filled"].sort_values(["entry_time", "symbol"])
    open_: list[tuple[pd.Timestamp, str, int]] = []
    keep = []
    for idx, row in t.iterrows():
        open_ = [o for o in open_ if o[0] > row["entry_time"]]
        if len(open_) >= max_open:
            continue
        if sum(1 for o in open_ if o[2] == row["side"]) >= max_same_dir:
            continue
        if one_per_symbol and any(o[1] == row["symbol"] for o in open_):
            continue
        open_.append((row["exit_time"], row["symbol"], row["side"]))
        keep.append(idx)
    return t.loc[keep]

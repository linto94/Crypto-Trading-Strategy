import numpy as np
import pandas as pd
import pytest

from cts.costs import CostModel
from cts.features import build_features
from cts.metrics import max_consecutive, summarize, wilson
from cts.nulls import random_entries
from cts.sim import Simulator, Target, TradeSpec
from cts.strategies.sweep import SweepParams, generate
from cts.synthetic import synthetic_1m

T0 = pd.Timestamp("2024-01-01 00:00", tz="UTC")


def bars(rows):
    """rows: list of (open, high, low, close) 1m bars starting at T0."""
    idx = pd.date_range(T0, periods=len(rows), freq="1min", tz="UTC", name="open_time")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1.0
    df["taker_buy_volume"] = 0.5
    return df


def spec(side=1, stop=99.0, targets=None, ts=100, be=False):
    return TradeSpec("X", "t", side, T0, T0, stop, targets or [Target(1.0, 1.0)], ts, be, bar_minutes=1)


def test_long_hits_target():
    df = bars([(100, 100.5, 99.8, 100.2), (100.2, 101.2, 100.1, 101)])
    r = Simulator(df, CostModel.zero()).run(spec())
    assert r["reason"] == "target" and r["R"] == pytest.approx(1.0)


def test_target_needs_trade_through():
    df = bars([(100, 101.0, 99.8, 100.2), (100.2, 100.5, 100.1, 100.3), (100.3, 100.4, 100.2, 100.3)])
    r = Simulator(df, CostModel.zero()).run(spec(ts=2))
    assert r["reason"] == "time"


def test_same_bar_ambiguity_assumes_stop():
    df = bars([(100, 101.5, 98.5, 100)])
    r = Simulator(df, CostModel.zero()).run(spec())
    assert r["reason"] == "stop" and r["R"] == pytest.approx(-1.0)


def test_gap_through_stop_fills_at_open():
    df = bars([(100, 100.2, 99.9, 100), (98.0, 98.5, 97.5, 98)])
    r = Simulator(df, CostModel.zero()).run(spec())
    assert r["reason"] == "stop" and r["R"] == pytest.approx(-2.0)


def test_short_mirror():
    df = bars([(100, 100.2, 99.5, 99.6), (99.6, 99.7, 98.9, 99)])
    r = Simulator(df, CostModel.zero()).run(spec(side=-1, stop=101.0))
    assert r["reason"] == "target" and r["R"] == pytest.approx(1.0)


def test_scale_out_then_breakeven():
    df = bars([(100, 100.9, 99.9, 100.8), (100.8, 100.85, 99.95, 100.0)])
    tg = [Target(0.8, 0.5), Target(1.6, 0.5)]
    r = Simulator(df, CostModel.zero()).run(spec(targets=tg, be=True))
    assert r["tp1_hit"] and r["reason"] == "be_stop"
    assert r["R"] == pytest.approx(0.4)


def test_costs_reduce_R():
    df = bars([(100, 100.5, 99.8, 100.2), (100.2, 101.5, 100.1, 101)])
    cm = CostModel.for_symbol("BTCUSDT")
    r = Simulator(df, cm).run(spec())
    assert r["R"] < r["gross_R"]


def test_time_stop_exits_at_open():
    df = bars([(100, 100.2, 99.9, 100.1)] * 3 + [(100.3, 100.4, 100.2, 100.3)])
    r = Simulator(df, CostModel.zero()).run(spec(ts=3))
    assert r["reason"] == "time" and r["R"] == pytest.approx(0.3)


def test_metrics_basics():
    assert max_consecutive(np.array([1, 1, 0, 1, 1, 1, 0], bool)) == 3
    lo, hi = wilson(70, 100)
    assert lo < 0.7 < hi
    s = summarize(pd.DataFrame({"R": [1, 1, -1, 1], "status": "filled"}))
    assert s["win_rate"] == 0.75 and s["expectancy_R"] == 0.5 and s["profit_factor"] == 3


@pytest.fixture(scope="module")
def synth():
    df = synthetic_1m(days=150, seed=5)
    return df, build_features(df)


def test_features_are_causal(synth):
    df, full = synth
    cut = df.index[len(df) * 2 // 3]
    part = build_features(df[df.index < cut])
    cols = ["atr", "vwap96", "z", "rvol", "delta", "sigma_r", "regime", "slope_4h", "volpct_1h",
            "atr_1h", "er48_1h"]
    a = full.m15.loc[part.m15.index, cols]
    b = part.m15[cols]
    num = [c for c in cols if c != "regime"]
    np.testing.assert_allclose(a[num].to_numpy(float), b[num].to_numpy(float), rtol=1e-9, equal_nan=True)
    assert (a["regime"].astype(str) == b["regime"].astype(str)).all()


def test_sweep_signals_are_causal(synth):
    df, full = synth
    p = SweepParams(cost_filter_mult=0, rvol_min=0, require_absorption=False, room_R=0,
                    room_R_transition=0, close_pos_min=0.5)
    cm = CostModel.zero()
    cut = df.index[len(df) * 2 // 3]
    s_full = [(s.signal_time, s.side, round(s.stop, 8)) for s in generate(full, "X", p, cm)]
    part = build_features(df[df.index < cut])
    s_part = [(s.signal_time, s.side, round(s.stop, 8)) for s in generate(part, "X", p, cm)]
    last = part.m15.index[-2]
    assert len(s_part) > 10
    assert [x for x in s_full if x[0] < last] == [x for x in s_part if x[0] < last]


def test_random_entries_match_geometric_null(synth):
    df, feat = synth
    t = Simulator(df, CostModel.zero()).run_many(random_entries(feat, "X", 1500, 1.0, seed=1))
    s = summarize(t)
    assert abs(s["win_rate"] - 0.5) < 0.05
    assert abs(s["expectancy_R"]) < 3 * s["expectancy_se"] + 0.02

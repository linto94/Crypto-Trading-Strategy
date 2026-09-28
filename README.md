# Crypto-Trading-Strategy

Research into a high-win-rate, positive-expectancy strategy for liquid crypto pairs (15m execution, 1h/4h context).

- [Phase 1 — strategy architectures & testing methodology](docs/strategy_research.md)

## Backtest engine (`cts/`)

| Module | Purpose |
|---|---|
| `cts/data.py` | Download Binance USDT-M 1m klines + funding from data.binance.vision; resample to 15m/1h/4h/1D/1W |
| `cts/features.py` | Multi-timeframe features and the regime classifier (§1). HTF values are joined by bar close time, so there is no look-ahead |
| `cts/sim.py` | Trade simulator with 1m fill resolution: next-open entries, stop gaps, same-bar stop-first rule, trade-through limit targets, scale-out + breakeven, time stops, funding |
| `cts/costs.py` | Fee / spread / slippage model per liquidity tier, with a 2× stress multiplier |
| `cts/metrics.py` | Win rate + Wilson CI, breakeven WR, expectancy, PF, drawdown, losing streaks, Monte Carlo |
| `cts/portfolio.py` | Book limits: max 3 open, max 2 same direction, 1 per symbol |
| `cts/nulls.py` | Random-entry and matched-random null benchmarks |
| `cts/strategies/sweep.py` | Strategy A — liquidity sweep reversal |
| `cts/synthetic.py` | No-edge synthetic data used to validate the engine |

## Usage

```bash
pip install -r requirements.txt
python -m pytest -q tests                      # engine + causality tests
python scripts/download_data.py --start 2021-01 --end 2026-08   # ~1-2 GB, needs access to data.binance.vision
python scripts/run_strategy_a.py               # grid + walk-forward OOS + nulls -> reports/strategy_a/report.md
python scripts/run_strategy_a.py --lockbox     # one-shot lockbox evaluation, once the design is frozen
python scripts/run_strategy_a.py --synthetic   # pipeline check on synthetic no-edge data
```

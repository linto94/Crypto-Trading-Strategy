# High-Win-Rate Crypto Strategy Research — Phase 1: Architectures & Test Plan

Scope: liquid USDT perpetuals/spot (BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, DOGEUSDT, ADAUSDT, AVAXUSDT, LINKUSDT, LTCUSDT).
Status: **design only — nothing here has been backtested yet.** All win-rate figures below are hypotheses to be falsified, not results.

---

## 0. First principles: what a 70% win rate actually requires

### 0.1 Win rate is mostly a property of exit geometry, not of edge

For a driftless random walk, the probability of hitting a take-profit at distance `b·R` before a stop at distance `1·R` is:

```
P_null(win) = 1 / (1 + b)
```

| Reward:Risk (b) | Null win rate (zero edge) |
|---|---|
| 0.5 | 66.7% |
| 0.8 | 55.6% |
| 1.0 | 50.0% |
| 1.5 | 40.0% |
| 2.0 | 33.3% |

So a strategy with **zero edge** and a 0.5R target already "wins" 67% of the time. **Win rate on its own says nothing about edge.** The number that matters is the **excess hit rate**:

```
Edge_hit = WR_observed − 1/(1+b)
```

Every strategy in this document is judged on `Edge_hit`, expectancy and profit factor, not on raw WR.

### 0.2 Costs are measured in R, and on 15m they are large

Let `c` = round-trip cost (fees + spread + slippage) divided by stop distance. A winner earns `b − c`, a loser loses `1 + c`. The breakeven win rate is:

```
WR_breakeven = (1 + c) / (1 + b)
```

Example: a BTC 15m stop of 0.6% with round-trip taker costs of ~0.12% gives `c = 0.2`.

| b | WR breakeven, c = 0.1 | WR breakeven, c = 0.2 | Expectancy at 70% WR, c = 0.1 | Expectancy at 70% WR, c = 0.2 |
|---|---|---|---|---|
| 0.5 | 73.3% | 80.0% | **−0.05R** | **−0.15R** |
| 0.8 | 61.1% | 66.7% | +0.16R | +0.06R |
| 1.0 | 55.0% | 60.0% | +0.30R | +0.20R |
| 1.5 | 44.0% | 48.0% | +0.65R | +0.55R |
| 2.0 | 36.7% | 40.0% | +1.00R | +0.90R |

(Expectancy = `WR·(b − c) − (1 − WR)·(1 + c)`.)

Consequences:

1. **A 70% WR with small targets (b ≤ 0.5) loses money after costs.** Most "90% win rate" scalping systems are this trap.
2. **Cost control is a design constraint.** Rule used throughout: **the stop distance must be ≥ 8× the round-trip cost**, which keeps `c ≤ 0.125`. On 15m this rules out tight stops. It is also the main reason not to go down to 5m.
3. **What is realistic:** 70% at **b ≈ 0.8–1.0** means an excess hit rate of about 15–20 points over the null. That is strong but plausible for heavily filtered, regime-gated setups. **70% at 1:1.5** (excess ≈ 30 points, expectancy ≈ +0.65R/trade) or **70% at 1:2** (≈ +1R/trade) is not credible for a systematic mid-frequency strategy on liquid majors. If a backtest shows it, assume a bug first: look-ahead, same-bar TP/SL ambiguity, or survivorship.

### 0.3 The structural way to raise win rate honestly: partial exits

A scale-out (take 50% at TP1, move the stop on the rest to breakeven + costs) turns every trade that reaches TP1 into a net winner. The WR then becomes `P(TP1 before SL)`, and the runner keeps the average win large enough. Example with TP1 = 0.8R and TP2 = 1.6R:

```
P(TP1) = 0.68, P(TP2 | TP1) = 0.45
E = 0.68 · (0.5·0.8 + 0.45·0.5·1.6 + 0.55·0) − 0.32 · 1.0
  = 0.68 · 0.76 − 0.32 = +0.197R before costs
```

The same trades with a single 1R exit would need a 60%+ hit rate to match. **Definition used everywhere: a trade is a "win" only if its total net P&L after all costs is > 0.** Breakeven-stopped runners after TP1 count as wins because TP1 was banked. A trade stopped at exactly BE with no TP1 counts as a loss (its costs are negative).

### 0.4 Answers to the core questions (summary)

1. **Which strategy type can realistically reach high WR?** Ones whose target sits *inside* a statistically likely excursion and whose stop sits *beyond* a level where the thesis is objectively invalid. In practice: (a) failed-breakout/liquidity-sweep reversals back into a range, (b) pullback entries in the direction of an established higher-timeframe trend, (c) snapbacks after forced-liquidation overshoots, and (d) regime-gated mean reversion. Pure breakout and pure trend-following have structurally low WR (30–45%) and are excluded as primary engines.
2. **Conditions/filters for 70%+:** a regime gate (trend/range/shock classification on 1h/4h), a location filter (only at mapped levels or statistical extremes), an order-flow confirmation (taker-delta absorption or exhaustion), a room-to-target filter, and a cost filter. Each has to be justified by improving the *out-of-sample* excess hit rate, not the in-sample WR.
3. **Appropriate R:R:** effective b of **0.8–1.2**, preferably made up of a TP1 at 0.7–1.0R plus a runner. Never a single target below 0.7R.
4. **Is 70% achievable?** At 1:1 net: possible but on the edge, so budget for 60–68% OOS. At 1:1.5 or 1:2: no. With scale-out (TP1 ≈ 0.8R, runner to 1.6R+): the most plausible route to ≥ 70% *and* positive expectancy.
5. **Trades to reject:** see §6.
6. **Regime behaviour:** see §7.

### 0.5 Recommended timeframe

**15m execution, 1h regime classification, 4h directional bias. 1m data is used only to simulate fills.**

- 5m: stop distances needed for `c ≤ 0.125` are too wide relative to 5m noise, so the timeframe loses its benefit.
- 1h entries: lower cost ratio, but ~4× fewer trades, which makes the statistics weak within 1–2 years. Run Strategies A and B on 1h as a **robustness check**. If the edge disappears on 1h, it was probably noise on 15m.

---

## 1. Shared definitions (used by all strategies)

All higher-timeframe (HTF) values use **only the last fully closed HTF bar**. Subscripts: `15` = 15m bars, `1h`, `4h`, `D` = UTC day.

| Symbol | Definition |
|---|---|
| `ATR15` | Wilder ATR(14) on 15m |
| `ATR1h`, `ATR4h` | Wilder ATR(14) on 1h / 4h |
| `ER(n)` | Kaufman efficiency ratio = `|C_t − C_{t−n}| / Σ_{i=t−n+1..t} |C_i − C_{i−1}|` (0 = pure noise, 1 = straight line) |
| `Slope_4h` | `(EMA50_4h,t − EMA50_4h,t−5) / ATR4h` (trend slope in ATR units per 20h) |
| `Slope_1h` | `(EMA50_1h,t − EMA50_1h,t−12) / ATR1h` |
| `VWAP96` | Rolling 96-bar (24h) volume-weighted average of typical price `(H+L+C)/3` on 15m |
| `SD96` | Rolling 96-bar standard deviation of `(C − VWAP96)` |
| `z` | `(C − VWAP96) / SD96` |
| `VolPct` | Percentile rank of `ATR1h / C_1h` within the trailing 90 days (2160 1h bars) |
| `RVOL` | `V_t / median(V at the same 15m time-of-day slot over the last 20 days)` (time-of-day adjusted, which matters in crypto because of the Asia/EU/US volume cycle) |
| `Delta` | Taker delta = `2·TakerBuyBaseVol − Vol` (Binance klines provide taker-buy volume). `DeltaRatio = Delta / Vol` ∈ [−1, 1] |
| Pivot high (k) | Bar `i` is a pivot high if `H_i > max(H_{i−k..i−1})` and `H_i ≥ max(H_{i+1..i+k})`. **Only usable from bar `i+k` onward** (confirmation delay, no look-ahead). k = 3 on 15m, k = 2 on 1h. Pivot lows are the mirror. |
| `σ_r` | Standard deviation of 15m log returns over the trailing 7 days (672 bars) |
| Round-trip cost `RTC` | fees + spread + slippage for entry and exit (see §9.3) |

### 1.1 Regime classifier (recomputed at each 1h close)

Evaluated in order; the first match wins.

| Regime | Rule |
|---|---|
| **SHOCK** | `VolPct ≥ 95`, **or** any 15m bar in the last 4 bars had `TrueRange ≥ 3.5·ATR15`, **or** `|return over last 4 × 15m| ≥ 5·σ_r·√4` |
| **TREND_UP** | `ER_4h(20) ≥ 0.30` **and** `Slope_4h ≥ +0.75` **and** `C_4h > EMA50_4h` **and** `EMA20_1h > EMA50_1h` |
| **TREND_DOWN** | mirror of TREND_UP |
| **RANGE** | `ER_1h(48) ≤ 0.20` **and** `|Slope_1h| ≤ 0.5` **and** `20 ≤ VolPct ≤ 85` |
| **TRANSITION** | everything else |
| **DEAD** | overlay: `VolPct < 10` (too little movement to cover costs; blocks all entries) |

Thresholds are starting points. They are tuned **only** on the in-sample set, **only** in coarse steps, and must sit on a plateau (§9.6).

### 1.2 Universal trade filters

- **Cost filter:** reject if `stop distance < 8 · RTC`.
- **Stop-width filter:** reject if `stop distance > 3.0 · ATR15` (the trade would be too large a bet on one bar's structure). The stop is **never widened** to satisfy a rule; the trade is skipped instead.
- **One position per symbol.** Max 3 concurrent positions across the book, and max 2 in the same direction (crypto majors have correlations of 0.7–0.9 during stress).
- **Event blackout:** no new entries from 30 min before to 60 min after scheduled US CPI, FOMC decision and NFP releases (calendar file maintained by hand; tested with and without the filter).

---

## 2. Strategy A — Range-Extreme Liquidity Sweep Reversal ("SWEEP")

**Thesis.** Stops and breakout orders cluster just beyond obvious highs and lows. In non-trending conditions, a brief pierce that triggers them and then closes back inside the range is a *failed auction*. The liquidity has been used up and price rotates back toward value. The target sits inside the range (high hit probability) and the stop sits beyond the sweep extreme (objective invalidation).

**Market regime:** RANGE or TRANSITION. Never SHOCK or DEAD. Never counter to TREND_UP/DOWN (for example, no short sweeps in TREND_UP).

**Features:** PDH/PDL, prior-week high/low, 1h pivots, ATR15, ATR1h, RVOL, Delta, VWAP96, regime.

### Level map (rebuilt at each 1h close)
Candidate levels:
1. Prior UTC day high/low (PDH/PDL).
2. Prior ISO-week high/low.
3. Confirmed 1h pivot highs/lows (k = 2) from the last 72h that are **untouched**, meaning no 1h close beyond them since they formed, and whose age is ≥ 8 1h bars.

Levels within `0.25·ATR1h` of each other are merged into one level at the extreme (the highest of a cluster of highs, the lowest of a cluster of lows). A level is **consumed** after it produces one trade or after a 1h close beyond it.

### LONG conditions (mirror for SHORT)
Let `Lvl` be a support level from the map. A **sweep window** is up to 3 consecutive 15m bars `t0..t2`:
1. **Pierce:** `min(L_{t0..tn}) ≤ Lvl − 0.10·ATR15` (a real pierce, not a tick).
2. **Not a breakdown:** `min(L_{t0..tn}) ≥ Lvl − 1.50·ATR15`.
3. **Reclaim:** the signal bar `ts` (the last bar of the window) closes `C_ts > Lvl + 0.05·ATR15`.
4. **Rejection shape:** on `ts`, `(C_ts − L_ts) / (H_ts − L_ts) ≥ 0.60` (closes in the upper 40% of its range).
5. **Participation:** `max(RVOL_{t0..ts}) ≥ 1.5`.
6. **Absorption (order-flow):** `Σ Delta_{t0..ts} < 0` (aggressive sellers were net active during the sweep) **and** `C_ts > Lvl`. Sellers hit the bid and price still closed back inside, so passive buyers absorbed them.
7. **Room to target:** the nearest opposing level (resistance) is ≥ `2.0R` away from entry.
8. **Regime:** RANGE or TRANSITION; for longs also `Slope_4h > −0.75`.

### Entry trigger
Market order at the **open of bar `ts+1`**. Variant A2 (tested separately): a limit at `(C_ts + Lvl)/2`, valid for 2 bars, counted as filled only if price trades *through* the limit by at least 1 tick.

### Stop-loss
`SL = min(L_{t0..ts}) − 0.25·ATR15`. The trade is skipped if `(entry − SL)` falls outside `[0.8, 3.0]·ATR15` or fails the cost filter.

### Take-profit & management
- **A-fixed (baseline):** TP = entry + 1.0R. Time stop: exit at market after 16 bars (4h) if neither TP nor SL is hit.
- **A-scale:** 50% at 0.8R, then stop to `entry + RTC` (breakeven after costs), remaining 50% at `min(1.6R, VWAP96 + 1·SD96)`. Time stop 24 bars.

**R:R:** 1:1 (fixed) or effective ~1:1.1 (scale).

### No-trade conditions
SHOCK/DEAD regime; the level was already swept earlier the same UTC day; the sweep bar range > 3.5·ATR15 (that is news, not a stop-run); event blackout; 4h trend opposite with `|Slope_4h| ≥ 0.75`.

**Expected frequency:** about 1–3 trades per symbol per week after filters, so roughly 10–30 per week across 10 symbols.

**Why it could reach a high WR:** the target is a rotation back into value, which only needs the failed auction to follow through a little. The stop is placed where the trade idea is objectively wrong. Absorption and room filters remove the sweeps that are really the start of a breakdown.

**Weaknesses / failure conditions:** regime shifts from range to trend (sweeps turn into real breakouts, and losses cluster at the start of trends); news-driven moves; weekend thin-liquidity fake-outs that go both ways; levels that are "obvious" to everyone and get front-run.

**Overfitting risks:** too many level types and merge tolerances; tuning the penetration band (0.10/1.50) finely; allowing the level list to grow until every bar is "near a level". Mitigation: freeze the level definition before any backtest and tune only 3 parameters (penetration max, RVOL threshold, TP multiple).

---

## 3. Strategy B — Trend-Regime Pullback Continuation ("PULLBACK")

**Thesis.** When a higher-timeframe trend is objectively established, the drift is on your side. A counter-trend pullback into dynamic value (EMA/VWAP), followed by a resumption trigger with aggressive buyers, has a modestly higher-than-random chance of retesting the prior swing high. Taking partial profit before that retest turns the drift into a high hit rate.

**Market regime:** TREND_UP for longs, TREND_DOWN for shorts. Nothing else.

**Features:** EMA20/50 (1h), EMA50 (4h), EMA50 (15m), VWAP96, ATR15, ATR1h, RSI(14) on 15m and 1h, Delta, ER, Slope_4h.

### LONG conditions (mirror for SHORT)
Let `HH = max(H over last 32 × 15m bars)`, `tHH` = the bar where it occurred, and `PL = min(L since tHH)` (pullback low).
1. **Regime:** TREND_UP, and `RSI14_1h > 50`.
2. **Impulse present:** `HH − min(L over the 64 bars before tHH) ≥ 3·ATR1h` (there was a real leg up).
3. **Pullback age:** `t − tHH ≥ 4` bars (a real pullback, not one red candle).
4. **Pullback depth:** `(HH − PL) / ATR1h ∈ [0.8, 2.5]`, **and** retracement `(HH − PL) / impulse ≤ 0.618`.
5. **Touches value:** `PL ≤ max(EMA50_15m, VWAP96)` at some bar during the pullback.
6. **Momentum reset:** `min(RSI14_15m since tHH) < 40`.
7. **Pullback is corrective, not impulsive:** mean `DeltaRatio` over the pullback bars > −0.15 (no heavy aggressive selling), **and** no pullback bar with `TrueRange > 2.5·ATR15`.

### Entry trigger
On a 15m bar `ts` within 3 bars of `PL` being set:
`C_ts > H_{ts−1}` **and** `C_ts > O_ts` **and** `DeltaRatio_ts > +0.10`. Enter at market on the open of `ts+1`.

### Stop-loss
`SL = PL − 0.20·ATR15`. The trade must satisfy `(entry − SL) ∈ [1.0, 3.0]·ATR15` plus the cost filter.

### Take-profit & management
- **B-scale (primary):** 50% at TP1 = entry + 0.8R, then stop to `entry + RTC`. The remaining 50% targets TP2 = entry + 2.0R **or** trails below the most recent confirmed 15m pivot low minus 0.2·ATR15, whichever exits first.
- **B-fixed (comparison):** single TP at 1.0R, time stop 32 bars (8h).
- **Regime-exit:** if the regime drops out of TREND_UP while the runner is open, close it at the next 15m open.

**R:R:** effective 1:1.2–1.4 (scale); 1:1 (fixed).

### No-trade conditions
The regime is not TREND in the trade direction; price is > 4·ATR4h from EMA50_4h (over-extended trend; late entries fail); the pullback low undercut the previous 15m pivot low (structure broken); a 4h bar in the last 3 closed against the trend with body > 1.5·ATR4h; funding rate (last settlement) in the top 5% of its 90-day distribution in the trade direction (crowded).

**Expected frequency:** only while trending, so about 1–2 per symbol per week on average, clustered (0 in ranges, 3–5 per week in strong trends).

**Why it could reach a high WR:** it trades with the drift, buys at a statistically cheap location relative to the trend, and requires evidence that selling pressure was passive. With TP1 at 0.8R the null hit rate is 55.6%, and the trend drift plus the filters only need to add ~13 points.

**Weaknesses:** trend exhaustion (the last pullback of a trend looks identical to the earlier ones); V-shaped reversals; trend-regime detection lags, so the first pullback after a regime flip is often missed and the last one is often taken.

**Overfitting risks:** many interacting thresholds (depth band, retracement cap, RSI reset). Treat conditions 4–7 as a *block* and ablate them (§9.7). Any condition whose removal doesn't hurt OOS is deleted.

---

## 4. Strategy C — Liquidation-Cascade Snapback ("CASCADE")

**Thesis.** In leveraged perpetual markets, a fast move triggers forced liquidations, which trigger more (a cascade). Liquidations are *price-insensitive* market orders, so the move overshoots fair value. Once the forced flow ends, which shows up as open interest having dropped, price partially retraces. The signal is structural (leverage mechanics), not just a chart pattern, which makes it more likely to persist.

**Market regime:** fires in SHOCK conditions. It is the only strategy allowed to trade in SHOCK.

**Features:** 15m OHLCV, taker-buy volume, open interest (Binance futures `metrics`, 5m, aggregated to 15m), funding rate, σ_r, ATR15.

### LONG conditions (downside cascade; mirror for SHORT)
Cascade window `w ∈ {1, 2, 3}` bars ending at bar `tc`:
1. **Overshoot:** `ln(C_tc / C_{tc−w}) ≤ −4.0 · σ_r · √w`.
2. **Forced flow:** open-interest change over the window `ΔOI/OI ≤` the 2nd percentile of all `w`-bar OI changes over the trailing 30 days (OI fell sharply because positions were closed, not opened).
3. **Aggression:** `Σ Delta / Σ Vol ≤ −0.25` over the window, and `max RVOL ≥ 3`.
4. **Crowding before the move:** mean funding over the 24h before `tc−w` ≥ 0 (longs were paying, so longs were crowded, which is who got liquidated).

Let `CH = max(H_{tc−w..tc})` (cascade high), `CL` = the lowest low from `tc−w` to the entry bar, and `Range = CH − CL`.

### Entry trigger (exhaustion confirmation), within 4 bars after `tc`
Signal bar `ts` with: `C_ts > O_ts`, `(C_ts − L_ts)/(H_ts − L_ts) ≥ 0.5`, `L_ts ≥ CL − 0.25·ATR15` (no meaningful new low), and `DeltaRatio_ts > 0`. Enter at market on the open of `ts+1`. **Cancel** the setup if any bar makes a new low `< CL − 1.0·ATR15` before a signal bar forms (second leg down).

### Stop-loss
`SL = CL − 0.5·ATR15`, with a minimum distance of `1.5·ATR15` (volatility is extreme, so slippage on stops is modelled at 3× normal).

### Take-profit & management
- TP = `CL + 0.382·Range`. The trade is skipped if TP distance < 0.8R. If TP distance > 1.5R, TP is capped at 1.5R.
- Time stop: 8 bars (2h). Snapbacks are fast; if nothing happens, the thesis is gone.
- Optional scale: 50% at 0.7R, stop to BE after costs.

**R:R:** 0.8–1.5 (variable, set by geometry).

### No-trade conditions
The cascade coincides with a scheduled macro event (it's news, not a liquidation overshoot); the cascade breaks the prior-week low/high by > 2·ATR4h (a structural break, possibly a regime change); OI or funding data is missing for the window; more than 2 open CASCADE trades across symbols (market-wide crashes are correlated, and those trades would effectively be one bet).

**Expected frequency:** rare, about 1–4 per symbol per month. Pooled across 10 symbols that is ~15–40 per month, but clustered on volatile days.

**Why it could reach a high WR:** it trades a mechanically caused overshoot, and the target is only a 38.2% retracement of a move that was, by construction, ≥ 4σ.

**Weaknesses:** true information shocks (exchange collapses, hacks, regulatory news) where there is no snapback; fills during cascades are much worse than modelled; OI data quality/latency in live trading; rarity means statistical confidence builds slowly.

**Overfitting risks:** few events means every threshold can be tuned to the few dozen big crashes. Mitigation: fix σ-multiple, percentile and retracement **a priori** (the values above), tune nothing, and just measure.

---

## 5. Strategy D — Variance-Ratio-Gated Mean Reversion ("STATREV")

**Thesis.** Classic z-score fading works in some periods and blows up in others. The problem is not the entry, it is the missing *regime test*. The Lo–MacKinlay variance ratio directly measures whether recent returns are mean-reverting (VR < 1) or trending (VR > 1). Only fade extremes when the recent return process is itself measurably mean-reverting.

**Market regime:** RANGE, plus the statistical gate below.

**Features:** 15m log returns, VR, VWAP96, SD96, z, ER_1h, VolPct.

### Regime gate (both directions)
- `VR(q=8) = Var(r^{(8)}) / (8 · Var(r^{(1)}))` computed on the trailing 384 × 15m bars (4 days), overlapping 8-bar returns. Require **`VR ≤ 0.80`**.
- `ER_1h(24) ≤ 0.25` and `20 ≤ VolPct ≤ 80`.

### LONG conditions (mirror for SHORT)
1. `z_ts ≤ −2.25`.
2. **Deceleration:** `|C_ts − C_{ts−1}| < |C_{ts−1} − C_{ts−2}|` **or** `(C_ts − L_ts)/(H_ts − L_ts) ≥ 0.5`.
3. No 1h close outside the prior 24h high/low range in the last 2 hours (no active range breakout).

### Entry trigger
Market on the open of `ts+1`.

### Stop-loss
`SL = entry − 1.5·SD96` (≈ z of −3.75), with a minimum of `1.2·ATR15`.

### Take-profit & management
- **Dynamic target:** exit when `z ≥ −0.25` (price has returned to ≈ VWAP). Checked on bar close and executed at the next open, *or* by a resting limit order at the level of `VWAP96 − 0.25·SD96` recomputed every bar.
- **Hard cap:** TP at 1.5R regardless.
- **Time stop:** 12 bars (3h), exit at market.

**R:R:** variable, typically 0.9–1.3 at entry (it shrinks as VWAP moves toward price, which *raises* WR and lowers the average win; watch this trade-off).

### No-trade conditions
VR > 0.80; any TREND or SHOCK regime; `z` got to ≤ −2.25 in a single bar with `TrueRange > 3·ATR15` (that's a shock, which belongs to Strategy C's domain); weekends if the weekend-vs-weekday OOS split shows degradation (tested, not assumed).

**Expected frequency:** 2–5 per symbol per week in range regimes, zero in trends.

**Why it could reach a high WR:** the dynamic target is close (VWAP), the regime is statistically verified as mean-reverting, and the time stop removes slow bleeders. Mean reversion is the natural "high WR, small wins" archetype. The VR gate is what's meant to keep it from being "high WR, catastrophic tail".

**Weaknesses:** VR is estimated on noisy data (4 days × 96 bars has large sampling error); regime breaks happen suddenly, so the first trade of a new trend is always a full loss; the average win is small, so costs matter most here.

**Overfitting risks:** the VR window, q, threshold and z threshold interact strongly. Tune only `z_entry ∈ {2.0, 2.25, 2.5}` and `VR_max ∈ {0.75, 0.80, 0.85}` and require a plateau.

---

## 6. Trades rejected outright (all strategies)

1. **Cost-dominated trades:** stop < 8 × RTC (makes the arithmetic in §0.2 unwinnable).
2. **Targets below 0.7R** as the *only* exit (the null WR is already ≥ 59%, so apparent high WR there is fake).
3. **Counter-trend fades against a TREND regime** with `|Slope_4h| ≥ 0.75` (Strategies A and D).
4. **Trend entries in RANGE/TRANSITION** (Strategy B).
5. **Anything in the SHOCK regime** except Strategy C.
6. **DEAD regime** (VolPct < 10): the edge can't cover costs.
7. **Scheduled macro-event windows** (−30 / +60 min).
8. **Setups needing a widened stop** to fit the rules. Skip them; never widen.
9. **A 4th concurrent position, or a 3rd in the same direction.**
10. **Data anomalies:** a 15m bar with zero volume, a missing OI/funding window, or a price gap between consecutive bars > 3·ATR15 (exchange incident).
11. **After 4 consecutive losses in one strategy:** pause that strategy for 24h. This is **not** assumed to add edge. It is a risk-control rule, and it is backtested both with and without to confirm it doesn't *destroy* edge.

---

## 7. Behaviour by regime

| Regime | SWEEP (A) | PULLBACK (B) | CASCADE (C) | STATREV (D) | Sizing |
|---|---|---|---|---|---|
| TREND_UP | Longs only (sweeps of support); shorts disabled | **Longs active** | Longs & shorts if triggered | Off | 1.0× |
| TREND_DOWN | Shorts only | **Shorts active** | Both | Off | 1.0× |
| RANGE | **Both sides active** | Off | Both | **Active** | 1.0× |
| TRANSITION | Both, but room filter raised to 2.5R | Off | Both | Off | 0.5× |
| SHOCK | Off | Off; open runners are closed | **Only strategy active** | Off | 0.5× (wider stops, same R-risk) |
| DEAD | Off | Off | Off | Off | — |

The design intent is that the four engines are **complementary by regime**: B makes money when A and D are switched off, and C covers the conditions where A, B and D are switched off. This is also the portfolio's main defence against regime drift.

---

## 8. Meta-layer (Phase 2, only after single strategies prove out): Regime router + meta-labeling

Once A–D have individually passed OOS, add a **secondary classifier** (meta-labeling, as in López de Prado) that predicts `P(win)` for each primary signal and only takes signals with `P ≥ p*`.

- **Model:** L2-regularised logistic regression first; gradient-boosted trees (max depth 3, monotonic constraints where the sign is known) only if logistic is clearly insufficient.
- **Features (≤ 12, all known at entry):** regime one-hot, VolPct, ER_1h, Slope_4h, VR, RVOL, DeltaRatio, distance to VWAP in SD, hour-of-day bucket (4 buckets), funding z-score, BTC's regime (for alts), stop distance / ATR15.
- **Training:** purged k-fold CV with an embargo of 1 day (trade labels overlap in time), then walk-forward.
- **Choice of `p*`:** chosen in-sample to maximise expectancy (not WR), then frozen.
- **Risk:** this is the most overfitting-prone component in the whole plan. It is only accepted if it improves **OOS expectancy by ≥ 0.05R** and doesn't cut the trade count below 150/year across the universe.

---

## 9. Testing methodology

### 9.1 Data
- **Source:** Binance public data archive (`data.binance.vision`), USDT-M futures:
  - `klines` 1m and 15m (1m is used for intrabar fill resolution; 15m/1h/4h are resampled from 1m to guarantee consistency),
  - `metrics` (5m open interest, long/short ratios),
  - `fundingRate`.
- **Period:** **2021-01-01 → latest full month** (≥ 5 years). The minimum requirement is 1 year, but 1 year covers only one or two regimes. This span includes the 2021 bull market, the 2022 bear market with cascades (LUNA, FTX), the 2023 chop and the 2024–25 bull market, which is what makes regime robustness testable.
- **Universe:** BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, LTC. Symbols are split into a **development set** (BTC, ETH, SOL, BNB, XRP, DOGE) and a **held-out set** (ADA, AVAX, LINK, LTC) that is never looked at until final validation (cross-sectional OOS).
- **Survivorship:** the universe is fixed *ex-ante* by today's list, which is survivorship-biased. Acknowledge it and add 2 delisted/decayed large caps if data exists (e.g. LUNA until May 2022, FTT until Nov 2022) as a stress test for Strategy C.

### 9.2 Time splits
```
2021-01 ────────── 2024-06 │ 2024-07 ─── 2025-06 │ 2025-07 ─── latest
   In-sample (design)      │ Out-of-sample (WF)  │ Lockbox (touch ONCE)
```
- **In-sample:** all parameter choices, on development symbols only.
- **Walk-forward (within IS+OOS):** 12-month train / 3-month test, rolled forward 3 months. Report every test window separately and concatenate test windows for the headline equity curve.
- **Lockbox:** final ~12 months, run exactly once per strategy version, on all 10 symbols. If it fails, the strategy is rejected or goes back to design. It is **not** re-tuned against the lockbox.

### 9.3 Cost model (conservative defaults; verify against the current Binance fee schedule)
| Component | Assumption |
|---|---|
| Fees | taker 0.05% per side (market entries, stops, time stops); maker 0.02% per side for limit TPs **only** if the fill model requires trading through the price |
| Spread (half) | BTC/ETH 0.5 bp; SOL/BNB/XRP/DOGE 1 bp; others 2 bp |
| Slippage (market) | BTC/ETH 1 bp; SOL/BNB/XRP/DOGE 3 bp; others 5 bp |
| Stop slippage | 2× market slippage; if the bar **opens** beyond the stop, fill at the open (gap) |
| SHOCK-regime slippage | 3× the above |
| Funding | actual historical funding charged/credited at 00/08/16 UTC for open positions |
| Stress test | every headline result is re-run at **2× total costs**; a strategy that goes negative there is marked fragile |

### 9.4 Fill simulation (critical for high-WR systems)
- Signals on bar close, fills at the **next bar open** (never at the signal bar's close).
- **Same-bar TP & SL ambiguity:** when a 15m bar spans both TP and SL, resolve it with 1m data. If the 1m bar also spans both, **assume SL first**. This single rule is the most common reason a "72% WR" backtest turns out to be 61% live.
- Limit orders fill only if price trades *through* the limit by ≥ 1 tick.
- No position sizing that uses future information (for example ATR from the current, unfinished bar).

### 9.5 Metrics (reported per strategy × symbol × regime × WF window, and pooled)
| Metric | Notes |
|---|---|
| Trades (N) | require N ≥ 200 OOS per strategy for conclusions (C may need pooling) |
| Win rate + **Wilson 95% CI** | net-P&L definition from §0.3 |
| **Excess hit rate** | `WR − 1/(1+b_effective)` |
| Average win (R), average loss (R), **average R per trade (expectancy)** | net of all costs |
| Profit factor | gross net-wins / gross net-losses |
| Max drawdown | in R and in % at 0.5% risk/trade |
| **Max consecutive losses** | compared against the expectation below |
| Sharpe / Sortino (daily) | plus **Deflated Sharpe Ratio** using the logged number of trials |
| Exposure, avg holding time, trades/week | |
| MAE/MFE distributions | used to check that stops/targets aren't sitting on a cliff |

**Expected losing streaks:** with loss probability `q = 0.30` over `N` trades, the longest losing streak is ≈ `log(N)/log(1/q)`. For N = 500 that is ≈ 5.2, and 7–8 happens regularly by chance. **Plan for 8 consecutive losses** at any claimed 70% WR. At 0.5% risk per trade that is a ~4% drawdown from streaks alone.

### 9.6 Robustness tests
1. **Parameter plateau:** perturb each tuned parameter ±20% (one at a time, then jointly on a small grid). OOS expectancy must stay > 0 over ≥ 80% of the neighbourhood. A lone peak means rejection.
2. **Monte Carlo:** 10,000 bootstrap resamples of the OOS trade sequence give distributions of max DD, max losing streak, and 12-month return. Report the 5th/95th percentiles.
3. **Randomised-entry null:** same exits, same regime gate, random entry bars (matched count). The strategy's excess hit rate must beat 95% of 1,000 null runs. This proves the *entry* has edge, not only the geometry.
4. **Timeframe shift:** re-run on 1h (A, B) and on 15m bars offset by 5 min (resample from 1m starting at :05). An edge that only exists at the :00 bar boundary is noise.
5. **Cost stress:** 2× costs (see 9.3).
6. **Symbol breadth:** positive expectancy on ≥ 7 of 10 symbols, including ≥ 3 of the 4 held-out ones.
7. **Regime/period breadth:** positive in ≥ 70% of walk-forward windows; no single calendar year contributing > 50% of total profit.

### 9.7 Ablation discipline
For each strategy, remove each filter one at a time and re-run OOS. A filter is kept only if removing it **reduces OOS expectancy** (not only WR) by more than the bootstrap standard error. Every trial run is logged (parameters, split, result) to an experiment log so the Deflated Sharpe calculation has an honest trial count.

### 9.8 Acceptance criteria (a strategy "passes" only if ALL hold on walk-forward OOS + lockbox)
| Criterion | Threshold |
|---|---|
| OOS trades | ≥ 200 (pooled symbols) |
| Expectancy | ≥ +0.10R per trade after costs; ≥ 0 at 2× costs |
| Profit factor | ≥ 1.30 |
| Win rate | Wilson 95% lower bound > breakeven WR + 3 pts. **Target** point estimate ≥ 65%; **70% is a goal, not a pass condition** |
| Excess hit rate | ≥ +8 pts over the geometric null, and beats 95% of the random-entry nulls |
| Max DD (0.5% risk/trade) | ≤ 12% on the historical path, ≤ 20% at the Monte Carlo 95th percentile |
| Breadth | ≥ 7/10 symbols positive; ≥ 70% of WF windows positive |
| Plateau | passes the ±20% neighbourhood test |

### 9.9 Order of investigation (recommended)
1. **Build the research harness first** (data loader, resampler, regime classifier, event-driven backtester with 1m fill resolution, cost model, metrics, experiment log). Validate it by showing that **random entries with fixed b produce WR ≈ 1/(1+b) and expectancy ≈ −c**. If they don't, the harness has a bug.
2. **Strategy A (SWEEP)**: most direct high-WR mechanism, needs only OHLCV + taker volume, and is frequent enough for fast statistical feedback.
3. **Strategy B (PULLBACK)**: complementary regime coverage; tests whether scale-out management delivers the WR/expectancy combination from §0.3.
4. **Strategy D (STATREV)**: cheap to add once A's infrastructure exists; the VR gate is a novel, testable hypothesis.
5. **Strategy C (CASCADE)**: needs the OI/funding pipeline; lowest frequency; potentially the most durable edge, and the lowest correlation with the others.
6. **Portfolio + meta-layer (§8)** only after ≥ 2 strategies pass individually.

---

## 10. Honest prior expectations

These are pre-test guesses, written down now so that later results can be judged against them rather than rationalised:

| Strategy | Expected OOS WR (net) | Expected effective b | Expected expectancy | Confidence it passes §9.8 |
|---|---|---|---|---|
| A SWEEP (scale) | 62–70% | ~1.0 | +0.05 to +0.20R | moderate |
| B PULLBACK (scale) | 60–70% | ~1.2 | +0.05 to +0.20R | moderate |
| C CASCADE | 60–72% | ~1.0 | +0.10 to +0.30R | moderate-low (sample size) |
| D STATREV | 63–72% | ~0.9 | −0.05 to +0.12R | low–moderate (cost-sensitive) |

**Bottom line:** a sustained ≥ 70% win rate *with* positive expectancy after costs is at the top edge of what is realistic for 15m crypto majors. The most plausible way to get there is **regime-gated entries at objective locations, plus scale-out exit management at an effective R:R of 0.8–1.2**. If OOS results land at 62–67% WR with expectancy ≥ +0.1R and PF ≥ 1.3, that is a *good* strategy, and it should not be degraded by tuning just to reach the 70% number.

*This is research material, not financial advice. Leveraged crypto trading can lose more than the initial margin.*

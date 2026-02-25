# Best Trend — Strategy Overview + Canonical Pipeline (XAUUSD)

**Last updated:** 2026-02-25 (Europe/Berlin)

This document describes the current **best_trend** variant in two layers:

1) **Overview (stable):** what the strategy is trying to do and which types of controls it uses.
2) **Canonical pipeline (precise):** the exact gate order + parameters as configured in `configs/trend_based/best_sma_trend.yaml`.

Reference reports:
- Equity/perf: `reports/trend_based/best_sma_trend.html`
- Trades: `reports/trend_based/best_sma_trend_trades.html`
- Yearly equity/perf: `reports/trend_based/best_sma_trend_yearly.html`

---

## 0) SMA results (snapshot)

Holdout policy: **2026 is excluded from header score aggregates** (`score_exclude: ['2026']`).

From `reports/trend_based/best_sma_trend.html`:
- Sum PnL (scored): **417.70%**
- Worst MaxDD (scored): **-16.33%**
- Avg Sharpe (scored): **1.47**
- # Trades: **1,434**
- Per period:
  - 2020–2022: **51.50%** (vs B&H **+31.52%**), MaxDD **-16.33%**, Sharpe **0.70** (CI **-0.29/1.57**)
  - 2023–2025: **366.20%** (vs B&H **+229.85%**), MaxDD **-11.98%**, Sharpe **2.25** (CI **1.28/3.13**)
  - 2026 (holdout): **172.07%** (vs B&H **+151.07%**), MaxDD **-16.92%**, Sharpe **5.40** (CI **1.48/8.56**)

---

## 1) Overview (what it is)

**Hypothesis:** long-only trend following on XAUUSD where most value comes from (a) staying out of chop and (b) cutting toxic mid-duration losers, while allowing long holds to run.

**Core structure:**
- **Base signal:** 5m SMA crossover (fast/slow)
- **Entry filters (regime):** EMA separation + NoChop
- **Time filter:** force-flat around scheduled macro windows (FOMC / econ calendar) + seasonal blockouts (e.g. flat June)
- **Sizing overlays:** size up on strong trends
- **Trade frequency control:** churn debounce + cooldown
- **Post-entry exits:** kill-switches for mid-duration losers, no-recovery trades, and shock days

**Meta-order (recommended):**

`base signal → entry filters → time filter → sizing overlays → trade frequency control → post-entry exits`

---

## 2) Canonical config (where truth lives)

Canonical SMA config:
- `configs/trend_based/best_sma_trend.yaml`

Report generation:
- `from quantlab.webui.runner import run_backtest; run_backtest("best_trend")` (uses `configs/trend_based/current.yaml`, so keep in mind `current.yaml` may point to EMA now)
- For the SMA report files above, we treat `best_sma_trend.html` as the reference snapshot.

Holdout policy:
- Periods can define `periods.score_exclude` (e.g. `["2026"]`) to show holdouts in output while excluding them from `quantlab.rnd` scoring.

---

## 3) Account / execution model (engine semantics)

- Starting capital: **$1,000**
- Leverage / margin: **20:1** (required margin = notional / 20)
- Contract: XAUUSD, **100 oz per 1.00 lot**
- Sizing: discrete sizes map to lots via `BacktestConfig.lot_per_size` (canonical configs use 0.01/0.02 lots via size 1/2)
- Execution lag: **trade on next bar** (`lag = 1`)
- Margin policy: **skip_entry** (if required margin > equity on entry, remain flat)
- Costs (absolute per lot per side):
  - `fee_per_lot = 3.50`
  - `spread_per_lot = 3.50`
  - (≈ $14 / lot roundtrip)

---

## 4) Canonical pipeline (exact order + params)

Source of truth: `configs/trend_based/best_sma_trend.yaml` → `trend:` + `pipeline:`

**Base signal:** SMA crossover on 5m close (`fast=30`, `slow=75`).

1. **ema_sep** — HTF EMA separation filter (ATR-scaled)
   - params: `ema_fast=50`, `ema_slow=300`, `atr_n=20`, `sep_k=0.065`

2. **nochop** — HTF NoChop filter
   - params: `ema=20`, `lookback=40`, `min_closes=20`, `entry_held=false`
   - Note: `exit_bad_bars` logic was removed.

3. **time_filter** — force-flat allow-mask
   - params: `{}`
   - mask is built by the runner from `time_filter:` in `configs/trend_based/best_sma_trend.yaml` (includes FOMC + month blockouts like flat June).

4. **ema_strength_sizing** — size up on strong separation (segment-held)
   - params: `ema_fast=40`, `ema_slow=300`, `atr_n=20`, `strong_k=0.20`, `size_base=1.0`, `size_strong=2.0`

5. **churn** — debounce + cooldown (trade frequency control)
   - params: `min_on_bars=3`, `cooldown_bars=8`

6. **mid_loss_limiter** — mid-duration loss kill-switch
   - params: `min_bars=13`, `max_bars=48`, `stop_ret=-0.010`

7. **no_recovery_exit** — no-recovery exit
   - params: `bar_n=24`, `min_ret=-0.005`

8. **shock_exit** — shock day kill-switch (+ optional cooldown)
   - params: `shock_exit_abs_ret=0.006`, `shock_exit_sigma_k=0.0`, `shock_exit_sigma_window=96`, `shock_cooldown_bars=0`

---

## 5) Data inputs

- Base prices: 5m OHLC close (XAUUSD)
- HTF bars: 15m OHLC (XAUUSD)
- Corr series: optional (disabled in canonical pipeline)
- Time filter calendar: depends on `time_filter.kind` (canonical: `fomc` around 19:00 UTC, pre=2h post=0.5h)

---

## 6) Implementation pointers

- Strategy + pipeline construction:
  - `strategies/trend_following.py` → `TrendStrategyWithGates`
- Gates:
  - `strategies/gates/filters.py`
  - `strategies/gates/sizing.py`
  - `strategies/gates/churn.py`
  - `strategies/gates/exits.py`
- Gate registry + config instantiation:
  - `strategies/gates/registry.py`
- Backtest engine:
  - `engine/backtest.py` → `backtest_positions_account_margin`
- Runner (canonical wiring: data load + allow_mask + report generation):
  - `webui/runner.py`

---

## 7) Notes / guardrails

- When changing anything that affects execution semantics (gates, costs, time-filtering, return math, trade extraction), run the golden regression:
  - `./check.sh`

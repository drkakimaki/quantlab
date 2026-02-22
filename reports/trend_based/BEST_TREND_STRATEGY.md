# Best Trend Variant — Strategy Logic & Parameters (XAUUSD)

**Last updated:** 2026-02-22 (Europe/Berlin) — corr removed; sizing via EMA-strength; added post-entry time-stop + mid-loss limiter

This document describes the current **best trend-following variant** under the (now-default) **account/margin backtest model**.

Key instrument/execution assumptions are summarized once below under **Account / execution model**.

Reference report:
- `reports/trend_based/best_trend.html`

---

## Active config (as currently run)

**Periods (as used in the canonical report config):**
- `2020-2022`: 2020-01-01 → 2022-12-31
- `2023-2025`: 2023-01-01 → 2025-12-31
- `2026`: 2026-01-01 → today

**Data sources:**
- Base: 5m OHLC close (`data/dukascopy_5m_ohlc/XAUUSD/...`)
- HTF: 15m OHLC (`data/dukascopy_15m_ohlc/XAUUSD/...`)
- Corr series: **not used** (corr gate disabled)

**Enabled modules (best config):**
- HTF EMA separation filter: EMA40/300, TR-ATR20, k=0.05
- HTF NoChop filter: EMA20, lookback=40, min_closes=24
- EMA-strength sizing (entry-only, segment-held):
  - Size=2 when EMA separation is *strong*: `(EMA40-EMA300) > strong_k * ATR20` on HTF (15m)
  - Canonical: `strong_k=0.20`
- FOMC time filter (force-flat):
  - Force flat around **19:00 UTC (pre=2h, post=0.5h)** on FOMC decision days
  - Source: `data/econ_calendar/fomc_decision_days.csv` (date-only)
- Shock exit (kill-switch):
  - If **abs(5m return) ≥ 0.006 (0.6%)** on the last closed bar → force flat for the rest of the segment
  - Purpose: cut the tail risk on the worst drawdown-deepening days
- Churn gate (enabled):
  - Entry debounce: `min_on_bars=3`
  - Re-entry cooldown: `cooldown_bars=8`
- Mid-duration loss limiter (enabled):
  - If a segment is within **13–48 bars since entry** and unrealized return <= **-1.0%**, kill the remainder of the segment.
- Time-stop (enabled):
  - If a segment is **>=24 bars old** and unrealized return is still <= **-0.5%**, kill the remainder of the segment.
- Seasonality size cap (enabled):
  - June soft cap: size <= 1.0 during June (`seasonality.month_size_cap: {6: 1.0}`)
- Corr stability gate: **OFF**
- Sizing tiers: base=1.0, strong=2.0 (mapped to 0.01/0.02 lots)

**Account / execution model:**
- Starting capital: $1,000
- Leverage / margin: 20:1 (required margin = notional / 20)
- Contract: XAUUSD, 100 oz per 1.00 lot
- Allowed lot sizes: 0.01, 0.02 (mapped from sizing)
- Execution model: trade on next bar (engine lag = 1)
- Entry handling when margin is insufficient: **skip entry** (remain flat)
- Costs (IC Markets rough baseline):
  - `fee_per_lot = 3.50` ($ per lot per side)
  - `spread_per_lot = 3.50` ($ per lot per side)
  - Total ≈ $14 / lot roundtrip (commission + spread)

---

## Strategy logic (only what exists in the clean implementation)

### 1) Base signal (5m)
- SMA30/75 on 5m close
- Long-only: on when `SMA30 > SMA75`

### 2) HTF confirm (15m)
- SMA30/75 on 15m close
- Gate is forward-filled to 5m using the last closed 15m bar

### 3) Filters / gates
These are applied as gates (must all be true to allow a segment):

- **EMA separation (HTF)**
  - `EMA40_htf > EMA300_htf`
  - `(EMA40_htf - EMA300_htf) > 0.05 * ATR20_htf`

- **NoChop (HTF)**
  - EMA20 on HTF close
  - over last 40 HTF bars: ≥24 closes above EMA20

### 4) EMA-strength sizing (entry-only, held for the segment)
- Use HTF (15m) EMA separation strength to size up.
- Base size = 1.
- Strong size = 2 when `(EMA_fast - EMA_slow) > strong_k * ATR_n` (computed on HTF).
- Uses last closed HTF info (shifted by 1 bar) and is held for the whole segment.

### 5) FOMC time filter (entry-only)
- On dates listed in `data/econ_calendar/fomc_decision_days.csv`, force flat within an approximate decision window:
  - **19:00 UTC (pre=2h, post=0.5h)** (date-only approximation; ignores DST shifts)
- Semantics: **force flat** (do not hold through the window).

### 6) Churn gate (enabled)
- Entry debounce: delay entry until signal has been ON for N consecutive bars (`churn.min_on_bars`).
  - Canonical: `min_on_bars=3`
- Re-entry cooldown: after an exit, block new entries for C bars (`churn.cooldown_bars`).
  - Canonical: `cooldown_bars=8`

### 7) Post-entry controls (enabled)

- **Mid-duration loss limiter**
  - Within a segment, if `13 <= bars_since_entry <= 48` and unrealized return since entry <= -1.0% (using last closed bar), kill the remainder of the segment.

- **Time-stop**
  - If `bars_since_entry >= 24` and unrealized return since entry <= -0.5% (using last closed bar), kill the remainder of the segment.

### 8) Shock exit (kill-switch)
- Compute 5m close-to-close returns on the base series.
- If **abs(return) ≥ 0.006 (0.6%)** on the *last closed bar* (shifted by 1 bar), then:
  - force flat for the remainder of the current segment
- This is intentionally discrete (no fractional sizing), aligned with 0.01/0.02 lot constraint.

### 9) Sizing
At entry:
- Default size = 1.0.
- If EMA-strength condition is true on the last closed bar → size = 2.0.
- Seasonality cap may reduce size (e.g. June cap to 1.0).

---

## On/off semantics

This strategy is configured via `configs/trend_based/current.yaml`.

Module convention:
- a module is **ON** if its config block is present (non-null)
- a module is **OFF** if the block is missing/null

Notable blocks:
- `churn:` (debounce + cooldown) — ON in canonical config
- `risk:` (shock exit) — ON in canonical config

---

## Implementation pointers

- Strategy implementation:
  - `strategies/trend_following.py` → `TrendStrategyWithGates`
  - Gates: `HTFConfirmGate`, `EMASeparationGate`, `NoChopGate`, `TimeFilterGate`, `EMAStrengthSizingGate`, `SeasonalitySizeCapGate`, `ChurnGate`, `MidDurationLossLimiterGate`, `TimeStopGate`, `RiskGate`
- Backtest engine (unified):
  - `engine/backtest.py` → `backtest_positions_account_margin`
- Report generation:
  - `reporting/generate_bt_report.py` (equity/performance HTML)
  - `reporting/generate_trades_report.py` (trade breakdown HTML)
- WebUI runner (canonical wiring for best_trend):
  - `webui/runner.py`

---

## Canonical config + CLI used (for the reference report)

Canonical config is defined in:
- `configs/trend_based/current.yaml`

Generate the report (uses the YAML defaults; CLI flags override YAML for experiments):

```bash
.venv/bin/python scripts/report_trend_variants.py \
  --config configs/trend_based/current.yaml \
  --out-dir reports/trend_based --out-name best_trend.html
```

To experiment with turning modules off:
- `--no-ema-sep`
- `--no-nochop`
- `--no-corr`
- `--no-fomc`


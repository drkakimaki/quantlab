# Best Trend Variant — Strategy Logic & Parameters (XAUUSD)

**Last updated:** 2026-02-21 (Europe/Berlin)

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
- Corr series (5m OHLC close): XAGUSD + EURUSD

**Enabled modules (best config):**
- HTF EMA separation filter: EMA40/300, TR-ATR20, k=0.05
- HTF NoChop filter: EMA20, lookback=40, min_closes=24
- Corr stability (entry-only, segment-held):
  - XAGUSD(win=40, abs≥0.10, flips≤0/50) OR EURUSD(win=75, abs≥0.10, flips≤5/75)
- FOMC time filter (force-flat):
  - Force flat around **19:00 UTC (pre=2h, post=0.5h)** on FOMC decision days
  - Source: `data/econ_calendar/fomc_decision_days.csv` (date-only)
- Shock exit (kill-switch):
  - If **abs(5m return) ≥ 0.006 (0.6%)** on the last closed bar → force flat for the rest of the segment
  - Purpose: cut the tail risk on the worst drawdown-deepening days
- Churn gate (optional, currently OFF in canonical config):
  - Entry debounce (`min_on_bars`) + re-entry cooldown (`cooldown_bars`) to reduce toxic churn
- Sizing mode: confirm (one=1.0, both=2.0 → sizes map to 0.01/0.02 lots)

**Account / execution model:**
- Starting capital: $1,000
- Leverage / margin: 20:1 (required margin = notional / 20)
- Contract: XAUUSD, 100 oz per 1.00 lot
- Allowed lot sizes: 0.01, 0.02 (mapped from sizing)
- Execution model: trade on next bar (engine lag = 1)
- Entry handling when margin is insufficient: **skip entry** (remain flat)

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

### 4) Corr stability + sizing (entry-only, held for the segment)
- Compute rolling corr stability on 5m returns.

### 5) FOMC time filter (entry-only)
- On dates listed in `data/econ_calendar/fomc_decision_days.csv`, force flat within an approximate decision window:
  - **19:00 UTC (pre=2h, post=0.5h)** (date-only approximation; ignores DST shifts)
- Semantics: **force flat** (do not hold through the window).

### 6) Churn gate (optional)
- Entry debounce: delay entry until signal has been ON for N consecutive bars (`churn.min_on_bars`).
- Re-entry cooldown: after an exit, block new entries for C bars (`churn.cooldown_bars`).
- Current canonical config: OFF (no `churn:` block).

### 7) Shock exit (kill-switch)
- Compute 5m close-to-close returns on the base series.
- If **abs(return) ≥ 0.006 (0.6%)** on the *last closed bar* (shifted by 1 bar), then:
  - force flat for the remainder of the current segment
- This is intentionally discrete (no fractional sizing), aligned with 0.01/0.02 lot constraint.

- Stability condition for each corr series:
  - `abs(corr) >= min_abs` and `flip_count <= max_flips` over `flip_lookback`
- Combine: **XAG stable OR EUR stable**.

At entry:
- If corr stability is false on the last closed bar, the entire segment is blocked.
- If allowed:
  - exactly one stable → size=1.0 (0.01 lots)
  - both stable → size=2.0 (0.02 lots)

---

## On/off semantics

This strategy is configured via `configs/trend_based/current.yaml`.

Module convention:
- a module is **ON** if its config block is present (non-null)
- a module is **OFF** if the block is missing/null

Notable optional blocks:
- `churn:` (debounce + cooldown) — OFF unless explicitly configured
- `risk:` (shock exit) — ON in canonical config

---

## Implementation pointers

- Strategy implementation:
  - `strategies/trend_following.py` → `TrendStrategyWithGates`
  - Gates: `HTFConfirmGate`, `EMASeparationGate`, `NoChopGate`, `CorrelationGate`, `TimeFilterGate`, `ChurnGate`, `RiskGate`
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


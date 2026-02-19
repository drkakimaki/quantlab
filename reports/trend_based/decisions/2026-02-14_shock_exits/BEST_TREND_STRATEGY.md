# Best Trend Variant — Strategy Logic & Parameters (XAUUSD)

This document describes the current **best trend-following variant** under the (now-default) **account/margin backtest model**.

- **Starting capital:** $1,000
- **Leverage / margin:** 20:1 (required margin = notional / 20)
- **Instrument:** XAUUSD
- **Contract size:** 100 oz per 1.00 lot
- **Allowed lot sizes:** 0.01, 0.02 (mapped from sizing)
- **Execution model:** trade on next bar (engine lag = 1)
- **Entry handling when margin is insufficient:** **skip entry** (remain flat)

Reference report:
- `reports/trend_based/best_trend.html`

---

## Active config (as currently run)

**Periods (as used in the report):**
- `2021-2022`: 2021-01-01 → 2022-12-31
- `2023-2025`: 2023-01-01 → 2025-12-31
- `2026`: 2026-01-01 → 2026-02-13

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
- Sizing mode: confirm (one=1.0, both=2.0 → sizes map to 0.01/0.02 lots)

**Account / execution model:**
- initial_capital=1000
- leverage=20
- lag=1 bar (trade on next bar)
- margin policy: skip entry when insufficient margin

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

- Stability condition for each corr series:
  - `abs(corr) >= min_abs` and `flip_count <= max_flips` over `flip_lookback`
- Combine: **XAG stable OR EUR stable**.

At entry:
- If corr stability is false on the last closed bar, the entire segment is blocked.
- If allowed:
  - exactly one stable → size=1.0 (0.01 lots)
  - both stable → size=2.0 (0.02 lots)

---

## On/off arguments in code

The clean code supports toggling modules on/off (defaults match the best config):
- `ema_sep_filter` (default True)
- `nochop_filter` (default True)
- `corr_filter` (default True)

If `corr_filter=False`, sizing simplifies to a fixed `confirm_size_one` when the non-corr gates are on.

---

## Implementation pointers

- Strategy implementation:
  - `src/quantlab/strategies/trend_following.py` → `trend_following_ma_crossover_htf_confirm`
- Backtest engine (unified):
  - `src/quantlab/backtest.py` → `backtest_positions_account_margin`
- Report runner (clean):
  - `scripts/report_trend_variants.py`

---

## Exact CLI used (for the reference report)

```bash
.venv/bin/python scripts/report_trend_variants.py \
  --symbol XAUUSD \
  --corr-symbol XAGUSD \
  --corr2-symbol EURUSD \
  --out-dir reports/trend_based --out-name best_trend.html \
  --p3-end 2026-02-13
```

To experiment with turning modules off:
- `--no-ema-sep`
- `--no-nochop`
- `--no-corr`
- `--no-fomc`


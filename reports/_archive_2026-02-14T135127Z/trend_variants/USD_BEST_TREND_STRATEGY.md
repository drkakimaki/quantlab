# Best Trend USD Variant — Strategy Logic & Parameters (XAUUSD)

This document describes the current **best trend-following variant** under the **USD account backtest model**:

- **Starting capital:** $1,000
- **Leverage / margin:** 20:1 (required margin = notional / 20)
- **Instrument:** XAUUSD
- **Contract size:** 100 oz per 1.00 lot
- **Allowed lot sizes:** 0.01, 0.02 (mapped from sizing)
- **Execution model:** trade on next bar (engine lag = 1)
- **Entry handling when margin is insufficient:** **skip entry** (remain flat)

Reference report:
- `reports/trend_variants/usd_account/base_best_usd_1000_20x_size1_2.html`

---

## Data & Timeframes

- **Base timeframe:** 5-minute bars
  - Source: `data/dukascopy_5m_ohlc/XAUUSD/...` (use **5m OHLC close**)
- **Higher timeframe (HTF):** 15-minute bars
  - Source: `data/dukascopy_15m_ohlc/XAUUSD/...` (use **15m OHLC** for true-range ATR)
- **Timezone:** UTC (timestamps are UTC)

---

## Core Signal (Trend)

### Base trend signal (5m)
- Compute **SMA fast/slow** on 5m close:
  - `fast = 30`
  - `slow = 75`
- **Long-only signal:**
  - `base_signal = 1` if `SMA30 > SMA75`, else `0`

### HTF confirmation (15m)
- Resample/align using the **last closed 15m bar**, forward-filled to 5m.
- Compute 15m SMA fast/slow (defaults to base fast/slow because no overrides used):
  - `htf_fast = 30`
  - `htf_slow = 75`
- `htf_signal = 1` if `SMA30_htf > SMA75_htf`, else `0`

### Base position gate
- Before filters: `pos = base_signal * htf_signal` (aligned to 5m)

---

## HTF Filters

### A) EMA separation filter (HTF)
Enabled.

Parameters:
- `ema_fast = 50`
- `ema_slow = 250`
- `atr_n = 14`
- `sep_k = 0.15`

Logic (evaluated on 15m bars; forward-filled to 5m):
- Compute HTF EMAs on 15m close: `EMA50_htf`, `EMA250_htf`
- Compute **true-range ATR** on 15m OHLC:
  - `TR = max(high-low, |high-prev_close|, |low-prev_close|)`
  - `ATR = SMA(TR, atr_n)`
- Pass if:
  - `EMA50_htf > EMA250_htf`
  - and `(EMA50_htf - EMA250_htf) > sep_k * ATR`

### B) No-chop filter (HTF)
Enabled.

Parameters:
- `nochop_ema = 15`
- `nochop_lookback = 20` (HTF bars)
- `nochop_min_closes = 12`

Logic:
- Compute `EMA15_htf` on 15m close.
- Over the last 20 HTF candles, require at least 12 closes above `EMA15_htf`.

### C) HTF slope filter
Not enabled in this variant.

---

## Base timeframe filter

### Base slope filter
Enabled.

Parameters:
- `base_slope_ema = 100`
- `base_slope_window = 30`
- `base_slope_eps = 0.0`

Logic:
- Compute `EMA100` on 5m close.
- Slope approximation:
  - `slope = (EMA100 - EMA100.shift(window)) / window`
- Require `slope > 0`.

---

## Cross-asset confirmation

### Beta confirmation (XAGUSD)
Not used in this variant.

---

## Regime Stability Filter (Rolling Correlation)
Enabled. This is an **entry-only filter held through the trade segment**.

We compute rolling correlation of **5m returns** and measure stability (absolute corr + sign flip count).

### Corr series 1: XAGUSD vs XAUUSD
- `corr_symbol = XAGUSD`
- `corr_window = 50`
- `corr_min_abs = 0.25`
- `corr_flip_lookback = 100`
- `corr_max_flips = 1`

### Corr series 2: EURUSD vs XAUUSD
- `corr2_symbol = EURUSD`
- `corr2_window = 75`
- `corr2_min_abs = 0.20`
- `corr2_flip_lookback = 50`
- `corr2_max_flips = 4`

### Combination logic
- `corr_logic = "or"`

### Stability definition
For each corr series:
1) Compute rolling correlation `c` between base returns and corr-symbol returns.
2) Compute sign series `sign(c)` (0 treated as neutral).
3) Count sign flips over `flip_lookback` window.
4) Stable if:
   - `abs(c) >= min_abs`
   - and `flip_count <= max_flips`

### Application (entry-only)
- Define `entry_bar` when position goes from 0 -> >0.
- Entry is allowed only if stability condition is true on the **last closed bar** (shifted by 1 in code).
- If entry is allowed, the whole segment is allowed; otherwise the entire segment is blocked.

---

## Position Sizing (Lots)

Mode: `sizing_mode = "confirm"`

Allowed sizing levels are discrete; for this USD account model we ultimately map to:
- size 0 → **0.00 lots**
- size 1 → **0.01 lots**
- size 2 → **0.02 lots**

Sizing parameters:
- `confirm_size_one = 1.0`
- `confirm_size_both = 2.0`

Interpretation:
- If **exactly one** corr stability passes (OR logic true but not both), size = 1.0
- If **both** corr stabilities pass, size = 2.0

Sizing is applied **at entry time** and held constant through the segment.

---

## Exits

- Default exit is the base trend crossover turning off (and any filters gating the segment), plus any enabled exit filters.
- **Exit EMA trail:** not enabled in this variant.

---

## Costs

- `fee_bps`: as configured at run time (default 0)
- `slippage_bps`: as configured at run time (default 0)

Applied on turnover (change in units) in the USD backtest engine.

---

## Implementation pointers

- Strategy implementation:
  - `src/quantlab/strategies/trend_following.py` → `trend_following_ma_crossover_htf_confirm`
- USD account backtest engine:
  - `src/quantlab/backtest_usd.py` → `backtest_positions_usd_margin`
- Report runner:
  - `scripts/report_trend_variants.py`

---

## Exact CLI used (for the reference report)

```bash
.venv/bin/python scripts/report_trend_variants.py \
  --symbol XAUUSD --use-ohlc-base --use-htf-ohlc \
  --fast 30 --slow 75 --htf-rule 15min \
  --ema-sep-filter --ema-fast 50 --ema-slow 250 --atr-n 14 --sep-k 0.15 \
  --nochop-filter --nochop-ema 15 --nochop-lookback 20 --nochop-min-closes 12 \
  --base-slope-filter --base-slope-ema 100 --base-slope-window 30 --base-slope-eps 0.0 \
  --corr-filter --corr-symbol XAGUSD --corr-window 50 --corr-min-abs 0.25 --corr-flip-lookback 100 --corr-max-flips 1 \
  --corr2-symbol EURUSD --corr-logic or --corr2-window 75 --corr2-min-abs 0.20 --corr2-flip-lookback 50 --corr2-max-flips 4 \
  --sizing-mode confirm --confirm-size-one 1.0 --confirm-size-both 2.0 \
  --out-dir reports/trend_variants/usd_account --out-name base_best_usd_1000_20x_size1_2.html
```

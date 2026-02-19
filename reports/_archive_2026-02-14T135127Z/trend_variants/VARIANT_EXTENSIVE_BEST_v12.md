# Trend variant v12 — “Extensive search best” (OHLC-based)

This document describes the **current best-performing variant** found by the extensive OHLC-based hyperparameter search.

- **Symbol (typical):** XAUUSD
- **Base timeframe:** **5-minute OHLC** bars, strategy uses **5m close** as the price series
- **Higher timeframe (HTF):** **15-minute OHLC** bars, strategy uses **15m close** for MAs/EMAs and **true-range ATR(15m)**
- **Direction:** long-only

Artifacts:
- **Report (generated):** `reports/trend_variants/v12_extensive_best.html`
- **Search results (CSV):** `reports/trend_variants/extensive_search_results.csv`

---

## Parameter set (best found)

### Base (5m) trend signal
- `fast = 30`
- `slow = 75`

Base signal:
- `base_signal = 1 if SMA(close_5m, fast) > SMA(close_5m, slow) else 0`

### HTF confirmation (15m)
- `htf_rule = "15min"`
- `htf_fast = fast` (defaults to 30)
- `htf_slow = slow` (defaults to 75)

HTF trend signal:
- `htf_signal = 1 if SMA(close_15m, 30) > SMA(close_15m, 75) else 0`

### Option A — EMA separation filter (15m)
Enabled: `ema_sep_filter = True`

Params:
- `ema_fast = 50`
- `ema_slow = 250`
- `atr_n = 14`
- `sep_k = 0.15`

Filter condition:
- Require `EMA50(15m) > EMA250(15m)` AND
- `(EMA50 - EMA250) > sep_k * ATR_TR(15m, 14)`

where ATR_TR is the **true-range ATR** computed from 15m OHLC:
- `prev_close = close.shift(1)`
- `TR = max(high-low, abs(high-prev_close), abs(low-prev_close))`
- `ATR_TR = SMA(TR, 14)`

### Option B — “No-chop” filter (15m EMA touches)
Enabled: `nochop_filter = True`

Params:
- `nochop_ema = 15`
- `nochop_lookback = 20`
- `nochop_min_closes = 12`

Filter condition (for longs):
- Compute `EMA15(15m)`
- Over the last 20 HTF candles, count closes above EMA15:
  - `above = 1 if close_15m > EMA15 else 0`
  - `above_cnt = SUM(above over last 20)`
- Require `above_cnt >= 12`

### HTF slope filter
- `htf_slope_filter = False` (disabled in the best run)

### Base slope filter (5m)
Enabled: `base_slope_filter = True`

Params:
- `base_slope_ema = 100`
- `base_slope_window = 30`
- `base_slope_eps = 0.0`

Condition:
- `EMA100_5m = EMA(close_5m, 100)`
- `slope = (EMA100_5m - EMA100_5m.shift(30)) / 30`
- Require `slope > 0.0`

---

## Combined gating logic

1) Compute base signal on 5m close:
- `base_signal ∈ {0,1}`

2) Compute HTF gate on 15m bars:
- `htf_gate_15m = htf_signal * ema_sep_ok * nochop_ok`

3) Align HTF gate to 5m timestamps (use last closed HTF bar):
- `htf_gate_5m = htf_gate_15m.reindex(close_5m.index, method="ffill").fillna(0)`

4) Apply base slope filter (5m):
- `base_pos = base_signal` but set to 0 when `EMA100_5m slope <= 0`

5) Final position (long-only):
- `pos_5m = base_pos * htf_gate_5m`

---

## Backtest conventions (engine)

This strategy uses `quantlab.backtest.backtest_positions`:

- Returns: `prices_to_returns(close_5m)` i.e. `pct_change()`
- **Engine lag = 1 bar** (default):
  - positions decided at bar *t* are applied to returns of bar *t+1*
  - this reduces lookahead when using close-based indicators
- Transaction costs:
  - `cost_rate = (fee_bps + slippage_bps) * 1e-4`
  - `costs = abs(delta_position) * cost_rate`

Defaults used in this variant run:
- `fee_bps = 0.0`
- `slippage_bps = 0.0`

---

## How to reproduce (generate the v12 report)

```bash
.venv/bin/python scripts/report_trend_variants.py \
  --symbol XAUUSD \
  --use-ohlc-base \
  --use-htf-ohlc \
  --fast 30 --slow 75 \
  --ema-sep-filter --ema-fast 50 --ema-slow 250 --atr-n 14 --sep-k 0.15 \
  --nochop-filter --nochop-ema 15 --nochop-lookback 20 --nochop-min-closes 12 \
  --base-slope-filter --base-slope-ema 100 --base-slope-window 30 \
  --out-name v12_extensive_best.html
```

---

## Notes / caveats

- The search score used was: **Sharpe + 2×max_drawdown** (where max_drawdown is negative). This rewards high Sharpe while penalizing deeper drawdowns.
- This is still an **in-sample** style selection across the same three periods (2022, 2023–2025, 2026-to-date). For extra confidence:
  - re-run the search on 2022–2024 and validate on 2025–2026,
  - or do a simple walk-forward / rolling split.
- Because this is long-only and heavily filtered, it may underperform in strongly range-bound regimes (but should keep DD lower).

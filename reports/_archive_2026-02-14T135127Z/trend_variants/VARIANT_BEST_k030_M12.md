# Trend variant — HTF-confirm + EMA separation + No-chop (best: k=0.30, M=12)

This describes the current “best” trend-following variant found by the recent (k, M) grid search.

- **Base timeframe:** 5-minute OHLC bars; strategy uses the **5m close** as the price series
- **Higher timeframe (HTF):** 15-minute OHLC bars; strategy uses the **15m close** for MAs/EMA filters, and **true-range ATR(15m)** for the separation filter
- **Direction:** long-only

Output report:
- `reports/trend_variants/v06_best_k030_M12_OHLCbased.html`

---

## Summary of logic

We trade a simple **SMA crossover** on the base timeframe (5m), but only when the higher timeframe (15m) agrees and passes two additional HTF filters:

1) **HTF trend agreement (15m SMA crossover)**
2) **Option A — EMA separation filter (15m)**
3) **Option B — “No-chop” filter (15m EMA20 touches)**

Final position is:

- `pos(t) = base_pos(t) * htf_gate(t)`

where `htf_gate(t)` is forward-filled from 15m onto 5m timestamps (so we use the most recently closed 15m bar).

---

## Base strategy (5m): SMA crossover

Inputs:
- `prices_5m`: 5m close/mid series

Parameters:
- `fast` (int): **20**
- `slow` (int): **100**
- `long_only` (bool): **True**

Computation:
- `sma_fast_5m = SMA(prices_5m, fast)`
- `sma_slow_5m = SMA(prices_5m, slow)`

Signal/position:
- `base_signal = 1 if sma_fast_5m > sma_slow_5m else 0`
- `base_pos = base_signal` (long when 1, flat when 0)

---

## HTF confirmation (15m): SMA crossover

We use the precomputed 15m OHLC bars from:
- `data/dukascopy_15m_ohlc/{SYMBOL}/...`

(Internally we still forward-fill HTF gates onto 5m timestamps, which corresponds to using the **last closed 15m bar**.)

Parameters:
- `htf_rule`: **"15min"**
- `htf_fast`: defaults to `fast` (**20**) if not specified
- `htf_slow`: defaults to `slow` (**100**) if not specified

Computation:
- `sma_fast_15m = SMA(prices_15m, htf_fast)`
- `sma_slow_15m = SMA(prices_15m, htf_slow)`

HTF trend gate:
- `htf_trend = 1 if sma_fast_15m > sma_slow_15m else 0`

---

## Option A (15m): EMA separation filter

Goal: only take longs when the 15m trend has meaningful separation (avoid weak/flat regimes).

Parameters:
- `ema_sep_filter`: **True**
- `ema_fast`: **50**
- `ema_slow`: **200**
- `atr_n`: **14**
- `sep_k`: **0.30**  ← **best k** from the search

Computation on 15m closes:
- `ema50 = EMA(prices_15m, 50)`
- `ema200 = EMA(prices_15m, 200)`

ATR (true range, using 15m OHLC):
- `prev_close = close.shift(1)`
- `tr = max(high-low, abs(high-prev_close), abs(low-prev_close))`
- `atr = SMA(tr, atr_n)`

Pass condition:
- `ema_sep_ok = 1 if (ema50 > ema200) and ((ema50 - ema200) > sep_k * atr) else 0`

---

## Option B (15m): “No-chop” filter via EMA20 touches

Goal: avoid mean-reverting/choppy periods that kill trend systems.

Parameters:
- `nochop_filter`: **True**
- `nochop_ema`: **20**
- `nochop_lookback`: **20** (candles)
- `nochop_min_closes`: **12**  ← **best M** from the search

Computation on 15m closes:
- `ema20 = EMA(prices_15m, 20)`
- `above = 1 if close > ema20 else 0`
- `above_cnt = SUM(above over last 20 HTF candles)`

Pass condition (for longs):
- `nochop_ok = 1 if above_cnt >= 12 else 0`

---

## HTF gate alignment (15m → 5m)

We combine the HTF gates on the 15m index:

- `htf_gate_15m = htf_trend * ema_sep_ok * nochop_ok`

Then align to 5m timestamps using forward-fill:

- `htf_gate_5m = htf_gate_15m.reindex(prices_5m.index, method="ffill").fillna(0)`

This implements the “use **last closed 15m bar**” requirement.

---

## Final position and backtest conventions

Final trading position:
- `pos_5m = base_pos_5m * htf_gate_5m`

Backtest engine details (see `quantlab.backtest.backtest_positions`):
- Returns are computed as `prices.pct_change()` on the base series.
- **Engine lag = 1 bar** by default:
  - a position decided at bar *t* applies to returns of bar *t+1*.
- Transaction costs are applied on turnover:
  - `cost_rate = (fee_bps + slippage_bps) * 1e-4`
  - `costs = abs(delta_position) * cost_rate`

Parameters used in our reports (unless overridden):
- `fee_bps`: 0.0
- `slippage_bps`: 0.0

---

## Quick “how to run”

Generate the report for this exact best variant (OHLC-based):

```bash
.venv/bin/python scripts/report_trend_variants.py \
  --symbol XAUUSD \
  --use-ohlc-base \
  --use-htf-ohlc \
  --ema-sep-filter --sep-k 0.30 \
  --nochop-filter --nochop-min-closes 12 \
  --out-name v06_best_k030_M12_OHLCbased.html
```

---

## Notes / caveats

- HTF ATR in Option A is now computed as **true-range ATR** from the 15m OHLC bars (`data/dukascopy_15m_ohlc`).
- Strategy is currently long-only; short-side rules can be added symmetrically.

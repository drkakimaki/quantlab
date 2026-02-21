# Decision: icm_costs_0p07spread_7rt_commission

Timestamp: 2026-02-21T21:22:05

Objective: maximize sum(period PnL%) subject to worst MaxDD <= 999.0%

## Best score

- OK under cap: True
- Sum PnL%: 498.86
- Worst MaxDD%: -17.40
- Avg Sharpe: 4.13

## Assumptions / mapping (IC Markets rough)

- Instrument: **XAUUSD**, contract size assumed **100 oz / 1.00 lot**.
- Broker info (user-provided):
  - Avg spread (last 3 months): **$0.07**
  - Commission roundtrip: **$7 / lot**
- Spread mapping:
  - $0.07/oz × 100 oz/lot ≈ **$7 / lot roundtrip**
- Engine cost model (per side, per lot):
  - `cost = abs(d_lots) * (fee_per_lot + spread_per_lot)` on every position change
  - Therefore 1 lot roundtrip costs: `2 * (fee_per_lot + spread_per_lot)`
- We set:
  - `fee_per_lot = 3.50` (=$7 RT commission / 2)
  - `spread_per_lot = 3.50` (=$7 RT spread / 2)
  - Total ≈ **$14 / lot roundtrip**

## Comparison vs zero-cost baseline

For context, the previous canonical baseline used zero costs.
We include a compact before/after table:

- `compare_zero_vs_icm_costs.csv`

Headline effect (TOTAL):
- Sum PnL%: **534.72 → 498.86**
- Worst MaxDD%: **-16.21 → -17.40**
- Avg Sharpe: **4.29 → 4.13**

## Files

- best.yaml
- results.csv
- compare_zero_vs_icm_costs.csv
- notes.json

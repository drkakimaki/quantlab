# Decision: churn_gate_debounce_cooldown_v1

Timestamp: 2026-02-21T14:56:27

Objective: maximize sum(period PnL%) subject to worst MaxDD <= 20.0%

## Best score

- OK under cap: True
- Sum PnL%: 534.72
- Worst MaxDD%: -16.21
- Avg Sharpe: 4.29

## Files

- best.yaml
- results.csv
- top_k.csv (if sweep)
- notes.json

## Detailed comparison: no-churn vs churn (yearly trade breakdown)

To understand *why* the churn gate helps, we compared the yearly trade breakdown HTML for:

- **No churn** (same config, `churn:` removed):
  - `reports/trend_based/best_trend_trades_y_no_churn.html`
- **Churn enabled** (canonical promoted config):
  - `reports/trend_based/best_trend_trades_y.html`

### Total (all years aggregated)

| Variant | #Trades | Win rate | Profit factor | Sum PnL (net) | Avg bars/trade |
|---|---:|---:|---:|---:|---:|
| No churn | 1,449 | 41.48% | 1.66 | 5,067.46 | 64.06 |
| Churn (3,8) | 1,359 | 40.18% | 1.76 | 5,249.18 | 64.07 |

**Delta (churn − no churn):**
- Trades: **-90**
- Win rate: **-1.30 pp**
- Profit factor: **+0.10**
- Sum PnL (net): **+181.72**
- Avg bars/trade: ~flat

### By year

| Year | Variant | Trades | Win rate | Avg return | Sum PnL | Profit factor |
|---:|---|---:|---:|---:|---:|---:|
| 2023 | No churn | 226 | 39.82% | 0.21% | 568.40 | 1.64 |
| 2023 | Churn (3,8) | 210 | 40.00% | 0.21% | 533.40 | 1.62 |
| 2024 | No churn | 249 | 40.56% | 0.15% | 422.61 | 1.28 |
| 2024 | Churn (3,8) | 237 | 40.08% | 0.19% | 549.88 | 1.41 |
| 2025 | No churn | 261 | 49.43% | 0.47% | 2,429.35 | 2.32 |
| 2025 | Churn (3,8) | 240 | 45.83% | 0.50% | 2,335.32 | 2.35 |
| 2026 | No churn | 45 | 51.11% | 2.20% | 1,322.42 | 3.28 |
| 2026 | Churn (3,8) | 39 | 53.85% | 2.62% | 1,406.96 | 5.01 |

**Net for 2023–2025 (sum PnL):** essentially flat (churn shifts PnL between years).

### By holding duration (bars)

Key observation: edge remains **tail-driven** (long holds pay), but the churn gate reduces losses in the toxic mid-duration zone.

| Duration bin | Variant | Trades | Win rate | Avg return | Sum PnL | Profit factor |
|---|---|---:|---:|---:|---:|---:|
| 193+ | No churn | 53 | 98.11% | 5.22% | 3,795.91 | 15,622.03 |
| 193+ | Churn (3,8) | 49 | 100.00% | 5.38% | 3,694.81 | inf |
| 97–192 | No churn | 239 | 92.05% | 1.64% | 5,603.77 | 74.70 |
| 97–192 | Churn (3,8) | 220 | 90.45% | 1.70% | 5,331.78 | 66.82 |
| 49–96 | No churn | 452 | 45.80% | 0.05% | 281.22 | 1.15 |
| 49–96 | Churn (3,8) | 428 | 45.09% | 0.06% | 265.85 | 1.15 |
| 25–48 | No churn | 341 | 8.50% | -0.58% | -2,698.29 | 0.20 |
| 25–48 | Churn (3,8) | 326 | 9.82% | -0.51% | -2,283.57 | 0.22 |
| 13–24 | No churn | 168 | 20.83% | -0.62% | -1,202.50 | 0.16 |
| 13–24 | Churn (3,8) | 167 | 14.97% | -0.66% | -1,338.81 | 0.10 |

**Toxic mid-duration zone (13–48 bars):**
- No churn: **-3,900.79**
- Churn (3,8): **-3,622.38**
- Improvement: **+278.41**

### By calendar month (aggregated)

June remains a red flag in both variants.

| Month | Variant | Trades | Win rate | Avg return | Sum PnL | Profit factor |
|---|---|---:|---:|---:|---:|---:|
| Jun | No churn | 90 | 27.78% | -0.13% | -160.12 | 0.68 |
| Jun | Churn (3,8) | 87 | 25.29% | -0.14% | -175.31 | 0.65 |

(Other months shift modestly; June is consistently negative.)

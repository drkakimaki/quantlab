# Drawdown attribution — best_trend (XAUUSD)

Date: 2026-02-14

Artifacts:
- `drawdown_episodes.csv` — peak→trough→recovery episodes from equity curve
- `drawdown_contrib_daily.csv` — daily drawdown deepen/recovery attribution + regime fractions

## Top drawdown episodes — 2021-2022
- peak 2022-03-08 17:00:00+00:00 → trough 2022-10-03 14:45:00+00:00 → rec UNRECOVERED | maxDD -15.49% (-208.28 USD)
- peak 2021-05-07 12:55:00+00:00 → trough 2021-07-28 22:40:00+00:00 → rec 2022-02-11 20:25:00+00:00 | maxDD -11.21% (-128.48 USD)
- peak 2022-02-24 11:00:00+00:00 → trough 2022-03-01 09:05:00+00:00 → rec 2022-03-08 15:10:00+00:00 | maxDD -7.18% (-93.44 USD)
- peak 2021-03-15 01:30:00+00:00 → trough 2021-03-25 07:15:00+00:00 → rec 2021-04-15 14:10:00+00:00 | maxDD -7.06% (-73.99 USD)
- peak 2021-01-20 09:20:00+00:00 → trough 2021-02-01 15:55:00+00:00 → rec 2021-03-11 02:10:00+00:00 | maxDD -6.14% (-62.60 USD)

## Top drawdown episodes — 2023-2025
- peak 2023-12-03 23:35:00+00:00 → trough 2024-02-16 13:40:00+00:00 → rec 2024-04-05 14:45:00+00:00 | maxDD -16.98% (-264.99 USD)
- peak 2024-04-12 15:05:00+00:00 → trough 2024-05-09 13:20:00+00:00 → rec 2024-05-20 05:35:00+00:00 | maxDD -16.67% (-276.94 USD)
- peak 2023-05-03 22:05:00+00:00 → trough 2023-07-17 13:10:00+00:00 → rec 2023-10-13 17:55:00+00:00 | maxDD -16.64% (-211.09 USD)
- peak 2024-07-17 02:10:00+00:00 → trough 2024-09-10 14:30:00+00:00 → rec 2024-10-30 01:55:00+00:00 | maxDD -12.87% (-225.02 USD)
- peak 2023-01-09 07:20:00+00:00 → trough 2023-02-28 21:00:00+00:00 → rec 2023-03-12 22:05:00+00:00 | maxDD -9.26% (-98.20 USD)

## Top drawdown episodes — 2026
- peak 2026-01-28 23:40:00+00:00 → trough 2026-01-29 01:10:00+00:00 → rec 2026-02-04 01:15:00+00:00 | maxDD -12.05% (-283.08 USD)
- peak 2026-01-26 09:35:00+00:00 → trough 2026-01-27 11:55:00+00:00 → rec 2026-01-27 20:55:00+00:00 | maxDD -11.75% (-186.71 USD)
- peak 2026-01-12 16:15:00+00:00 → trough 2026-01-14 14:50:00+00:00 → rec 2026-01-20 23:50:00+00:00 | maxDD -11.71% (-141.41 USD)
- peak 2026-01-28 08:35:00+00:00 → trough 2026-01-28 15:10:00+00:00 → rec 2026-01-28 20:30:00+00:00 | maxDD -10.91% (-202.79 USD)
- peak 2026-01-05 10:55:00+00:00 → trough 2026-01-06 09:20:00+00:00 → rec 2026-01-11 23:45:00+00:00 | maxDD -10.21% (-109.35 USD)

## Biggest drawdown-deepening weeks (by USD)
- 2023-2025 2025-10-14/2025-10-20: deepen 2312.44 USD
- 2026 2026-01-27/2026-02-02: deepen 2219.47 USD
- 2026 2026-01-20/2026-01-26: deepen 1811.81 USD
- 2026 2026-02-03/2026-02-09: deepen 1727.81 USD
- 2023-2025 2025-10-07/2025-10-13: deepen 1256.65 USD
- 2023-2025 2025-09-30/2025-10-06: deepen 1233.60 USD
- 2026 2026-01-06/2026-01-12: deepen 1217.26 USD
- 2023-2025 2025-04-15/2025-04-21: deepen 1036.50 USD
- 2023-2025 2025-12-23/2025-12-29: deepen 935.41 USD
- 2023-2025 2025-04-08/2025-04-14: deepen 935.02 USD

## Notes / interpretation
- `deepen_usd_*` uses **changes in (equity - running_peak_equity)** per 5m bar; positive values mean the drawdown got worse on that bar/day.
- Regime columns (`*_bad`) attribute drawdown deepening that occurred while the corresponding filter condition was **false** (still possible because position can remain held, or because these are raw regime states independent of entry/segment logic).
- `in_position` uses executed position from the account/margin backtest (after time filter + lag).

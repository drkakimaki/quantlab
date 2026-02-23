# Decision: move_june_flat_to_time_filter_v1

Date: 2026-02-24


## Change

- Implement June force-flat as part of `time_filter` (allow_mask overlay) instead of a separate pipeline gate.

- Config change: `time_filter.months.block: [6]`

- Removed `seasonality_cap` gate (June size cap) since June is now fully flat.


## Summary (train objective only; 2026 excluded from score)

| variant                  | ok   |   sum_pnl |   worst_maxdd |   avg_sharpe |
|:-------------------------|:-----|----------:|--------------:|-------------:|
| canon_current            | True |   405.22  |      -17.2024 |      1.40127 |
| june_flat_in_time_filter | True |   417.704 |      -16.3276 |      1.4738  |


## Period breakdowns
Canonical:

| period    |   pnl_percent |   maxdd_percent |   sharpe | scored   |
|:----------|--------------:|----------------:|---------:|:---------|
| 2020-2022 |       45.3977 |        -17.2024 | 0.624113 | True     |
| 2023-2025 |      359.822  |        -12.8521 | 2.17843  | True     |
| 2026      |      130.961  |        -16.9173 | 4.77135  | False    |
| TOTAL     |      405.22   |        -17.2024 | 1.40127  | True     |


Candidate:

| period    |   pnl_percent |   maxdd_percent |   sharpe | scored   |
|:----------|--------------:|----------------:|---------:|:---------|
| 2020-2022 |       51.5049 |        -16.3276 |  0.69908 | True     |
| 2023-2025 |      366.199  |        -11.981  |  2.24852 | True     |
| 2026      |      130.961  |        -16.9173 |  4.77135 | False    |
| TOTAL     |      417.704  |        -16.3276 |  1.4738  | True     |

# Decision: promote_2020_recovery_v1

Date: 2026-02-23


## Context

- We observed that the sep_k=0.07 + min_closes=22 canonical improved later years but underperformed 2020–2022.

- Ran a targeted 60-candidate sweep to recover 2020–2022 while keeping 2026 holdout non-catastrophic.

- Source sweep bundle: `reports/trend_based/decisions/2026-02-23_sweep_2020_recovery_v1/`


## Decision

- Promote the 2020–2022 recovery configuration to canonical.

- Holdout policy: 2026 is a soft veto unless catastrophic; we accept higher 2026 DD as long as it remains within tolerance.


## Summary (train objective only; 2026 excluded from score)

| variant                     | ok   |   sum_pnl |   worst_maxdd |   avg_sharpe |
|:----------------------------|:-----|----------:|--------------:|-------------:|
| prev_canon_ref_sep0.07_nc22 | True |   405.051 |      -18.9334 |      1.32859 |
| new_canon_sep0.065_nc20     | True |   405.22  |      -17.2024 |      1.40127 |


## Period breakdowns

Reference (prev-canon-like):

| period    |   pnl_percent |   maxdd_percent |   sharpe | scored   |
|:----------|--------------:|----------------:|---------:|:---------|
| 2020-2022 |       24.9224 |        -18.9334 | 0.402698 | True     |
| 2023-2025 |      380.129  |        -13.3529 | 2.25449  | True     |
| 2026      |      148.548  |        -10.6581 | 5.72136  | False    |
| TOTAL     |      405.051  |        -18.9334 | 1.32859  | True     |


New canonical:

| period    |   pnl_percent |   maxdd_percent |   sharpe | scored   |
|:----------|--------------:|----------------:|---------:|:---------|
| 2020-2022 |       45.3977 |        -17.2024 | 0.624113 | True     |
| 2023-2025 |      359.822  |        -12.8521 | 2.17843  | True     |
| 2026      |      130.961  |        -16.9173 | 4.77135  | False    |
| TOTAL     |      405.22   |        -17.2024 | 1.40127  | True     |


## Rationale

- New canonical materially improves 2020–2022 PnL/Sharpe while keeping train MaxDD under the 20% cap.

- 2026 holdout drawdown worsens vs reference but is not considered catastrophic under the stated policy.


# Decision: force_flat_june_v1

Timestamp: 2026-02-24T00:21:27


## Change

- Replaced June sizing cap with a force-flat month filter: added `month_flat(months=[6])` and removed `seasonality_cap`.


## Summary (train objective only; 2026 excluded from score)

| variant         | ok   |   sum_pnl |   worst_maxdd |   avg_sharpe |
|:----------------|:-----|----------:|--------------:|-------------:|
| canon_current   | True |   405.22  |      -17.2024 |      1.40127 |
| force_flat_june | True |   417.704 |      -16.3276 |      1.4738  |


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


## Month breakdown (TRAIN)

Canonical:

|   entry_month |   n_trades |   sum_pnl |
|--------------:|-----------:|----------:|
|             1 |        136 |    -3.394 |
|             2 |        121 |   181.398 |
|             3 |        145 |   645.024 |
|             4 |        125 |   987.682 |
|             5 |        115 |   429.021 |
|             6 |        100 |  -120.62  |
|             7 |        153 |   112.683 |
|             8 |        119 |   125.544 |
|             9 |        117 |   426.578 |
|            10 |        134 |   810.525 |
|            11 |        118 |   557.991 |
|            12 |        150 |  -353.168 |


Candidate:

|   entry_month |   n_trades |   sum_pnl |
|--------------:|-----------:|----------:|
|             1 |        136 |    -3.394 |
|             2 |        121 |   181.398 |
|             3 |        145 |   645.024 |
|             4 |        125 |   987.682 |
|             5 |        115 |   427.568 |
|             7 |        155 |   111.636 |
|             8 |        119 |   125.544 |
|             9 |        117 |   426.578 |
|            10 |        134 |   810.525 |
|            11 |        118 |   557.991 |
|            12 |        150 |  -414.24  |

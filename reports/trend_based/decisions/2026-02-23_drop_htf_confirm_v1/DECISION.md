# Decision: drop_htf_confirm_v1

Date: 2026-02-23


## Change

- Candidate removes `htf_confirm` from the best_trend gate pipeline.


## Objective / scoring

- Objective: maximize sum(PnL%) over scored periods subject to worst MaxDD <= 20%.

- Holdout excluded from objective: ['2026']


## Summary (TRAIN objective only; 2026 excluded)

| variant          | ok   |   sum_pnl |   worst_maxdd |   avg_sharpe |
|:-----------------|:-----|----------:|--------------:|-------------:|
| canon_best       | True |   379.325 |      -17.3863 |      1.35179 |
| drop_htf_confirm | True |   388.852 |      -19.3916 |      1.34943 |


## Period breakdowns

Canonical:

| period    |   pnl_percent |   maxdd_percent |   sharpe | scored   |
|:----------|--------------:|----------------:|---------:|:---------|
| 2020-2022 |       32.4678 |       -17.3863  | 0.547729 | True     |
| 2023-2025 |      346.857  |       -14.4862  | 2.15585  | True     |
| 2026      |      161.391  |        -9.56503 | 6.39847  | False    |
| TOTAL     |      379.325  |       -17.3863  | 1.35179  | True     |


Candidate:

| period    |   pnl_percent |   maxdd_percent |   sharpe | scored   |
|:----------|--------------:|----------------:|---------:|:---------|
| 2020-2022 |        29.983 |        -19.3916 | 0.498759 | True     |
| 2023-2025 |       358.869 |        -14.3173 | 2.20011  | True     |
| 2026      |       154.903 |        -10.9074 | 5.89249  | False    |
| TOTAL     |       388.852 |        -19.3916 | 1.34943  | True     |


## Duration bucket deltas

TRAIN (candidate - canon):

| duration_bin   |   n_trades |   sum_pnl |
|:---------------|-----------:|----------:|
| 1              |         -2 |   -8.688  |
| 2-3            |         -5 |    2.667  |
| 4-6            |         -1 |   13.382  |
| 7-12           |         -9 |  -13.272  |
| 13-24          |          9 | -126.058  |
| 25-48          |         14 |  -28.228  |
| 49-96          |         15 |    5.53   |
| 97-192         |          9 |  -31.1545 |
| 193+           |          3 |  309.445  |


HOLDOUT (candidate - canon):

| duration_bin   |   n_trades |       sum_pnl |
|:---------------|-----------:|--------------:|
| 1              |          0 |   0           |
| 2-3            |          0 |   0           |
| 4-6            |          0 |   0           |
| 7-12           |          0 |   0           |
| 13-24          |          1 | -42.43        |
| 25-48          |          0 |   1.13687e-13 |
| 49-96          |          1 | -22.163       |
| 97-192         |          0 |  -2.27374e-13 |
| 193+           |          0 |  -2.27374e-13 |


## Files

- canon.yaml

- candidate.yaml

- summary.csv

- canon_periods.csv

- candidate_periods.csv

- canon_duration_train.csv

- candidate_duration_train.csv

- duration_delta_train_candidate_minus_canon.csv

- canon_duration_holdout.csv

- candidate_duration_holdout.csv

- duration_delta_holdout_candidate_minus_canon.csv


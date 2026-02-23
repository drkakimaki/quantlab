# Decision: archive_pre_drop_htf_confirm_v1

Date: 2026-02-23


## What this is

- An archival snapshot of the *canonical* best_trend configuration from before the `htf_confirm` gate was removed.

- Purpose: keep a strong historical reference point to guard against overfitting and to enable apples-to-apples comparisons.

- Source commit: `edbfd45`


## Metrics (from quantlab.rnd run)

|   maxdd_percent | period    |   pnl_percent | scored   |   sharpe |
|----------------:|:----------|--------------:|:---------|---------:|
|       -17.3863  | 2020-2022 |       32.4678 | True     | 0.547729 |
|       -14.4862  | 2023-2025 |      346.857  | True     | 2.15585  |
|        -9.56503 | 2026      |      161.391  | False    | 6.39847  |
|       -17.3863  | TOTAL     |      379.325  | True     | 1.35179  |


## Files

- best.yaml (archived config)

- results.csv (per-period metrics incl TOTAL)

- notes.json


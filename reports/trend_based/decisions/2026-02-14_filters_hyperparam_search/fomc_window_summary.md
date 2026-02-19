# FOMC window hyperparam search (best_trend)

Generated: 2026-02-14T19:09:16.814908Z

Configs evaluated: 50 (full grid)

Semantics: time_allow_mask passed into strategy (pre-engine), time_entry_shift=1; no post-filtering.

FOMC days source: `data/econ_calendar/fomc_decision_days.csv`


## Top by mean Sharpe (across all 3 periods)

| config_id    |   mean_sharpe |   worst_max_drawdown |   sum_pnl_usd |   sharpe_2023_2025 |   sharpe_2026 |   pre_h |   post_h | mode       | time_filter_mode         |
|:-------------|--------------:|---------------------:|--------------:|-------------------:|--------------:|--------:|---------:|:-----------|:-------------------------|
| 1452ff13f662 |       3.97915 |            -0.169846 |       5085.03 |            2.62116 |       8.84876 |     2   |      0.5 | force-flat | force_flat               |
| c7487c0848d5 |       3.94741 |            -0.170103 |       5065.37 |            2.61954 |       8.75514 |     1.5 |      0.5 | force-flat | force_flat               |
| 16ef9053365f |       3.94162 |            -0.173083 |       5047.18 |            2.58847 |       8.74993 |     1   |      0.5 | force-flat | force_flat               |
| 95f4e43eee6e |       3.92715 |            -0.199711 |       4970.4  |            2.48963 |       8.73717 |     0.5 |      0.5 | no-entry   | block_entry_hold_segment |
| 4363bd6d402f |       3.91675 |            -0.170817 |       5028.89 |            2.60297 |       8.67975 |     3   |      0.5 | force-flat | force_flat               |
| 08b3bd38d248 |       3.91518 |            -0.188611 |       5024.46 |            2.55292 |       8.73717 |     1.5 |      0.5 | no-entry   | block_entry_hold_segment |
| 1a6989b19bd7 |       3.91518 |            -0.188611 |       5024.46 |            2.55292 |       8.73717 |     2   |      0.5 | no-entry   | block_entry_hold_segment |
| 89cb3b751857 |       3.91362 |            -0.200486 |       4919.94 |            2.44902 |       8.73717 |     0.5 |      1   | no-entry   | block_entry_hold_segment |
| e8df694b19ba |       3.91362 |            -0.200486 |       4919.94 |            2.44902 |       8.73717 |     0.5 |      1.5 | no-entry   | block_entry_hold_segment |
| ae22aeb8c588 |       3.90803 |            -0.200486 |       4904.82 |            2.43228 |       8.73717 |     0.5 |      2   | no-entry   | block_entry_hold_segment |
| 62f070995395 |       3.90803 |            -0.200486 |       4904.82 |            2.43228 |       8.73717 |     0.5 |      3   | no-entry   | block_entry_hold_segment |
| 7e9d36d86553 |       3.90169 |            -0.189343 |       4974    |            2.51243 |       8.73717 |     1.5 |      1   | no-entry   | block_entry_hold_segment |
| a38d670dbdb5 |       3.90169 |            -0.189343 |       4974    |            2.51243 |       8.73717 |     1.5 |      1.5 | no-entry   | block_entry_hold_segment |
| 026ec0a4d5ca |       3.90169 |            -0.189343 |       4974    |            2.51243 |       8.73717 |     2   |      1   | no-entry   | block_entry_hold_segment |
| e7c2a06af4e9 |       3.90169 |            -0.189343 |       4974    |            2.51243 |       8.73717 |     2   |      1.5 | no-entry   | block_entry_hold_segment |

## Top by worst MaxDD (least-negative drawdown across periods)

| config_id    |   mean_sharpe |   worst_max_drawdown |   sum_pnl_usd |   sharpe_2023_2025 |   sharpe_2026 |   pre_h |   post_h | mode       | time_filter_mode         |
|:-------------|--------------:|---------------------:|--------------:|-------------------:|--------------:|--------:|---------:|:-----------|:-------------------------|
| 58aeb09254a2 |       3.74353 |            -0.169352 |       4928.78 |            2.62198 |       8.14746 |     2   |      1.5 | force-flat | force_flat               |
| cf600971e95a |       3.71102 |            -0.169609 |       4909.11 |            2.62035 |       8.05153 |     1.5 |      1.5 | force-flat | force_flat               |
| 1452ff13f662 |       3.97915 |            -0.169846 |       5085.03 |            2.62116 |       8.84876 |     2   |      0.5 | force-flat | force_flat               |
| c7487c0848d5 |       3.94741 |            -0.170103 |       5065.37 |            2.61954 |       8.75514 |     1.5 |      0.5 | force-flat | force_flat               |
| 47338ad034b7 |       3.67811 |            -0.170318 |       4872.64 |            2.60373 |       7.96943 |     3   |      1.5 | force-flat | force_flat               |
| 4363bd6d402f |       3.91675 |            -0.170817 |       5028.89 |            2.60297 |       8.67975 |     3   |      0.5 | force-flat | force_flat               |
| d4e0e4bc45cd |       3.70578 |            -0.172577 |       4890.93 |            2.58918 |       8.04802 |     1   |      1.5 | force-flat | force_flat               |
| 16ef9053365f |       3.94162 |            -0.173083 |       5047.18 |            2.58847 |       8.74993 |     1   |      0.5 | force-flat | force_flat               |
| a8a9e34a0b1e |       3.87154 |            -0.17334  |       5002.37 |            2.59659 |       8.55595 |     2   |      1   | force-flat | force_flat               |
| 20ca9e96fa11 |       3.83948 |            -0.17334  |       4982.71 |            2.59498 |       8.46139 |     1.5 |      1   | force-flat | force_flat               |
| e17033de9a37 |       3.80788 |            -0.17334  |       4946.23 |            2.57841 |       8.38316 |     3   |      1   | force-flat | force_flat               |
| 6c41c824b886 |       3.83395 |            -0.174623 |       4964.52 |            2.5639  |       8.45691 |     1   |      1   | force-flat | force_flat               |
| 8443bad70e62 |       3.89178 |            -0.176533 |       4982.83 |            2.53665 |       8.64889 |     0.5 |      0.5 | force-flat | force_flat               |
| d237dbf5f0fd |       3.65683 |            -0.185749 |       4826.57 |            2.53734 |       7.94966 |     0.5 |      1.5 | force-flat | force_flat               |
| 08b3bd38d248 |       3.91518 |            -0.188611 |       5024.46 |            2.55292 |       8.73717 |     1.5 |      0.5 | no-entry   | block_entry_hold_segment |

## Top by Sharpe (2023-2025)

| config_id    |   mean_sharpe |   worst_max_drawdown |   sum_pnl_usd |   sharpe_2023_2025 |   sharpe_2026 |   pre_h |   post_h | mode       | time_filter_mode   |
|:-------------|--------------:|---------------------:|--------------:|-------------------:|--------------:|--------:|---------:|:-----------|:-------------------|
| 58aeb09254a2 |       3.74353 |            -0.169352 |       4928.78 |            2.62198 |       8.14746 |     2   |      1.5 | force-flat | force_flat         |
| 1452ff13f662 |       3.97915 |            -0.169846 |       5085.03 |            2.62116 |       8.84876 |     2   |      0.5 | force-flat | force_flat         |
| cf600971e95a |       3.71102 |            -0.169609 |       4909.11 |            2.62035 |       8.05153 |     1.5 |      1.5 | force-flat | force_flat         |
| c7487c0848d5 |       3.94741 |            -0.170103 |       5065.37 |            2.61954 |       8.75514 |     1.5 |      0.5 | force-flat | force_flat         |
| 47338ad034b7 |       3.67811 |            -0.170318 |       4872.64 |            2.60373 |       7.96943 |     3   |      1.5 | force-flat | force_flat         |
| 4363bd6d402f |       3.91675 |            -0.170817 |       5028.89 |            2.60297 |       8.67975 |     3   |      0.5 | force-flat | force_flat         |
| a8a9e34a0b1e |       3.87154 |            -0.17334  |       5002.37 |            2.59659 |       8.55595 |     2   |      1   | force-flat | force_flat         |
| 20ca9e96fa11 |       3.83948 |            -0.17334  |       4982.71 |            2.59498 |       8.46139 |     1.5 |      1   | force-flat | force_flat         |
| d4e0e4bc45cd |       3.70578 |            -0.172577 |       4890.93 |            2.58918 |       8.04802 |     1   |      1.5 | force-flat | force_flat         |
| 16ef9053365f |       3.94162 |            -0.173083 |       5047.18 |            2.58847 |       8.74993 |     1   |      0.5 | force-flat | force_flat         |
| e17033de9a37 |       3.80788 |            -0.17334  |       4946.23 |            2.57841 |       8.38316 |     3   |      1   | force-flat | force_flat         |
| fb83d007f104 |       3.68457 |            -0.197904 |       4856.11 |            2.57834 |       8.01163 |     2   |      2   | force-flat | force_flat         |
| e2e41c584466 |       3.65188 |            -0.197904 |       4836.45 |            2.57672 |       7.91517 |     1.5 |      2   | force-flat | force_flat         |
| 2a95ee639378 |       3.4902  |            -0.197904 |       4732.11 |            2.57421 |       7.43263 |     2   |      3   | force-flat | force_flat         |
| 0f7c53a1a577 |       3.45686 |            -0.197904 |       4712.45 |            2.57258 |       7.33424 |     1.5 |      3   | force-flat | force_flat         |

## Top by Sharpe (2026)

| config_id    |   mean_sharpe |   worst_max_drawdown |   sum_pnl_usd |   sharpe_2023_2025 |   sharpe_2026 |   pre_h |   post_h | mode       | time_filter_mode         |
|:-------------|--------------:|---------------------:|--------------:|-------------------:|--------------:|--------:|---------:|:-----------|:-------------------------|
| 1452ff13f662 |       3.97915 |            -0.169846 |       5085.03 |            2.62116 |       8.84876 |     2   |      0.5 | force-flat | force_flat               |
| c7487c0848d5 |       3.94741 |            -0.170103 |       5065.37 |            2.61954 |       8.75514 |     1.5 |      0.5 | force-flat | force_flat               |
| 16ef9053365f |       3.94162 |            -0.173083 |       5047.18 |            2.58847 |       8.74993 |     1   |      0.5 | force-flat | force_flat               |
| 08b3bd38d248 |       3.91518 |            -0.188611 |       5024.46 |            2.55292 |       8.73717 |     1.5 |      0.5 | no-entry   | block_entry_hold_segment |
| 1a6989b19bd7 |       3.91518 |            -0.188611 |       5024.46 |            2.55292 |       8.73717 |     2   |      0.5 | no-entry   | block_entry_hold_segment |
| 7e9d36d86553 |       3.90169 |            -0.189343 |       4974    |            2.51243 |       8.73717 |     1.5 |      1   | no-entry   | block_entry_hold_segment |
| a38d670dbdb5 |       3.90169 |            -0.189343 |       4974    |            2.51243 |       8.73717 |     1.5 |      1.5 | no-entry   | block_entry_hold_segment |
| 026ec0a4d5ca |       3.90169 |            -0.189343 |       4974    |            2.51243 |       8.73717 |     2   |      1   | no-entry   | block_entry_hold_segment |
| e7c2a06af4e9 |       3.90169 |            -0.189343 |       4974    |            2.51243 |       8.73717 |     2   |      1.5 | no-entry   | block_entry_hold_segment |
| 2af7e4c9b813 |       3.89606 |            -0.193513 |       4958.88 |            2.49556 |       8.73717 |     1.5 |      2   | no-entry   | block_entry_hold_segment |
| a7640aa827cf |       3.89606 |            -0.193513 |       4958.88 |            2.49556 |       8.73717 |     1.5 |      3   | no-entry   | block_entry_hold_segment |
| d1e6f83a122e |       3.89606 |            -0.193513 |       4958.88 |            2.49556 |       8.73717 |     2   |      2   | no-entry   | block_entry_hold_segment |
| 5f5eca781008 |       3.89606 |            -0.193513 |       4958.88 |            2.49556 |       8.73717 |     2   |      3   | no-entry   | block_entry_hold_segment |
| 95f4e43eee6e |       3.92715 |            -0.199711 |       4970.4  |            2.48963 |       8.73717 |     0.5 |      0.5 | no-entry   | block_entry_hold_segment |
| 0dca1e08e92f |       3.90153 |            -0.199711 |       4937.84 |            2.48963 |       8.73717 |     1   |      0.5 | no-entry   | block_entry_hold_segment |

## HTML reports generated (top-k by mean Sharpe)

- reports/trend_based/fomc_window_search/top_1452ff13f662.html
- reports/trend_based/fomc_window_search/top_c7487c0848d5.html
- reports/trend_based/fomc_window_search/top_16ef9053365f.html
- reports/trend_based/fomc_window_search/top_95f4e43eee6e.html
- reports/trend_based/fomc_window_search/top_4363bd6d402f.html

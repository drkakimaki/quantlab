# FOMC filter sweep — best_trend

FOMC days source: `data/econ_calendar/fomc_decision_days.csv` (date-only; no exact decision times).

Interpretation:

- `fomc_day_*`: blocks the whole UTC day (represented as 12:00Z ± 12h).

- `fomc_19_2h_*`: blocks around 19:00 UTC ± 2h (approx statement time; ignores DST shifts).


## Aggregate ranking (mean Sharpe across all 3 periods)

         variant_id                                                   variant_label  mean_sharpe  worst_maxdd  sum_pnl
               none                                      No FOMC filter (reference)     3.531989   -20.990034 463.4442
fomc_19_1h_no_entry       Block around 19:00 UTC ±1h — block entries, hold segments     3.529839   -20.990034 465.8658
fomc_19_2h_no_entry       Block around 19:00 UTC ±2h — block entries, hold segments     3.521553   -20.990034 464.9056
    fomc_19_1h_flat Block around 19:00 UTC ±1h — force flat (approx statement time)     3.452168   -20.990034 462.5154
    fomc_19_2h_flat Block around 19:00 UTC ±2h — force flat (approx statement time)     3.263265   -20.990034 447.7717
  fomc_day_no_entry  Block FOMC decision *day* (UTC) — block entries, hold segments     3.103652   -20.990034 415.4041
fomc_19_4h_no_entry       Block around 19:00 UTC ±4h — block entries, hold segments     3.070686   -22.104590 410.1730
    fomc_19_4h_flat Block around 19:00 UTC ±4h — force flat (approx statement time)     3.034959   -20.990034 429.0253
      fomc_day_flat                    Block FOMC decision *day* (UTC) — force flat     2.851380   -20.990034 410.5575


## 2021-2022 — by Sharpe

         variant_id   sharpe  pnl_pct  maxdd_pct  trades  n_events
    fomc_19_1h_flat 0.384058  12.0429 -18.057718     416        16
               none 0.381828  11.9453 -18.057718     416         0
fomc_19_1h_no_entry 0.371304  11.4553 -18.057718     415        16
    fomc_19_2h_flat 0.361213  11.0357 -18.202839     416        16
      fomc_day_flat 0.347039  10.3505 -17.232189     409        16
    fomc_19_4h_flat 0.343437  10.2209 -18.430798     416        16
fomc_19_2h_no_entry 0.341199  10.1318 -18.281234     414        16
  fomc_day_no_entry 0.247735   5.9301 -20.822440     404        16
fomc_19_4h_no_entry 0.182767   3.0192 -22.104590     412        16


## 2023-2025 — by Sharpe

         variant_id   sharpe   pnl_pct  maxdd_pct  trades  n_events
    fomc_19_1h_flat 2.670355 331.25295 -15.864307     746        21
fomc_19_2h_no_entry 2.651576 328.71255 -15.602896     740        21
fomc_19_1h_no_entry 2.646327 328.34925 -15.602896     741        21
               none 2.642253 325.43765 -15.602896     745         0
    fomc_19_2h_flat 2.636431 328.21275 -19.868676     745        21
  fomc_day_no_entry 2.627641 326.74845 -17.266048     723        21
    fomc_19_4h_flat 2.600137 322.24785 -18.925854     744        21
fomc_19_4h_no_entry 2.593710 324.42825 -17.266048     736        21
      fomc_day_flat 2.575469 320.90775 -18.635170     731        21


## 2026 — by Sharpe

         variant_id   sharpe   pnl_pct  maxdd_pct  trades  n_events
               none 7.571885 126.06125 -20.990034      43         0
fomc_19_2h_no_entry 7.571885 126.06125 -20.990034      43         1
fomc_19_1h_no_entry 7.571885 126.06125 -20.990034      43         1
    fomc_19_1h_flat 7.302089 119.21955 -20.990034      44         1
    fomc_19_2h_flat 6.792151 108.52325 -20.990034      44         1
  fomc_day_no_entry 6.435580  82.72555 -20.990034      42         1
fomc_19_4h_no_entry 6.435580  82.72555 -20.990034      42         1
    fomc_19_4h_flat 6.161302  96.55655 -20.990034      43         1
      fomc_day_flat 5.631633  79.29925 -20.990034      43         1

# Lookahead / Fault-Confidence Audit — best_trend

This audit stress-tests the strategy for common lookahead/alignment faults by comparing results under more/less conservative variants.


## HTF alignment sanity (15m -> 5m)

CSV: `reports/trend_based/audits/htf_alignment.csv`

   period  n_base  n_htf  n_missing_htf  n_future_violation
2021-2022  141818  47273              2                   0
2023-2025  212724  70915              2                   0
     2026    8249   2750              2                   0


## Performance deltas under audit variants

CSV: `reports/trend_based/audits/lookahead_metrics.csv`


### 2021-2022

           variant  pnl_pct  maxdd_pct   sharpe  n_trades
corr_shift0_DANGER  8.30765 -18.999491 0.301084       416
engine_lag0_DANGER 10.01620 -16.185235 0.338493       416
       engine_lag2  9.67305 -16.844071 0.333623       416
  htf_extra_shift1  5.73265 -20.869551 0.243417       414
         prod_like 11.94530 -18.057718 0.381828       416
size_shift0_DANGER  8.30765 -18.999491 0.301084       416

### 2023-2025

           variant   pnl_pct  maxdd_pct   sharpe  n_trades
corr_shift0_DANGER 325.26230 -15.616112 2.651752       744
engine_lag0_DANGER 323.92570 -15.777055 2.625067       745
       engine_lag2 346.67605 -13.636874 2.823883       745
  htf_extra_shift1 327.95390 -15.703380 2.682231       740
         prod_like 325.43765 -15.602896 2.642253       745
size_shift0_DANGER 324.51245 -15.616112 2.647008       745

### 2026

           variant   pnl_pct  maxdd_pct   sharpe  n_trades
corr_shift0_DANGER 124.35975 -20.990034 7.432494        43
engine_lag0_DANGER 124.70285 -16.541160 7.624567        43
       engine_lag2 122.33370 -20.132055 7.358560        43
  htf_extra_shift1 116.12935 -16.818558 7.376966        44
         prod_like 126.06125 -20.990034 7.571885        43
size_shift0_DANGER 124.35975 -20.990034 7.432494        43


## How to interpret
- `prod_like` should be the reference.
- If `engine_lag0_DANGER` or `corr_shift0_DANGER` performs dramatically better, that indicates the strategy *would* benefit from lookahead, and we should be confident we are *not* using it in production.
- If `htf_extra_shift1` changes results a lot, HTF alignment is a sensitive area; this is not necessarily bias, but it is a key porting risk (e.g. MQL).


# Time filter sweep — best_trend

Data/index assumed UTC (parquet timestamps). These filters are *heuristic* session windows.


## Top by mean Sharpe (across all 3 periods)

         variant_id                                                     variant_label  mean_sharpe  worst_maxdd   sum_pnl
               none                                        No time filter (reference)     3.531989   -20.990034 463.44420
    no_ny_open_flat                                Block 13:30-14:29 UTC (force flat)     3.518503   -21.415899 448.33610
no_ny_open_no_entry              Block 13:30-14:29 UTC (block entries; hold segments)     3.476002   -20.990034 446.12980
no_late_us_no_entry              Block 20:00-23:59 UTC (block entries; hold segments)     3.145255   -23.327156 382.47840
   no_asia_no_entry         Block Asia 00:00-06:59 UTC (block entries; hold segments)     2.971532   -25.699455 363.10810
       no_asia_flat                           Block Asia 00:00-06:59 UTC (force flat)     2.753573   -23.018699 317.89605
only_07_21_no_entry Allow only 07:00-20:59 UTC (block entries outside; hold segments)     2.615264   -31.891797 304.68230
    no_late_us_flat                                Block 20:00-23:59 UTC (force flat)     2.110595   -22.794741 308.22345
    only_07_21_flat                   Allow only 07:00-20:59 UTC (force flat outside)     1.443114   -21.643536 205.27575


## 2021-2022 — Top by Sharpe

         variant_id    sharpe  pnl_pct  maxdd_pct  trades
    only_07_21_flat  0.416341 13.13645 -21.643536     358
       no_asia_flat  0.393191 12.35705 -23.018699     400
               none  0.381828 11.94530 -18.057718     416
    no_late_us_flat  0.335054  9.70785 -16.448212     466
   no_asia_no_entry  0.261755  6.43765 -25.699455     310
no_ny_open_no_entry  0.232662  5.19000 -18.565274     393
    no_ny_open_flat  0.190249  3.45860 -16.873576     487
no_late_us_no_entry  0.079565 -1.34245 -23.327156     364
only_07_21_no_entry -0.020790 -4.99750 -31.891797     267


## 2023-2025 — Top by Sharpe

         variant_id   sharpe   pnl_pct  maxdd_pct  trades
no_ny_open_no_entry 2.651714 316.89355 -17.156882     704
               none 2.642253 325.43765 -15.602896     745
    no_ny_open_flat 2.614025 318.94040 -15.932138     868
   no_asia_no_entry 2.415664 265.01865 -15.314118     526
no_late_us_no_entry 2.339958 268.66730 -17.076084     654
       no_asia_flat 2.329674 236.05925 -12.580971     703
only_07_21_no_entry 2.166079 226.67170 -16.345959     463
    no_late_us_flat 2.122862 247.32385 -16.444971     827
    only_07_21_flat 1.901701 175.66030 -12.144353     640


## 2026 — Top by Sharpe

         variant_id   sharpe   pnl_pct  maxdd_pct  trades
    no_ny_open_flat 7.751234 125.93710 -21.415899      46
               none 7.571885 126.06125 -20.990034      43
no_ny_open_no_entry 7.543631 124.04625 -20.990034      41
no_late_us_no_entry 7.016241 115.15355 -18.331165      40
   no_asia_no_entry 6.237178  91.65180 -15.884228      33
only_07_21_no_entry 5.700502  83.00810 -17.386975      30
       no_asia_flat 5.537853  69.47975 -18.222897      45
    no_late_us_flat 3.873868  51.19175 -22.794741      51
    only_07_21_flat 2.011300  16.47900 -18.533616      43

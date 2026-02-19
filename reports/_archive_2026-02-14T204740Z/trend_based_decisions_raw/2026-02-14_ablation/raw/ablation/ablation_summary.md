# Best-trend filter ablation (ema_sep / nochop / base_slope / corr)

- Variants dir: `reports/trend_based/ablation/variants`
- Parsed long metrics CSV: `reports/trend_based/ablation/ablation_metrics_long.csv`
- Parsed aggregate CSV: `reports/trend_based/ablation/ablation_metrics_agg.csv`

## Aggregate ranking (across periods)
Sorted by mean Sharpe (desc), then total PnL% (desc).

| variant | ema_sep | nochop | base_slope | corr | disabled | pnl_total_pct | sharpe_mean | maxdd_worst_pct | trades_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_trend__no-base_slope__no-corr | 1 | 1 | 0 | 0 | base_slope,corr | 258.8 | 3.987 | -11.05 | 1205 |
| best_trend__no-corr | 1 | 1 | 1 | 0 | corr | 256.7 | 3.98 | -11.27 | 1205 |
| best_trend__no-ema_sep__no-base_slope__no-corr | 0 | 1 | 0 | 0 | base_slope,corr,ema_sep | 243.6 | 3.777 | -18.36 | 1687 |
| best_trend__no-ema_sep__no-corr | 0 | 1 | 1 | 0 | corr,ema_sep | 241.9 | 3.773 | -17.78 | 1686 |
| best_trend__no-ema_sep__no-nochop__no-corr | 0 | 0 | 1 | 0 | corr,ema_sep,nochop | 244 | 3.313 | -21.71 | 1869 |
| best_trend__no-base_slope | 1 | 1 | 0 | 1 | base_slope | 408.5 | 3.2 | -20.99 | 1196 |
| best_trend__all_on | 1 | 1 | 1 | 1 | (none) | 404.3 | 3.19 | -20.99 | 1196 |
| best_trend__no-ema_sep__no-nochop__no-base_slope__no-corr | 0 | 0 | 0 | 0 | base_slope,corr,ema_sep,nochop | 240.4 | 3.183 | -24.56 | 1921 |
| best_trend__no-nochop__no-corr | 1 | 0 | 1 | 0 | corr,nochop | 243.7 | 3.173 | -12.39 | 1341 |
| best_trend__no-ema_sep__no-base_slope | 0 | 1 | 0 | 1 | base_slope,ema_sep | 380.3 | 3.06 | -33.86 | 1677 |
| best_trend__no-ema_sep | 0 | 1 | 1 | 1 | ema_sep | 376.7 | 3.053 | -33.84 | 1676 |
| best_trend__no-nochop__no-base_slope__no-corr | 1 | 0 | 0 | 0 | base_slope,corr,nochop | 238.3 | 2.95 | -15.46 | 1373 |
| best_trend__no-ema_sep__no-nochop | 0 | 0 | 1 | 1 | ema_sep,nochop | 360.6 | 2.577 | -34.87 | 1859 |
| best_trend__no-ema_sep__no-nochop__no-base_slope | 0 | 0 | 0 | 1 | base_slope,ema_sep,nochop | 351.7 | 2.433 | -39.61 | 1910 |
| best_trend__no-nochop | 1 | 0 | 1 | 1 | nochop | 360.2 | 2.33 | -22.69 | 1332 |
| best_trend__no-nochop__no-base_slope | 1 | 0 | 0 | 1 | base_slope,nochop | 347.5 | 2.06 | -28.54 | 1363 |

## Period: 2021-2022

### Top Sharpe

| variant | disabled | PnL_pct | MaxDD_pct | Sharpe | WinRate | ProfitFactor | AvgWin | AvgLoss | Trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_trend__no-nochop | nochop | 10.46 | -15.27 | 0.35 | 36.71 | 1.07 | 1.14 | -0.62 | 473 |
| best_trend__no-nochop__no-base_slope__no-corr | base_slope,corr,nochop | 6.58 | -11.63 | 0.34 | 36.78 | 1.06 | 0.63 | -0.35 | 483 |
| best_trend__no-nochop__no-corr | corr,nochop | 6.56 | -10.18 | 0.33 | 36.71 | 1.06 | 0.64 | -0.35 | 473 |
| best_trend__no-nochop__no-base_slope | base_slope,nochop | 8.08 | -18.7 | 0.3 | 36.78 | 1.06 | 1.12 | -0.63 | 483 |
| best_trend__no-corr | corr | 2.66 | -11.27 | 0.18 | 37.02 | 1.03 | 0.63 | -0.37 | 415 |
| best_trend__no-base_slope__no-corr | base_slope,corr | 2.57 | -11.05 | 0.17 | 37.17 | 1.03 | 0.63 | -0.37 | 416 |
| best_trend__no-ema_sep__no-nochop | ema_sep,nochop | -0.2 | -34.22 | 0.14 | 35.11 | 1.01 | 1.33 | -0.73 | 694 |
| best_trend__no-base_slope | base_slope | -0.41 | -20.24 | 0.1 | 37.17 | 1.01 | 1.13 | -0.67 | 416 |

### Top PnL (%)

| variant | disabled | PnL_pct | MaxDD_pct | Sharpe | WinRate | ProfitFactor | AvgWin | AvgLoss | Trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_trend__no-nochop | nochop | 10.46 | -15.27 | 0.35 | 36.71 | 1.07 | 1.14 | -0.62 | 473 |
| best_trend__no-nochop__no-base_slope | base_slope,nochop | 8.08 | -18.7 | 0.3 | 36.78 | 1.06 | 1.12 | -0.63 | 483 |
| best_trend__no-nochop__no-base_slope__no-corr | base_slope,corr,nochop | 6.58 | -11.63 | 0.34 | 36.78 | 1.06 | 0.63 | -0.35 | 483 |
| best_trend__no-nochop__no-corr | corr,nochop | 6.56 | -10.18 | 0.33 | 36.71 | 1.06 | 0.64 | -0.35 | 473 |
| best_trend__no-corr | corr | 2.66 | -11.27 | 0.18 | 37.02 | 1.03 | 0.63 | -0.37 | 415 |
| best_trend__no-base_slope__no-corr | base_slope,corr | 2.57 | -11.05 | 0.17 | 37.17 | 1.03 | 0.63 | -0.37 | 416 |
| best_trend__no-ema_sep__no-nochop | ema_sep,nochop | -0.2 | -34.22 | 0.14 | 35.11 | 1.01 | 1.33 | -0.73 | 694 |
| best_trend__all_on | (none) | -0.24 | -20.66 | 0.1 | 37.02 | 1.02 | 1.14 | -0.67 | 415 |

### Least drawdown (MaxDD closest to 0)

| variant | disabled | PnL_pct | MaxDD_pct | Sharpe | WinRate | ProfitFactor | AvgWin | AvgLoss | Trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_trend__no-nochop__no-corr | corr,nochop | 6.56 | -10.18 | 0.33 | 36.71 | 1.06 | 0.64 | -0.35 | 473 |
| best_trend__no-base_slope__no-corr | base_slope,corr | 2.57 | -11.05 | 0.17 | 37.17 | 1.03 | 0.63 | -0.37 | 416 |
| best_trend__no-corr | corr | 2.66 | -11.27 | 0.18 | 37.02 | 1.03 | 0.63 | -0.37 | 415 |
| best_trend__no-nochop__no-base_slope__no-corr | base_slope,corr,nochop | 6.58 | -11.63 | 0.34 | 36.78 | 1.06 | 0.63 | -0.35 | 483 |
| best_trend__no-nochop | nochop | 10.46 | -15.27 | 0.35 | 36.71 | 1.07 | 1.14 | -0.62 | 473 |
| best_trend__no-ema_sep__no-corr | corr,ema_sep | -9.21 | -17.78 | -0.26 | 34.69 | 0.94 | 0.72 | -0.41 | 613 |
| best_trend__no-ema_sep__no-base_slope__no-corr | base_slope,corr,ema_sep | -9.37 | -18.36 | -0.27 | 34.8 | 0.94 | 0.71 | -0.41 | 614 |
| best_trend__no-nochop__no-base_slope | base_slope,nochop | 8.08 | -18.7 | 0.3 | 36.78 | 1.06 | 1.12 | -0.63 | 483 |

## Period: 2023-2025

### Top Sharpe

| variant | disabled | PnL_pct | MaxDD_pct | Sharpe | WinRate | ProfitFactor | AvgWin | AvgLoss | Trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_trend__no-base_slope__no-corr | base_slope,corr | 172 | -9.57 | 2.69 | 39.54 | 1.68 | 0.83 | -0.33 | 746 |
| best_trend__no-base_slope | base_slope | 298.3 | -14.29 | 2.66 | 39.84 | 1.71 | 1.15 | -0.45 | 738 |
| best_trend__no-corr | corr | 169.8 | -9.73 | 2.66 | 39.49 | 1.68 | 0.84 | -0.33 | 747 |
| best_trend__all_on | (none) | 293.9 | -14.57 | 2.63 | 39.78 | 1.71 | 1.15 | -0.45 | 739 |
| best_trend__no-nochop__no-base_slope__no-corr | base_slope,corr,nochop | 175.1 | -11.57 | 2.43 | 41.09 | 1.54 | 0.84 | -0.39 | 847 |
| best_trend__no-nochop__no-corr | corr,nochop | 173.7 | -11.63 | 2.41 | 40.61 | 1.54 | 0.88 | -0.39 | 825 |
| best_trend__no-nochop__no-base_slope | base_slope,nochop | 288.9 | -16.75 | 2.31 | 41.05 | 1.54 | 1.19 | -0.55 | 838 |
| best_trend__no-nochop | nochop | 285.3 | -16.88 | 2.3 | 40.64 | 1.54 | 1.22 | -0.55 | 817 |

### Top PnL (%)

| variant | disabled | PnL_pct | MaxDD_pct | Sharpe | WinRate | ProfitFactor | AvgWin | AvgLoss | Trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_trend__no-base_slope | base_slope | 298.3 | -14.29 | 2.66 | 39.84 | 1.71 | 1.15 | -0.45 | 738 |
| best_trend__all_on | (none) | 293.9 | -14.57 | 2.63 | 39.78 | 1.71 | 1.15 | -0.45 | 739 |
| best_trend__no-nochop__no-base_slope | base_slope,nochop | 288.9 | -16.75 | 2.31 | 41.05 | 1.54 | 1.19 | -0.55 | 838 |
| best_trend__no-nochop | nochop | 285.3 | -16.88 | 2.3 | 40.64 | 1.54 | 1.22 | -0.55 | 817 |
| best_trend__no-ema_sep__no-base_slope | base_slope,ema_sep | 256.8 | -26.36 | 1.99 | 37.82 | 1.42 | 1.22 | -0.53 | 1018 |
| best_trend__no-ema_sep | ema_sep | 253.2 | -26.63 | 1.97 | 37.72 | 1.41 | 1.23 | -0.54 | 1018 |
| best_trend__no-ema_sep__no-nochop | ema_sep,nochop | 246.7 | -34.87 | 1.66 | 38.12 | 1.3 | 1.41 | -0.68 | 1120 |
| best_trend__no-ema_sep__no-nochop__no-base_slope | base_slope,ema_sep,nochop | 246.6 | -37.28 | 1.64 | 38.48 | 1.31 | 1.37 | -0.67 | 1159 |

### Least drawdown (MaxDD closest to 0)

| variant | disabled | PnL_pct | MaxDD_pct | Sharpe | WinRate | ProfitFactor | AvgWin | AvgLoss | Trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_trend__no-base_slope__no-corr | base_slope,corr | 172 | -9.57 | 2.69 | 39.54 | 1.68 | 0.83 | -0.33 | 746 |
| best_trend__no-corr | corr | 169.8 | -9.73 | 2.66 | 39.49 | 1.68 | 0.84 | -0.33 | 747 |
| best_trend__no-nochop__no-base_slope__no-corr | base_slope,corr,nochop | 175.1 | -11.57 | 2.43 | 41.09 | 1.54 | 0.84 | -0.39 | 847 |
| best_trend__no-nochop__no-corr | corr,nochop | 173.7 | -11.63 | 2.41 | 40.61 | 1.54 | 0.88 | -0.39 | 825 |
| best_trend__no-base_slope | base_slope | 298.3 | -14.29 | 2.66 | 39.84 | 1.71 | 1.15 | -0.45 | 738 |
| best_trend__all_on | (none) | 293.9 | -14.57 | 2.63 | 39.78 | 1.71 | 1.15 | -0.45 | 739 |
| best_trend__no-ema_sep__no-base_slope__no-corr | base_slope,corr,ema_sep | 151.3 | -16.45 | 2.11 | 37.68 | 1.43 | 0.84 | -0.36 | 1027 |
| best_trend__no-ema_sep__no-corr | corr,ema_sep | 149.5 | -16.69 | 2.09 | 37.59 | 1.43 | 0.84 | -0.36 | 1027 |

## Period: 2026

### Top Sharpe

| variant | disabled | PnL_pct | MaxDD_pct | Sharpe | WinRate | ProfitFactor | AvgWin | AvgLoss | Trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_trend__no-ema_sep__no-base_slope__no-corr | base_slope,corr,ema_sep | 101.7 | -11.56 | 9.49 | 36.17 | 4.03 | 5.69 | -0.8 | 46 |
| best_trend__no-ema_sep__no-corr | corr,ema_sep | 101.7 | -11.56 | 9.49 | 36.17 | 4.03 | 5.69 | -0.8 | 46 |
| best_trend__no-corr | corr | 84.22 | -10.98 | 9.1 | 36.36 | 3.78 | 5.41 | -0.82 | 43 |
| best_trend__no-base_slope__no-corr | base_slope,corr | 84.22 | -10.98 | 9.1 | 36.36 | 3.78 | 5.41 | -0.82 | 43 |
| best_trend__no-ema_sep__no-nochop__no-corr | corr,ema_sep,nochop | 88.3 | -12.69 | 8 | 44.68 | 2.95 | 5.09 | -1.39 | 46 |
| best_trend__no-ema_sep__no-nochop__no-base_slope__no-corr | base_slope,corr,ema_sep,nochop | 85.96 | -15.64 | 7.66 | 44.68 | 2.75 | 5.31 | -1.56 | 46 |
| best_trend__no-ema_sep__no-base_slope | base_slope,ema_sep | 145.7 | -21.54 | 7.45 | 36.96 | 3.22 | 8.24 | -1.5 | 45 |
| best_trend__no-ema_sep | ema_sep | 145.7 | -21.54 | 7.45 | 36.96 | 3.22 | 8.24 | -1.5 | 45 |

### Top PnL (%)

| variant | disabled | PnL_pct | MaxDD_pct | Sharpe | WinRate | ProfitFactor | AvgWin | AvgLoss | Trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_trend__no-ema_sep__no-base_slope | base_slope,ema_sep | 145.7 | -21.54 | 7.45 | 36.96 | 3.22 | 8.24 | -1.5 | 45 |
| best_trend__no-ema_sep | ema_sep | 145.7 | -21.54 | 7.45 | 36.96 | 3.22 | 8.24 | -1.5 | 45 |
| best_trend__no-ema_sep__no-nochop | ema_sep,nochop | 114.1 | -21.65 | 5.93 | 45.65 | 2.41 | 7.62 | -2.66 | 45 |
| best_trend__all_on | (none) | 110.6 | -20.99 | 6.84 | 37.21 | 2.93 | 7.65 | -1.55 | 42 |
| best_trend__no-base_slope | base_slope | 110.6 | -20.99 | 6.84 | 37.21 | 2.93 | 7.65 | -1.55 | 42 |
| best_trend__no-ema_sep__no-nochop__no-base_slope | base_slope,ema_sep,nochop | 109.5 | -28.88 | 5.59 | 45.65 | 2.28 | 8.19 | -3.02 | 45 |
| best_trend__no-ema_sep__no-base_slope__no-corr | base_slope,corr,ema_sep | 101.7 | -11.56 | 9.49 | 36.17 | 4.03 | 5.69 | -0.8 | 46 |
| best_trend__no-ema_sep__no-corr | corr,ema_sep | 101.7 | -11.56 | 9.49 | 36.17 | 4.03 | 5.69 | -0.8 | 46 |

### Least drawdown (MaxDD closest to 0)

| variant | disabled | PnL_pct | MaxDD_pct | Sharpe | WinRate | ProfitFactor | AvgWin | AvgLoss | Trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_trend__no-corr | corr | 84.22 | -10.98 | 9.1 | 36.36 | 3.78 | 5.41 | -0.82 | 43 |
| best_trend__no-base_slope__no-corr | base_slope,corr | 84.22 | -10.98 | 9.1 | 36.36 | 3.78 | 5.41 | -0.82 | 43 |
| best_trend__no-ema_sep__no-base_slope__no-corr | base_slope,corr,ema_sep | 101.7 | -11.56 | 9.49 | 36.17 | 4.03 | 5.69 | -0.8 | 46 |
| best_trend__no-ema_sep__no-corr | corr,ema_sep | 101.7 | -11.56 | 9.49 | 36.17 | 4.03 | 5.69 | -0.8 | 46 |
| best_trend__no-nochop__no-corr | corr,nochop | 63.51 | -12.39 | 6.78 | 45.45 | 2.64 | 4.55 | -1.44 | 43 |
| best_trend__no-ema_sep__no-nochop__no-corr | corr,ema_sep,nochop | 88.3 | -12.69 | 8 | 44.68 | 2.95 | 5.09 | -1.39 | 46 |
| best_trend__no-nochop__no-base_slope__no-corr | base_slope,corr,nochop | 56.6 | -15.46 | 6.08 | 45.45 | 2.35 | 4.59 | -1.63 | 43 |
| best_trend__no-ema_sep__no-nochop__no-base_slope__no-corr | base_slope,corr,ema_sep,nochop | 85.96 | -15.64 | 7.66 | 44.68 | 2.75 | 5.31 | -1.56 | 46 |

# Best (module-search combo) vs Baseline best_trend

This compares the newly-found “best combo” from module hyperparam searches against our previous baseline best_trend configuration.

Files:
- Period-by-period CSV: `reports/trend_based/module_search_combo/best_vs_baseline.csv`
- Aggregated CSV: `reports/trend_based/module_search_combo/best_vs_baseline_agg.csv`

## Definitions

### Baseline best_trend (reference)
- Corr (fixed): XAG(win40 abs>=0.10 flb50 mf0) OR EUR(win75 abs>=0.10 flb75 mf5)
- EMA-sep: `ema_fast=50, ema_slow=250, atr_n=14, sep_k=0.15`
- NoChop: `ema=15, lookback=20, min_closes=12`
- FOMC: **no-entry**, 19:00Z ±2h/±2h (entry_shift=1)
- Engine lag: 1

### Best combo (from searches)
- Corr (fixed): same as baseline
- EMA-sep: `ema_fast=40, ema_slow=300, atr_n=20, sep_k=0.05`
- NoChop: `ema=20, lookback=40, min_closes=24`
- FOMC: **force-flat**, 19:00Z pre=2h, post=0.5h (entry_shift=1)
- Engine lag: 1

> Note: FOMC days are date-only (`data/econ_calendar/fomc_decision_days.csv`), so the 19:00Z anchor is an approximation (DST ignored).

## Results (period-by-period)

| Period | Variant | PnL % | MaxDD % | Sharpe | Trades |
| --- | --- | ---:| ---:| ---:| ---:|
| 2021-2022 | baseline | 10.13 | -18.28 | 0.34 | 414 |
| 2021-2022 | best_combo | 14.57 | -15.49 | 0.47 | 414 |
| 2023-2025 | baseline | 328.35 | -15.60 | 2.65 | 741 |
| 2023-2025 | best_combo | 341.39 | -16.98 | 2.62 | 737 |
| 2026 | baseline | 126.06 | -20.99 | 7.57 | 43 |
| 2026 | best_combo | 152.55 | -12.05 | 8.85 | 42 |

## Aggregate across the 3 periods

| Variant | Mean Sharpe | Worst MaxDD % | Sum PnL % |
| --- | ---:| ---:| ---:|
| baseline | 3.52 | -20.99 | 464.54 |
| best_combo | 3.98 | -16.98 | 508.50 |

## Interpretation / what changed

- **2021–2022 improves materially** under best_combo (higher Sharpe, higher PnL, and smaller drawdown). This is driven mainly by the more selective NoChop and the more conservative EMA-sep.
- **2023–2025 is mixed**: PnL slightly higher, but Sharpe slightly lower and DD somewhat worse. This is the main regime where baseline was already very strong; best_combo trades slightly differently (and a bit less).
- **2026 improves strongly**: best_combo has much better Sharpe and dramatically better MaxDD in this window. The asymmetric FOMC force-flat window contributes here, but the NoChop(20/40/24) also appears to be doing meaningful risk control.

## Recommendation

If your primary objective is **robustness / risk-adjusted performance** across regimes, best_combo is a clear improvement on this evaluation.

If you care most about preserving the exact 2023–2025 behavior, we should consider a hybrid:
- keep NoChop(20/40/24)
- keep EMA-sep(40/300/20/0.05)
- but revert FOMC to no-entry ±2h (since force-flat can change DD structure and segment boundaries).

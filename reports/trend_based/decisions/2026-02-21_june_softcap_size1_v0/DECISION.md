# Decision (experiment): june_softcap_size1_v0

Timestamp: 2026-02-21T21:47:00

Goal
----
Test a softer June seasonality rule than full force-flat:
- Keep trading in June, but cap position size to **1** (disable size=2 tier) during June.

Experiment
----------
We ran the *current canonical* best_trend config (ICM costs + churn + FOMC force-flat) in **yearly** mode with two variants:

1) **Baseline**: canonical behavior
2) **June cap size=1**: same as baseline, but for bars in June (month==6):
   - `pos = min(pos, 1.0)`

This was applied as a one-off post-processing rule in a scratch script (not yet a configurable module).

Results (yearly)
--------------
See: `yearly_compare.csv`

Headline deltas (cap1 − baseline):
- Trades: unchanged (sizing change only)
- PnL%: mixed but mostly positive;
  - 2020: +4.53%
  - 2021: +0.52%
  - 2022: -0.36%
  - 2023: +4.84% (and MaxDD improves by +3.87pp)
  - 2024: +1.36%
  - 2025: -1.02%
  - 2026: unchanged

Interpretation
--------------
- This retains exposure (unlike full June force-flat) while reducing risk via sizing.
- It still hurts in some years (2022, 2025), so seasonality is not a free lunch.
- The big 2023 MaxDD improvement suggests a June drawdown period where size=2 amplified pain.

Three-block results
-------------------
We also evaluated the three-block breakdown (same config, same post-processing cap rule):

- `three_block_compare.csv`

Headline deltas (cap1 − baseline):
- 2020–2022: **+4.69%** PnL, MaxDD improves by **+0.72pp**, Sharpe **+0.08**
- 2023–2025: **+5.18%** PnL, MaxDD improves by **+3.05pp**, Sharpe **+0.12**
- 2026 (YTD): unchanged

Interpretation update
---------------------
On the three-block aggregate view, the June size cap looks broadly beneficial (improves both PnL and MaxDD in the two large historical blocks), while keeping 2026 unchanged.

Next steps (if we pursue)
------------------------
1) Validate with walk-forward (avoid overfitting calendar effects).
2) Consider a conditional rule: only cap size in June when corr confirmation is marginal (or when chop proxy is high).
3) If promoted, implement as a clean config option (month-based size cap inside a dedicated module).

Files
-----
- best.yaml
- yearly_compare.csv
- three_block_compare.csv

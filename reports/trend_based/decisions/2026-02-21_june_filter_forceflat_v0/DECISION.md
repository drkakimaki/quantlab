# Decision (experiment): june_filter_forceflat_v0

Timestamp: 2026-02-21T21:45:00

Goal
----
Test a simple seasonality hypothesis from trade breakdown:
- June entry-month is net negative.

Experiment
----------
We ran the *current canonical* best_trend config (ICM costs + churn) in **yearly** mode with two variants:

1) **Baseline**: existing time filter only (FOMC force-flat)
2) **No-June**: baseline allow-mask AND (month != June), i.e. force-flat for all bars whose timestamp month == 6

Important: this No-June filter is a **one-off mask in a scratch script**, not yet a configurable/canonical module.

Results (yearly)
--------------
See: `yearly_compare.csv`

Headline deltas (No-June − Baseline):
- Trade count decreases in most years (expected; we block a full month).
- Performance improves in several years, but not uniformly:
  - 2020: +7.48% pnl, MaxDD improves by +1.08pp (less negative)
  - 2021: +1.03% pnl, MaxDD improves by +0.92pp
  - 2022: -0.53% pnl (worse)
  - 2023: +9.70% pnl and MaxDD improves massively (+7.74pp)
  - 2024: +4.40% pnl
  - 2025: -2.12% pnl (worse)
  - 2026: unchanged (no June data yet / partial year)

Interpretation
--------------
- June is indeed weak *on average*, but hard-blocking June can also remove profitable exposure in strong trend years.
- The big 2023 MaxDD improvement suggests June may overlap with one of the nastier drawdown episodes in that year.

Next steps (if we pursue)
------------------------
1) Validate on a more robust split (walk-forward) and/or only apply the June filter after confirming weak June persists out-of-sample.
2) Prefer a softer rule (e.g. reduced size in June) rather than full force-flat.
3) If promoted, implement as a clean, explicit config module (e.g. `time_filter.month_block: [6]`).

Files
-----
- best.yaml (canonical config at time of test)
- yearly_compare.csv
- raw/current_zero_cost_snapshot.yaml (optional snapshot)

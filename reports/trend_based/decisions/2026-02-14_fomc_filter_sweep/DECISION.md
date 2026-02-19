# 2026-02-14 — FOMC filter sweep (date-only list)

## Purpose
Test practical FOMC filters using the repo’s date-only list:
- `data/econ_calendar/fomc_decision_days.csv`

## Key insights (from `fomc_filter_summary.md`)
- Whole-day blocking is **too blunt** and significantly reduces performance.
- Narrow windows around an approximate decision time (19:00Z) are more reasonable.
- In the initial sweep (with earlier strategy params), **no-entry ±1–2h** was the best simple heuristic.

## Outcome / decision taken
- We later re-optimized FOMC handling together with tuned EMA-sep + NoChop; the best solution shifted to an asymmetric **force-flat** window (see `2026-02-14_filters_hyperparam_search`).

## Evidence
- Summary: `fomc_filter_summary.md`
- Agg CSV: `fomc_filter_results_agg.csv`
- Long CSV: `fomc_filter_results_long.csv`
- Raw HTMLs: `raw/time_filter_fomc/`

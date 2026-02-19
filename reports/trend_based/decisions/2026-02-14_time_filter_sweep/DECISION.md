# 2026-02-14 — Time-of-day session filter sweep (UTC heuristics)

## Purpose
Check if simple session filters (e.g. no Asia, no NY open hour) improve robustness for best_trend.

## Key insights (from `time_filter_summary.md`)
- **No time filter** was best overall in mean Sharpe across the 3 evaluation periods.
- The only filter that came close was blocking the **NY open spike window** (13:30–14:29 UTC), but it didn’t clearly dominate.
- Broad filters like **no Asia** or **only 07:00–20:59 UTC** were generally too blunt and reduced performance.

## Outcome / decision taken
- We did not add a session time filter to canonical best_trend.
- We moved forward to event-based filters (FOMC) and later a dedicated time_filter module.

## Evidence
- Summary: `time_filter_summary.md`
- Agg CSV: `time_filter_results_agg.csv`
- Long CSV: `time_filter_results_long.csv`
- Raw HTMLs: `raw/time_filter/`

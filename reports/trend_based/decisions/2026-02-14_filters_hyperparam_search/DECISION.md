# 2026-02-14 — Promote best_trend “module-search combo”

## Decision
Promote the module-search combo configuration to be the new canonical `best_trend`.

## Why
- Material improvement in mean Sharpe and worst drawdown across the three evaluation periods.
- Particularly improved behavior in 2021–2022 and 2026, while keeping 2023–2025 broadly strong.

## What changed
- EMA separation tuned: EMA40/300, ATR20, k=0.05
- NoChop tuned: EMA20, lookback 40, min_closes 24
- FOMC tuned: force-flat @ 19:00Z, pre=2h, post=0.5h (date-only approximation)
- Corr module unchanged (fixed best corr)

## Evidence
- Head-to-head summary: `BEST_VS_BASELINE_REPORT.md`
- Supporting sweep summaries:
  - `ema_sep_summary.md`
  - `nochop_summary.md`
  - `fomc_window_summary.md`

## Follow-up (NoChop semantics experiment)
We tested making NoChop **entry-held** (segment-held) and adding an explicit exit if NoChop is bad for N consecutive 5m bars.

Result: it improved 2021–2022 materially, but **hurt 2023–2026 badly** across Sharpe, PnL, and drawdown. Increasing N (3→6→12→24) did not recover performance.

Conclusion: keep NoChop as-is (continuous gating) for the canonical best_trend.

Artifacts:
- Grid summary: `nochop_entryheld_exit_grid.csv`
- Raw HTMLs: (archived) `raw/nochop_exit_semantics/`

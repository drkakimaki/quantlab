# 2026-02-14 — Trend module ablation (early)

## Purpose
Understand which modules actually matter (ema_sep / nochop / corr) and which are redundant.

## Key insights (from `ablation_summary.md`)
- **Corr is the dominant risk/return lever** (in that version it also affected sizing): turning corr OFF tended to improve Sharpe / reduce worst drawdown, but reduced total PnL.
- **EMA-sep and NoChop are meaningful**: disabling either generally degraded performance (especially NoChop in 2026).
- **BaseSlope was effectively redundant** vs the other gates (later removed from code).

## Outcome / decision taken
- We removed BaseSlope from the project.
- We kept corr/ema_sep/nochop as the core “knobs”, then moved to targeted hyperparam searches.

## Evidence
- Summary: `ablation_summary.md`
- Aggregate CSV: `ablation_metrics_agg.csv`
- Long CSV: `ablation_metrics_long.csv`
- Raw HTML variants: `raw/ablation/variants/`

# 2026-02-14 — Corr-stability hyperparam search (prototype + sample)

## Purpose
Tune the cross-asset corr-stability gate (XAGUSD/EURUSD) because ablation indicated it is the most sensitive/impactful module.

## Key insights (from `corr_search_summary.md`)
- **OR logic dominated** the best configs by mean Sharpe in the sampled search; AND tended to be too restrictive.
- The most important corr knobs were usually **stability constraints** (flip_count / flip_lookback) and window length; the exact abs(corr) threshold mattered but was often second-order.
- A strong pattern was: **XAG very strict on flips** (mf=0) with low abs threshold, EUR more permissive, combined via OR.

## Outcome / decision taken
- We promoted corr cfg `6a477f1f248b` as the project’s corr defaults:
  - XAG(win=40, abs>=0.10, flip_lb=50, max_flips=0)
  - EUR(win=75, abs>=0.10, flip_lb=75, max_flips=5)
  - logic=OR

## Evidence
- Summary: `corr_search_summary.md`
- Agg CSV: `corr_search_results_agg.csv`
- Long CSV: `corr_search_results_long.csv`
- Grid/spec: `corr_search_grid.json`
- Raw reports: `raw/corr_search/`

# EMA migration bundle (SMA → EMA)

Created: 2026-02-25

This folder groups all decision bundles related to the ongoing migration from the canonical SMA-based best_trend toward an EMA-based best_ema_trend.

## Contents

1) `2026-02-24_ewma_search_v1/`
   - Early EMA exploration / search.

2) `2026-02-25_ewma_holdout20_v1/`
   - EMA holdout20 sweep + follow-up variants:
     - `best.yaml` (original sweep winner)
     - `best_excl2026.yaml` (apples-to-apples scoring: exclude 2026)
     - `best_excl2026_nochurn_nomidl_norecov.yaml` (simplified pipeline)

3) `2026-02-25_ewma_nochurn_fixup_small60_v1/`
   - 60-candidate fix-up sweep starting from simplified EMA pipeline.
   - Produces the current best EMA config used for `best_ema_trend.html`.

## Notes

- Canonical SMA decisions are **not** moved here.
- This is an organizational bundle only (no results changed).

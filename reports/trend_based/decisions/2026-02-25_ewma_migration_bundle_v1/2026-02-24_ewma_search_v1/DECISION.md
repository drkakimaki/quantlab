# Decision: ewma_search_v1

Timestamp: 2026-02-25T00:37:56+01:00

Objective: maximize sum(period PnL%) subject to worst MaxDD <= 20.0%

Holdout (excluded from score): ['2026']

## Best score (EMA winner)

Winner parameters (sourced from `raw/rerun_sweep1.json` topk[0]):
- EMA base signal: fast=30, slow=90
- ema_sep.sep_k=0.09
- nochop.lookback=45, nochop.min_closes=20
- churn.min_on_bars=3, churn.cooldown_bars=6

Score summary (scored periods 2020–2025):
- OK under cap: True
- Sum PnL%: 466.63
- Worst MaxDD%: -18.67
- Avg Sharpe: 1.63

(For full period breakdown, see `results.csv`.)

## Files

- best.yaml (config snapshot)
- results.csv
- notes.json
- raw/rerun_sweep1.json
- raw/rerun_sweep2.json

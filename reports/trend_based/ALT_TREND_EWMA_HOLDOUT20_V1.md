# Alt Trend — ewma_holdout20_v1 (XAUUSD)

**Last updated:** 2026-02-25 (Europe/Berlin)

This is an **alternative** report/config generated from the decision bundle:
- `reports/trend_based/decisions/2026-02-25_ewma_holdout20_v1/`

It does **not** replace the canonical `best_trend` artifacts.

## 1) What this variant is

**Hypothesis:** long-only trend following on XAUUSD with a slightly “slower / cleaner” core signal (EMA crossover) plus a gate pipeline focused on:
- staying out of chop (EMA separation + NoChop)
- avoiding macro event windows (FOMC force-flat window)
- controlling trade frequency (churn debounce + cooldown)
- cutting toxic trades (mid-loss limiter, no-recovery exit, shock exit)

Constraint used in selection: **worst MaxDD <= 20%** (passed; best candidate had worst MaxDD ≈ -17.72%).

## 2) Reference reports (generated)

- Equity/perf: `reports/trend_based/alt_trend_ewma_holdout20_v1.html`
- Trades: `reports/trend_based/alt_trend_ewma_holdout20_v1_trades.html`

## 3) Source config (exact parameters)

Source of truth:
- `reports/trend_based/decisions/2026-02-25_ewma_holdout20_v1/best.yaml`

Key headline parameters:
- EMA base signal: fast=32, slow=110
- Costs: `fee_per_lot=3.5`, `spread_per_lot=3.5`
- Periods: three-block (2020–2022, 2023–2025, 2026 YTD)
- Time filter: **FOMC 19:00 UTC**, block **2h pre** and **0.5h post**

Pipeline (gate order):
1. `ema_sep` (sep_k=0.075)
2. `nochop` (ema=20, lookback=40, min_closes=25)
3. `time_filter`
4. `ema_strength_sizing` (strong_k=0.2, size 1->2)
5. `churn` (min_on_bars=3, cooldown_bars=8)
6. `mid_loss_limiter` (min_bars=13, max_bars=48, stop_ret=-0.01)
7. `no_recovery_exit` (bar_n=24, min_ret=-0.005)
8. `shock_exit` (abs_ret=0.006)

## 4) Notes

- This report was generated via `quantlab/scripts/report_trend_variants.py` pointing at the decision bundle config.
- Canonical best trend remains:
  - `reports/trend_based/best_trend.html`
  - `reports/trend_based/BEST_TREND_STRATEGY.md`

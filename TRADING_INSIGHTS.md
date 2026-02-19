# TRADING_INSIGHTS.md

A timeless **state + lessons** document for this repo.

What this is:
- The current "best" strategy state (what to run + where outputs live)
- Durable lessons learned (to avoid repeating mistakes)
- Pointers to decision bundles for details

What this is *not*:
- A changelog
- A dump of all sweep tables (those live under `reports/trend_based/decisions/`)

---

## Current state (canonical)

### Reports
- Best trend (canonical): `reports/trend_based/best_trend.html`
- Best trend doc: `reports/trend_based/BEST_TREND_STRATEGY.md`
- Baselines (3): `reports/baselines/`

### Canonical best_trend (XAUUSD)

**Headline metrics** (from `best_trend.html`)
- 2021-2022: PnL **22.83%**, MaxDD **-11.59%**, Sharpe **0.68**
- 2023-2025: PnL **345.01%**, MaxDD **-15.22%**, Sharpe **2.84**
- 2026: PnL **142.95%**, MaxDD **-11.75%**, Sharpe **8.97**

**Ingredients**
- Base: 5m OHLC close
- HTF: 15m OHLC confirm
- Trend signal: SMA 30/75 (base) + HTF confirm SMA 30/75
- Filters / modules (all ON):
  - EMA separation (HTF): EMA40/300 + TR-ATR20 + k=0.05
  - NoChop (HTF): EMA20, lookback 40, min_closes 24
  - Corr stability (segment-held entry gate + sizing):
    - OR logic
    - XAG: win=40, abs>=0.10, flip_lb=50, max_flips=0
    - EUR: win=75, abs>=0.10, flip_lb=75, max_flips=5
  - FOMC time filter: force-flat @ 19:00 UTC (pre=2h, post=0.5h) on decision days
  - Shock kill-switch exit: **ShockExit(abs 5m return >= 0.006)** (decision shifted by 1 bar)
- Discrete sizing: 0.01 / 0.02 lots only (mapped from size 1 / 2)

### How to run
- Best trend report (canonical YAML):
  - `.venv/bin/python scripts/report_trend_variants.py --config configs/trend_based/current.yaml --out-dir reports/trend_based --out-name best_trend.html`
- Baselines:
  - `.venv/bin/python scripts/backtest_xauusd_5m_baseline_report.py --use-ohlc-close --out reports/baselines/trend.html --strategy trend --p3-end 2026-02-13`
  - Similar calls for `mean_reversion`, `buy_and_hold`.

---

## Durable lessons (keep these in mind)

### Research / tuning
- **Filter pile risk:** prefer 1-2 strong concepts over stacking many weak gates.
- **Regime bias is real:** later years look structurally bull; we're consciously focusing on long-only trend.
- **Near-parameter ensembling** is often more robust than chasing a single best setting (use sparingly).

**What mattered (from sweeps)**
- Ablation: corr is the main risk/return lever (also sizing); EMA-sep is essential; NoChop matters especially in 2026.
  - Source: `reports/trend_based/decisions/2026-02-14_ablation/`
- Corr tuning: OR logic dominated; stability (flip limits/lookback) mattered more than abs(corr) threshold; promoted config = XAG strict flips + EUR more permissive, combined via OR.
  - Source: `reports/trend_based/decisions/2026-02-14_corr_hyperparam_search/`
- FOMC tuning: wide windows/whole-day blocking too blunt; best ended up **force-flat** with **19:00Z pre=2h post=0.5h** under tuned modules.
  - Source: `reports/trend_based/decisions/2026-02-14_fomc_filter_sweep/`

### Drawdowns / risk
- **~98–99% of drawdown deepening happens while in-position.**
  - Clarification: This measures *additional loss after entry*, not that losses occur in positions (tautological).
  - Implication: look for **post-entry risk controls** (exits/kill-switches), not more entry gating.
  - Evidence: Most entries show profit at some point; drawdowns develop over multiple bars.
  - Source: `reports/trend_based/decisions/2026-02-14_drawdown_attribution/`
- **ShockExit helps but isn't the whole story.** Many worst in-position days are not single-bar shocks.
  - Implication: the next likely lever is a **multi-bar loss limiter** (daily stop or rolling-loss exit).
  - Source: `reports/trend_based/decisions/2026-02-15_loss_drawdown_deepdive/`

### Engineering / correctness
- When refactoring, regress on **position and equity series**, not just headline metrics.
- Profit factor should be computed **per-trade** (compounded per segment), not per-bar.

---

## Decision bundles (read-on-demand)

Token hygiene: only open/read older decision bundles when explicitly discussing past tuning.

- Decisions root: `reports/trend_based/decisions/`
- Key bundles:
  - Promotion to current best (filters): `reports/trend_based/decisions/2026-02-14_filters_hyperparam_search/`
  - Ablation: `reports/trend_based/decisions/2026-02-14_ablation/`
  - Corr hyperparam search: `reports/trend_based/decisions/2026-02-14_corr_hyperparam_search/`
  - FOMC filter sweep: `reports/trend_based/decisions/2026-02-14_fomc_filter_sweep/`
  - ShockExit search + promotion: `reports/trend_based/decisions/2026-02-14_shock_exits/`
  - Drawdown attribution: `reports/trend_based/decisions/2026-02-14_drawdown_attribution/`
  - Loss/drawdown deep dive: `reports/trend_based/decisions/2026-02-15_loss_drawdown_deepdive/`

# TRADING_INSIGHTS.md

A timeless **state + lessons** document for this repo.

What this is:
- The current "best" strategy state (what config + where outputs live)
- Durable lessons learned (general + strategy-specific)
- Pointers to decision bundles for evidence

What this is *not*:
- A changelog
- A how-to / CLI doc (see README)
- A dump of all sweep tables (those live under `reports/.../decisions/`)

---

## 1) Current state (canonical)

### Canonical config
- `configs/trend_based/current.yaml`

### Canonical reports
- Best trend: `reports/trend_based/best_trend.html`
- Best trend doc: `reports/trend_based/BEST_TREND_STRATEGY.md`
- Baselines: `reports/baselines/`

### Current best_trend headline metrics (XAUUSD)
(from `best_trend.html`)
- 2020-2022: PnL **48.92%**, MaxDD **-15.85%**, Sharpe **0.83**
- 2023-2025: PnL **345.10%**, MaxDD **-16.21%**, Sharpe **2.91**
- 2026: PnL **140.70%**, MaxDD **-10.46%**, Sharpe **9.13**

### Current best_trend ingredients (high level)
- Base: 5m OHLC close
- HTF: 15m OHLC confirm
- Trend signal: SMA 30/75 (base) + HTF confirm SMA 30/75
- Modules ON: EMA-sep, NoChop, Corr stability (+ sizing), FOMC force-flat, ShockExit
- Discrete sizing: 0.01 / 0.02 lots only

---

## 2) General lessons (strategy-agnostic)

### Research / tuning
- **Filter pile risk:** prefer 1-2 strong concepts over stacking many weak gates.
- **Near-parameter ensembling** can be more robust than chasing a single best setting (use sparingly).

### Evaluation mindset
- **Regime bias is real; we are intentionally long-only:** later years appear structurally bullish, so this project currently optimizes a **long-only trend** hypothesis (not a symmetric long/short system).

### Engineering / correctness
- When refactoring, regress on **position and equity series**, not just headline metrics.
- Profit factor should be computed **per-trade** (compounded per segment), not per-bar.

---

## 3) Strategy-specific insights — best_trend (XAUUSD)

### What mattered (from sweeps)
- Ablation: corr is the main risk/return lever (also sizing); EMA-sep is essential; NoChop matters especially in 2026.
  - Source: `reports/trend_based/decisions/2026-02-14_ablation/`
- Corr tuning: OR logic dominated; stability (flip limits/lookback) mattered more than abs(corr) threshold; promoted config = XAG strict flips + EUR more permissive, combined via OR.
  - Source: `reports/trend_based/decisions/2026-02-14_corr_hyperparam_search/`
- FOMC tuning: wide windows/whole-day blocking too blunt; best ended up **force-flat** with **19:00Z pre=2h post=0.5h** under tuned modules.
  - Source: `reports/trend_based/decisions/2026-02-14_fomc_filter_sweep/`

### Behavioral fingerprints (trade breakdown)
- **Seasonality:** June is consistently negative (entry-month aggregation). Jan/Oct are strong → expect “summer chop tax”.
- **Duration-driven edge:**
  - Very long holds pay: **97+ bars** are strongly positive.
  - The toxic zone is **13–48 bars** (strongly negative).
  - Promoting the churn gate (`min_on=3`, `cooldown=8`) reduced total trades (**1449 → 1359**) and improved net PnL mainly by reducing losses in **13–48 bars**.
  - Evidence: `reports/trend_based/decisions/2026-02-21_churn_gate_debounce_cooldown_v1/`

### Risk & drawdowns (best_trend-specific)
- **~98–99% of drawdown deepening happens while in-position.**
  - Clarification: This measures *additional loss after entry*, not that losses occur in positions (tautological).
  - Implication: focus on **post-entry risk controls** (exits/kill-switches), not more entry gating.
  - Evidence: Most entries show profit at some point; drawdowns develop over multiple bars.
  - Source: `reports/trend_based/decisions/2026-02-14_drawdown_attribution/`
- **ShockExit helps but isn't the whole story.** Many worst in-position days are not single-bar shocks.
  - Implication: next likely lever is a **multi-bar loss limiter** (daily stop or rolling-loss exit).
  - Source: `reports/trend_based/decisions/2026-02-15_loss_drawdown_deepdive/`

---

## 4) Decision bundles (evidence map)

Token hygiene: only open/read older decision bundles when explicitly discussing past tuning.

### Promotion / hyperparam work
- `reports/trend_based/decisions/2026-02-14_filters_hyperparam_search/`
- `reports/trend_based/decisions/2026-02-14_corr_hyperparam_search/`
- `reports/trend_based/decisions/2026-02-14_fomc_filter_sweep/`
- `reports/trend_based/decisions/2026-02-14_shock_exits/`

### Risk research
- `reports/trend_based/decisions/2026-02-14_drawdown_attribution/`
- `reports/trend_based/decisions/2026-02-15_loss_drawdown_deepdive/`

### Correctness / audits
- `reports/trend_based/decisions/2026-02-14_lookahead_audit/`

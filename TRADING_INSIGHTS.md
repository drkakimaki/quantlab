# TRADING_INSIGHTS.md

**Last updated:** 2026-02-22 (Europe/Berlin)

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

### Baseline semantics (important)
- **Buy & Hold** baseline is now **~1x notional / unlevered** at entry:
  - `position=1.0`
  - `leverage=1.0`
  - per-period `lot_per_size` is set so notional at entry ≈ `initial_capital`
- The previous Buy & Hold implementation was implicitly **levered** (fixed lots + `position=2.0`), which overstated baseline PnL/DD.

### Current best_trend headline metrics (XAUUSD)
(from `rnd run` / canonical config)
- 2020-2022: PnL **32.47%**, MaxDD **-17.39%**, Sharpe **0.55**
- 2023-2025: PnL **346.86%**, MaxDD **-14.49%**, Sharpe **2.16**
- 2026 (HOLDOUT): PnL **161.39%**, MaxDD **-9.57%**, Sharpe **6.40**

**Sharpe definition (industry standard):** computed on **daily** close-to-close returns derived from the equity curve (UTC days), annualized with **sqrt(252)**.

### Current best_trend ingredients (high level)
- Base: 5m OHLC close
- HTF: 15m OHLC confirm
- Trend signal: SMA 30/75 (base) + HTF confirm SMA 30/75
- Modules ON: EMA-sep, NoChop, **FOMC force-flat**, **EMA-strength sizing (size=2 only on strong separation)**, **SeasonalitySizeCap (June size<=1)**, **ChurnGate (debounce+cooldown)**, **Mid-duration loss limiter (13–48, -1%)**, **Time-stop (24 bars, -0.5%)**, ShockExit
- Corr stability gate: **OFF** (removed for simplicity; replaced by EMA-strength sizing)
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
- Ablation (when controlling for sizing): **NoChop** and **Churn** are the dominant gates; HTF confirm also matters.
  - Source: `reports/trend_based/decisions/2026-02-14_ablation/`
- Corr module: historically a big lever mainly because it combined **filtering + sizing**. We have now removed it and replaced the sizing role with an **EMA-strength sizing** gate to reduce parameters.
  - Source: `reports/trend_based/decisions/2026-02-22_no_corr_ema_strength_sizing_v1/`
- Post-entry controls: the biggest incremental improvements came from (a) a mid-duration loss limiter (13–48 bars, stop -1%) and (b) a time-stop (if not recovered above -0.5% by 24 bars).
  - Sources:
    - `reports/trend_based/decisions/2026-02-22_mid_loss_limiter_stopret_-0p010_v1/`
    - `reports/trend_based/decisions/2026-02-22_time_stop_24bars_-0p5pct_v1/`
- FOMC tuning: wide windows/whole-day blocking too blunt; best ended up **force-flat** with **19:00Z pre=2h post=0.5h** under tuned modules.
  - Source: `reports/trend_based/decisions/2026-02-14_fomc_filter_sweep/`
- Econ calendar expansion (CPI/NFP): implemented infra + wiring, but **disabled in canonical** for now (did not improve train score in quick sweeps).
  - Source: `reports/trend_based/decisions/2026-02-23_disable_econ_calendar_v1/`

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
- `reports/trend_based/decisions/2026-02-14_corr_hyperparam_search/` (historical; corr gate now OFF)
- `reports/trend_based/decisions/2026-02-14_fomc_filter_sweep/`
- `reports/trend_based/decisions/2026-02-14_shock_exits/`
- `reports/trend_based/decisions/2026-02-21_churn_gate_debounce_cooldown_v1/`
- `reports/trend_based/decisions/2026-02-21_june_softcap_size1_v0/`
- `reports/trend_based/decisions/2026-02-22_no_corr_ema_strength_sizing_v1/` (corr removed; sizing via EMA strength)
- `reports/trend_based/decisions/2026-02-22_mid_loss_limiter_stopret_-0p010_v1/` (post-entry loss limiter)
- `reports/trend_based/decisions/2026-02-22_time_stop_24bars_-0p5pct_v1/` (post-entry time-stop)
- `reports/trend_based/decisions/2026-02-23_disable_econ_calendar_v1/` (econ calendar CPI/NFP disabled)

### Risk research
- `reports/trend_based/decisions/2026-02-14_drawdown_attribution/`
- `reports/trend_based/decisions/2026-02-15_loss_drawdown_deepdive/`

### Correctness / audits
- `reports/trend_based/decisions/2026-02-14_lookahead_audit/`

# TRADING_INSIGHTS.md

**Last updated:** 2026-02-25 (Europe/Berlin)

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

### Canonical config (CURRENT)
- `configs/trend_based/current.yaml` (**EMA**)
- Pinned name: `configs/trend_based/reference/best_ema_trend.yaml`

### Canonical reports (EMA)
- Best EMA trend: `reports/trend_based/best_ema_trend.html`
- Best EMA trend doc: `reports/trend_based/BEST_TREND_STRATEGY.md`
- Yearly: `reports/trend_based/best_ema_trend_yearly.html`
- Baselines: `reports/baselines/`

### Archived previous canonical (SMA)
- Config: `configs/trend_based/best_sma_trend.yaml`
- Report: `reports/trend_based/best_sma_trend.html`
- Doc: `reports/trend_based/BEST_SMA_TREND_STRATEGY.md`

### Baseline semantics (important)
- **Buy & Hold** baseline is **~1x notional / unlevered** at entry:
  - `position=1.0`
  - `leverage=1.0`
  - per-period `lot_per_size` is set so notional at entry ≈ `initial_capital`
- The previous Buy & Hold implementation was implicitly **levered** (fixed lots + `position=2.0`), which overstated baseline PnL/DD.

### Current best_trend headline metrics (EMA, XAUUSD)
(from `reports/trend_based/best_ema_trend.html`)
- 2020–2022: PnL **50.23%**, vs B&H **+30.25%**, MaxDD **-15.63%**, Sharpe **0.74** (CI **-0.31/1.63**)
- 2023–2025: PnL **387.33%**, vs B&H **+250.99%**, MaxDD **-13.43%**, Sharpe **2.29** (CI **1.35/3.09**)
- 2026 (HOLDOUT): PnL **148.89%**, vs B&H **+127.89%**, MaxDD **-18.95%**, Sharpe **5.23** (CI **1.61/7.87**)

**Sharpe definition (industry standard):** computed on **daily** close-to-close returns derived from the equity curve (UTC days), annualized with **sqrt(252)**.

### Current best_trend ingredients (high level)
- Base: 5m OHLC close
- Base trend signal: **EMA 30/110** on 5m close (long-only)
- Canonical best_trend is a **config-driven gate pipeline** (`current.yaml: pipeline:`).

High-level order:
- Base signal → entry filters (regime) → time filter (force-flat) → sizing overlays → post-entry exits

Canonical pipeline gates (EMA):
- `ema_sep` (regime: HTF EMA separation vs ATR)
- `nochop` (regime: sustained closes above HTF EMA)
- `time_filter` (force-flat: FOMC window + month blocks like flat June)
- `ema_strength_sizing` (size up only when regime is strong)
- `shock_exit` (kill-switch exit)

---

## 2) General lessons (strategy-agnostic)

### Research / tuning
- **Smoother base signal can replace gates:** moving the base signal from SMA → EMA reduced the need for some “stability” gates (e.g. churn-style trade-frequency control) while improving scored performance in our tested EMA configs.
- **Filter pile risk:** prefer 1–2 strong concepts over stacking many weak gates.
- **Near-parameter ensembling** can be more robust than chasing a single best setting (use sparingly).

### Workflow / process lessons
- Keep strong reference configs (`configs/trend_based/reference/`) to sanity-check that changes aren’t just in-sample fitting.

### Evaluation mindset
- **Regime bias is real; we are intentionally long-only:** later years appear structurally bullish, so this project currently optimizes a **long-only trend** hypothesis (not a symmetric long/short system).

### Engineering / correctness
- When refactoring, regress on **position and equity series**, not just headline metrics.
- Profit factor should be computed **per-trade** (compounded per segment), not per-bar.

---

## 3) Strategy-specific insights — best_trend (XAUUSD)

### 3.1) SMA-era insights (historical)

(Keep these as historical context for what used to work under the SMA pipeline.)

#### What mattered (from decision bundles)
- Sensitivity work suggested **NoChop** and **Churn** were dominant knobs; EMA separation also mattered.
  - Sources:
    - `reports/trend_based/decisions/2026-02-23_nochop_sensitivity_v1/`
    - `reports/trend_based/decisions/2026-02-23_churn_sensitivity_v1/`
    - `reports/trend_based/decisions/2026-02-23_ema_sep_sensitivity_v1/`
- HTF confirm (`htf_confirm`) was removed from canonical as a redundancy reduction step.
  - Evidence: `reports/trend_based/decisions/2026-02-23_drop_htf_confirm_v1/`
- Post-entry controls mattered; exit sensitivity sweep is captured here:
  - `reports/trend_based/decisions/2026-02-23_exits_sensitivity_v1/`
- Econ calendar expansion (CPI/NFP): infra was added, but later disabled in canonical (at the time).
  - `reports/trend_based/decisions/2026-02-23_disable_econ_calendar_v1/`
- June was structurally weak; mitigation moved to **force-flat June** via time_filter.
  - Evidence: `reports/trend_based/decisions/2026-02-24_move_june_flat_to_time_filter_v1/`

#### Behavioral fingerprints (trade breakdown)
- **Seasonality:** June consistently negative (entry-month aggregation). Jan/Oct are strong → expect “summer chop tax”.
- **Duration-driven edge:**
  - Very long holds pay: **97+ bars** strongly positive.
  - Toxic zone: **13–48 bars** strongly negative.
  - Churn promotion reduced total trades and improved net PnL mainly by reducing losses in **13–48 bars**.
  - Evidence: `reports/trend_based/decisions/2026-02-21_churn_gate_debounce_cooldown_v1/`

#### Risk & drawdowns
- **~98–99% of drawdown deepening happens while in-position.**
  - Implication: focus on **post-entry risk controls** (exits/kill-switches), not endless entry gating.
  - Source: `reports/trend_based/decisions/2026-02-14_drawdown_attribution/`

### 3.2) EMA migration insights (current)

#### What changed
- Canonical base signal moved from SMA to **EMA**.
- EMA canonical pipeline is intentionally **simplified** (relative to SMA-era):
  - removed: `churn`, `mid_loss_limiter`, `no_recovery_exit`
  - kept: `ema_sep`, `nochop`, `time_filter`, `ema_strength_sizing`, `shock_exit`

#### What we learned so far
- In EMA ablation, the **stability backbone** remained:
  - `ema_sep` + `nochop` + `time_filter` (removing any of these broke the DD cap in the tested setup).
- **Churn was not needed** for EMA in the tested configuration; disabling it improved scored performance.
- The 60-candidate fix-up sweep produced an EMA config that beats the archived SMA snapshot on scored (2020–2025) aggregates.

Evidence bundle:
- `reports/trend_based/decisions/2026-02-25_ewma_migration_bundle_v1/` (folder name kept as-is)
  - includes:
    - `2026-02-25_ewma_holdout20_v1/` (early EMA sweep + variants)
    - `2026-02-25_ewma_nochurn_fixup_small60_v1/` (simplified pipeline + small sweep winner)

---

## 4) Decision bundles (evidence map)

Token hygiene: only open/read older decision bundles when explicitly discussing past tuning.

### EMA migration bundle (current)
- `reports/trend_based/decisions/2026-02-25_ewma_migration_bundle_v1/` (folder name kept as-is)

### Promotion / hyperparam work (SMA-era)
- `reports/trend_based/decisions/2026-02-23_drop_htf_confirm_v1/` (promotion: removed redundant HTF confirm)
- `reports/trend_based/decisions/2026-02-23_disable_econ_calendar_v1/` (econ calendar CPI/NFP disabled)
- `reports/trend_based/decisions/2026-02-23_promote_2020_recovery_v1/`
- `reports/trend_based/decisions/2026-02-24_move_june_flat_to_time_filter_v1/` (promotion: June force-flat via time_filter)

### Sensitivities / robustness probes (SMA-era)
- `reports/trend_based/decisions/2026-02-23_ema_sep_sensitivity_v1/`
- `reports/trend_based/decisions/2026-02-23_nochop_sensitivity_v1/`
- `reports/trend_based/decisions/2026-02-23_churn_sensitivity_v1/`
- `reports/trend_based/decisions/2026-02-23_exits_sensitivity_v1/`

### Risk research (SMA-era)
- `reports/trend_based/decisions/2026-02-14_drawdown_attribution/`
- `reports/trend_based/decisions/2026-02-15_loss_drawdown_deepdive/`

### Correctness / audits
- `reports/trend_based/decisions/2026-02-14_lookahead_audit/`

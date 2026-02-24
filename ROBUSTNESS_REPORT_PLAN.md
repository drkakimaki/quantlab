# Robustness Report — Plan (Quantlab)

**Goal:** add a *single*, low-bullshit report that answers: “is this config robust, or did we just fit 2020–2025?”

This is a planning doc only. Keep the implementation small, diff-friendly, and reproducible.

---

## 0) Principles / constraints

- **Minimal metrics, maximal clarity.** Avoid a dashboard that nobody trusts.
- **One choke point** for evaluation semantics: reuse `quantlab.rnd` + WebUI loaders.
- **Diff-friendly artifacts:** CSV + a short MD summary.
- **No new alpha:** this is evaluation infra only.
- **No silent pass/fail:** always show the “why” (which years / which buckets).

---

## 1) Inputs

- Config YAML path (canonical or candidate)
- Period mode:
  - default: `three_block` (keeps current train/holdout convention)
  - robustness view: `yearly` (or derived yearly slices)
- MaxDD cap for scoring (train objective), default 20%

---

## 2) Core outputs (must-have)

### A) Train vs holdout summary
- Re-run `quantlab.rnd run` and print:
  - scored-period TOTAL: sum_pnl, worst_maxdd, avg_sharpe
  - holdout period(s): pnl/maxdd/sharpe

### B) Yearly breakdown table
For each calendar year (UTC):
- pnl_percent
- maxdd_percent
- sharpe
- trade_count (segment count)

(Implementation: either add `periods.mode=yearly` support to `rnd run`, or create a dedicated `robustness.py` that builds yearly periods and calls the same runner.)

### C) Failure mode diagnostics
- Duration-bucket PnL table (TRAIN and HOLDOUT separately)
  - bins already canonical in `reporting/trade_breakdown.py`

---

## 3) Sensitivity / perturbation tests (small, high-value)

Run a tiny local neighborhood around the config:

- `ema_sep.sep_k`: ±0.005 (e.g. 0.060/0.065/0.070)
- `nochop.min_closes`: ±2 (e.g. 18/20/22)
- optional: `churn.cooldown_bars`: ±2

For each perturbation variant, compute:
- train score tuple (sum_pnl, worst_maxdd, avg_sharpe)
- 2026 holdout maxdd

Output as `sensitivity.csv` sorted by train score, with deltas vs base.

---

## 4) Red-flag rules (warnings, not hard blockers)

Emit warnings (do not auto-reject) when:
- any single year MaxDD < -25%
- holdout MaxDD worsens by > +5pp vs reference
- duration toxic zone (13–48 bars) becomes materially more negative vs reference

---

## 5) Artifacts / file layout

Write a decision bundle under:
- `reports/trend_based/decisions/YYYY-MM-DD_robustness_<slug>/`

Files:
- `DECISION.md` (short narrative + key tables)
- `run_summary.json`
- `yearly.csv`
- `duration_train.csv`, `duration_holdout.csv`
- `sensitivity.csv`
- `notes.json` (config path, timestamp, reference config)

---

## 6) Implementation sketch

### CLI
Add: `python -m quantlab.rnd robust --config ...` **or** `python scripts/robustness_report.py --config ...`

Steps:
1) Load config
2) Compute train/holdout via existing `_run_best_trend_periods` + `_score_candidate`
3) Build yearly periods (reuse `webui.periods` style) and run per-year
4) Build trade ledgers and duration tables
5) Run small perturbation grid using improved `_set_in` (list-index support)
6) Write artifacts

---

## 7) Decisions needed (tomorrow)

- Yearly vs quarterly granularity?
- Exact warning thresholds (what is “catastrophic”)?
- Do we treat 2026 as the only holdout, or allow multiple holdouts?
- Should the report compare against a named reference config (e.g. `pre_htf_drop.yaml`)?

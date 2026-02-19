# 2026-02-14 — Lookahead / fault-confidence audit

## Purpose
Increase confidence the backtest is not benefiting from lookahead/alignment bugs.

## Key insights (from `lookahead_audit.md`)
- **HTF alignment check:** `n_future_violation = 0` for all periods (no future 15m values leaking into 5m).
- “Danger” variants (engine lag=0, corr_shift=0, size_shift=0) were **not** dramatically better than production-like settings.
- HTF-extra-shift changes results (as expected), making HTF mapping a key **porting risk** (e.g., MQL), not evidence of bias.

## Outcome / decision taken
- Keep lag=1 and last-closed-bar shifts as the production semantics.

## Evidence
- Audit narrative: `lookahead_audit.md`
- Metrics: `lookahead_metrics.csv`
- Alignment: `htf_alignment.csv`
- Raw: `raw/audits/`

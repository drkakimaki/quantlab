# Regression tests (golden series)

This folder holds **series-level regression tests** for quantlab.

Why
---
Headline metrics can stay roughly similar while the underlying execution semantics drift.
When refactoring anything that can alter signals/execution (gates, costs, time filtering,
return math, trade extraction), we regress on **position and equity series**.

Best Trend (2024–2025) golden
----------------------------
Files:
- `test_best_trend_2024_2025_series.py` — asserts:
  - index matches exactly (UTC)
  - `position` matches **exactly**
  - `equity` and `returns_net` match with tight tolerance (`atol=1e-10`)
- `update_golden_best_trend_2024_2025.py` — regenerates golden artifacts
- `golden/` — contains tracked golden artifacts:
  - `best_trend_2024_2025_series.csv.gz`
  - `best_trend_2024_2025_config.yaml`

Run
---
From repo root:

```bash
.venv/bin/python -m pytest -q tests/regression/test_best_trend_2024_2025_series.py
```

Update golden (intentional semantic changes only)
-------------------------------------------------
Only update the golden when you *intend* to change execution semantics.

```bash
.venv/bin/python tests/regression/update_golden_best_trend_2024_2025.py
```

Policy
------
- Golden updates should be accompanied by a commit message that explains *what semantic change* occurred
  (e.g. "TimeFilter: force-flat only", "Trade extraction: log1p compounding").
- If you are unsure whether a change is intended semantic drift, **do not update** the golden.
  Investigate the first-diff timestamp in the failing test.

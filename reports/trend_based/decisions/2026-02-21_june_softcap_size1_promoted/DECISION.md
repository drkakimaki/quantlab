# Decision: june_softcap_size1_promoted

Timestamp: 2026-02-21T21:49:00

Decision
--------
Promote a seasonality-based sizing rule to canonical:

- In **June** (month=6), cap position size to **1.0** (disables size=2 tier).

Implementation
--------------
Implemented as `SeasonalitySizeCapGate` configured via:

```yaml
seasonality:
  month_size_cap:
    6: 1.0
```

Rationale / evidence
--------------------
This promotion is based on the experiments:
- `2026-02-21_june_filter_forceflat_v0` (full June blackout — too blunt)
- `2026-02-21_june_softcap_size1_v0` (soft cap — preferred)

Results summary
---------------
See:
- `three_block_compare.csv`
- `yearly_compare.csv`

Headline three-block deltas (cap1 − baseline canonical without the cap):
- 2020–2022: +4.69% PnL, MaxDD improves +0.72pp
- 2023–2025: +5.18% PnL, MaxDD improves +3.05pp
- 2026: unchanged (no June yet)

Notes / cautions
----------------
- Calendar effects can overfit. Prefer to re-validate under walk-forward later.

Files
-----
- best.yaml
- three_block_compare.csv
- yearly_compare.csv

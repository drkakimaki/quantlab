# decisions/

This folder holds **small, human-readable decision bundles** for the `trend_based` strategy.

**Read policy (token hygiene):** Only open/read older decision bundles when we’re explicitly discussing past tuning/results/why a configuration was promoted. Default behavior is to not load these folders.

Goal: keep the `reports/trend_based/` area clean (current best + docs), while still preserving
just enough evidence to justify why we promoted a configuration.

## Convention

### Raw payload policy

If a decision bundle starts accumulating lots of HTMLs / grids, store them under a `raw/` subfolder while actively working.
When we’re done, archive that `raw/` folder out to `reports/_archive_*/` using:

- `scripts/archive_trend_reports.py`


Create one folder per decision:

- `YYYY-MM-DD_<short_name>/`
  - `DECISION.md` (short narrative: what changed + why + recommendation)
  - `*_summary.md` (links or copied summaries from sweeps)
  - `*_top.csv` (optional: top-N table)
  - `best_vs_baseline.csv` (optional)

Everything else (full grids, long CSVs, many HTMLs) can be archived elsewhere.

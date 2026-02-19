from __future__ import annotations

"""Build/normalize a USD economic calendar from a user-provided CSV export.

Because this environment may not have web tooling enabled, this script is designed
around *imports*:

- You export an economic calendar (CSV) from your provider.
- We map columns -> standard schema and write data/econ_calendar/usd_important_events.csv

Usage example:
  .venv/bin/python scripts/build_usd_econ_calendar_from_csv.py \
    --in /path/to/export.csv \
    --out data/econ_calendar/usd_important_events.csv

You will likely need to tweak `COLUMN_MAP` depending on the provider.
"""

import argparse
from pathlib import Path

import pandas as pd


COLUMN_MAP = {
    # standard -> possible input column names
    "ts_utc": ["ts_utc", "datetime", "date", "time", "timestamp"],
    "event": ["event", "title", "name"],
    "importance": ["importance", "impact", "severity"],
    "country": ["country"],
    "ccy": ["ccy", "currency"],
    "category": ["category"],
    "source": ["source"],
    "notes": ["notes"],
}


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/econ_calendar/usd_important_events.csv"))
    ap.add_argument("--default-country", type=str, default="US")
    ap.add_argument("--default-ccy", type=str, default="USD")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)

    out = pd.DataFrame()
    for k, cands in COLUMN_MAP.items():
        col = pick_col(df, cands)
        if col is not None:
            out[k] = df[col]

    if "country" not in out.columns:
        out["country"] = args.default_country
    if "ccy" not in out.columns:
        out["ccy"] = args.default_ccy

    if "ts_utc" not in out.columns:
        raise SystemExit("Could not find a timestamp column. Edit COLUMN_MAP.")

    # Normalize timestamp to ISO8601 Z
    ts = pd.to_datetime(out["ts_utc"], utc=True, errors="coerce")
    out["ts_utc"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Normalize importance
    if "importance" in out.columns:
        imp = out["importance"].astype(str).str.lower().str.strip()
        imp = imp.replace({"high": "high", "medium": "medium", "low": "low"})
        out["importance"] = imp

    out = out.dropna(subset=["ts_utc", "event"]).copy()

    # Keep USD + high by default
    out = out[(out["country"] == "US") | (out["ccy"] == "USD")].copy()
    if "importance" in out.columns:
        out = out[out["importance"].isin(["high", "medium", "low"])].copy()

    out = out.sort_values("ts_utc")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote: {args.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()

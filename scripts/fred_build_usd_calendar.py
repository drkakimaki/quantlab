from __future__ import annotations

"""Build USD important economic event calendar from the FRED API.

FRED provides release calendars via:
- /fred/releases
- /fred/release/dates

Docs:
https://fred.stlouisfed.org/docs/api/fred/

You need an API key:
https://fred.stlouisfed.org/docs/api/api_key.html

Usage:
  .venv/bin/python scripts/fred_build_usd_calendar.py --api-key "$FRED_API_KEY" \
    --start 2022-01-01 --end 2026-12-31 \
    --out data/econ_calendar/usd_important_events.csv

Notes:
- FRED release dates are typically "date" only (no time). We store as 00:00:00Z
  unless you provide a mapping override.
"""

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

FRED_BASE = "https://api.stlouisfed.org/fred"


DEFAULT_RELEASES: list[dict[str, Any]] = [
    # These names are for humans; IDs must be discovered/filled.
    # You can override/extend via --release-map.
    {"event": "CPI", "category": "inflation", "importance": "high", "release_id": None},
    {"event": "Core CPI", "category": "inflation", "importance": "high", "release_id": None},
    {"event": "PCE Price Index", "category": "inflation", "importance": "high", "release_id": None},
    {"event": "Nonfarm Payrolls", "category": "labor", "importance": "high", "release_id": None},
    {"event": "Unemployment Rate", "category": "labor", "importance": "high", "release_id": None},
    {"event": "FOMC Statement", "category": "rates", "importance": "high", "release_id": None},
    {"event": "Real GDP", "category": "growth", "importance": "high", "release_id": None},
    {"event": "Retail Sales", "category": "growth", "importance": "high", "release_id": None},
    {"event": "ISM Manufacturing PMI", "category": "survey", "importance": "high", "release_id": None},
    {"event": "CPI/NFP/etc (custom)", "category": "other", "importance": "high", "release_id": None},
]


def fred_get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"{FRED_BASE}{path}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def to_ts_utc(d: dt.date) -> str:
    return f"{d.isoformat()}T00:00:00Z"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build USD econ calendar from FRED release dates.")
    ap.add_argument("--api-key", type=str, default=None, help="FRED API key (or set FRED_API_KEY env var)")
    ap.add_argument("--start", type=parse_date, required=True)
    ap.add_argument("--end", type=parse_date, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/econ_calendar/usd_important_events.csv"))
    ap.add_argument(
        "--release-map",
        type=Path,
        default=Path("data/econ_calendar/fred_release_map.json"),
        help="JSON list of {event, category, importance, release_id}. Fill release_id values.",
    )
    ap.add_argument("--discover", action="store_true", help="List releases matching keywords and exit")
    ap.add_argument("--keywords", type=str, default="CPI,PCE,Employment,GDP,FOMC,Retail,ISM", help="Comma list")

    args = ap.parse_args()

    api_key = args.api_key or (Path("/dev/null").__class__.__name__ and None)
    # above trick avoids lint; replace below
    import os

    api_key = args.api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        raise SystemExit("Missing FRED API key. Pass --api-key or set FRED_API_KEY.")

    if args.discover:
        kw = [k.strip() for k in args.keywords.split(",") if k.strip()]
        # pull first 1000 releases and filter client-side
        data = fred_get("/releases", {"api_key": api_key, "file_type": "json", "limit": 1000})
        releases = data.get("releases", [])
        hits = []
        for rel in releases:
            name = str(rel.get("name", ""))
            if any(k.lower() in name.lower() for k in kw):
                hits.append({"id": rel.get("id"), "name": name, "link": rel.get("link")})
        print(json.dumps(hits[:200], indent=2))
        return

    # Ensure release map exists
    if args.release_map.exists():
        release_map = json.loads(args.release_map.read_text(encoding="utf-8"))
    else:
        args.release_map.parent.mkdir(parents=True, exist_ok=True)
        args.release_map.write_text(json.dumps(DEFAULT_RELEASES, indent=2), encoding="utf-8")
        raise SystemExit(
            f"Wrote template release map to {args.release_map}. Fill in release_id values, then rerun."
        )

    rows = []
    for item in release_map:
        rid = item.get("release_id")
        if rid in (None, "", 0):
            continue

        # FRED enforces a max page size (commonly 1000). Use paging.
        dates = []
        offset = 0
        while True:
            data = fred_get(
                "/release/dates",
                {
                    "api_key": api_key,
                    "file_type": "json",
                    "release_id": int(rid),
                    "realtime_start": args.start.isoformat(),
                    "realtime_end": args.end.isoformat(),
                    "include_release_dates_with_no_data": "false",
                    "limit": 1000,
                    "offset": offset,
                    "sort_order": "asc",
                },
            )
            batch = data.get("release_dates", [])
            if not batch:
                break
            dates.extend(batch)
            # Stop when fewer than limit returned
            if len(batch) < 1000:
                break
            offset += 1000
        for d in dates:
            ds = d.get("date")
            if not ds:
                continue
            dd = dt.date.fromisoformat(ds)
            if dd < args.start or dd > args.end:
                continue
            rows.append(
                {
                    "ts_utc": to_ts_utc(dd),
                    "country": "US",
                    "ccy": "USD",
                    "event": item.get("event", f"release_{rid}"),
                    "category": item.get("category", ""),
                    "importance": item.get("importance", "high"),
                    "source": f"FRED release_id={rid}",
                    "notes": "FRED release calendar date (time not provided; stored 00:00Z)",
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("No rows produced. Check release_id values in release map.")

    out = out.drop_duplicates(subset=["ts_utc", "event"]).sort_values(["ts_utc", "event"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote: {args.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()

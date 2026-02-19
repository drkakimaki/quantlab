from __future__ import annotations

"""Fetch FOMC decision days (meeting end dates) from the Federal Reserve site.

We use the official calendar page and extract the last date in each meeting range.
Decision day = end date of the meeting.

Source:
https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

Output:
- data/econ_calendar/fomc_decision_days.csv (date-only)
- optional merge into data/econ_calendar/usd_important_events.csv

No external HTML parser deps required (regex-based extraction).
"""

import argparse
import datetime as dt
import re
from pathlib import Path

import requests

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def to_ts_utc(d: dt.date) -> str:
    return f"{d.isoformat()}T00:00:00Z"


def parse_end(month_name: str, date_range: str, year: int) -> dt.date | None:
    """Parse date_range like '25-26' or '31-Jan 1' and return end date."""
    m1 = MONTHS.get(month_name.strip().lower())
    if not m1:
        return None
    s = date_range.strip()
    s = s.replace("–", "-")
    s = re.sub(r"\s+", " ", s)
    # Remove footnote markers like '*'
    s = re.sub(r"[^0-9A-Za-z\- ]+", "", s).strip()

    # Same-month: 25-26
    m = re.match(r"^(\d{1,2})\s*-\s*(\d{1,2})$", s)
    if m:
        d2 = int(m.group(2))
        return dt.date(year, m1, d2)

    # Cross-month: 31-Jan 1  OR 30-Jan 1
    m = re.match(r"^(\d{1,2})\s*-\s*([A-Za-z]{3,})\s*(\d{1,2})$", s)
    if m:
        mon2_raw = m.group(2).lower()
        # normalize 3-letter
        mon2 = None
        for k, v in MONTHS.items():
            if k.startswith(mon2_raw[:3]):
                mon2 = v
                break
        if mon2 is None:
            return None
        d2 = int(m.group(3))
        return dt.date(year, mon2, d2)

    return None


def extract_for_year(html: str, year: int) -> list[dt.date]:
    # Find the year header anchor: id="...">2022 FOMC Meetings
    pat_year = re.compile(rf'id="\d+">\s*{year}\s+FOMC Meetings', re.IGNORECASE)
    m = pat_year.search(html)
    if not m:
        return []
    start = m.start()

    # End at the next year section in document order (page lists years descending).
    year_hdr_re = re.compile(r'id="\d+">\s*(20\d{2})\s+FOMC Meetings', re.IGNORECASE)
    m2 = year_hdr_re.search(html, pos=m.end())
    end = m2.start() if m2 else len(html)
    block = html[start:end]

    # Match month+date within the same meeting row (no external parser needed).
    pair_re = re.compile(
        r"fomc-meeting__month[^>]*>\s*<strong>([^<]+)</strong>"  # month
        r"(?:(?!fomc-meeting__month).)*?"  # don't cross into next row's month
        r"fomc-meeting__date[^>]*>\s*([^<]+)\s*<",  # date range
        re.IGNORECASE | re.DOTALL,
    )

    out: list[dt.date] = []
    for mon, dr in pair_re.findall(block):
        d = parse_end(mon, dr, year)
        if d:
            out.append(d)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch FOMC decision days")
    ap.add_argument(
        "--url",
        type=str,
        default="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
    )
    ap.add_argument("--start-year", type=int, default=2022)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=Path("data/econ_calendar/fomc_decision_days.csv"))
    ap.add_argument("--merge", action="store_true", help="Also append to usd_important_events.csv")
    args = ap.parse_args()

    html = requests.get(args.url, timeout=30).text

    dates: list[dt.date] = []
    for y in range(int(args.start_year), int(args.end_year) + 1):
        dates.extend(extract_for_year(html, y))

    dates = sorted(set(dates))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("date,event\n" + "\n".join([f"{d.isoformat()},FOMC Decision" for d in dates]) + "\n", encoding="utf-8")
    print(f"Wrote: {args.out} ({len(dates)} rows)")

    if args.merge:
        target = Path("data/econ_calendar/usd_important_events.csv")
        if not target.exists():
            raise SystemExit(f"Missing {target}; run FRED builder first")
        import csv

        # read existing to avoid duplicates
        existing = set()
        with target.open("r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                existing.add((row.get("ts_utc"), row.get("event")))

        rows = []
        for d in dates:
            ts = to_ts_utc(d)
            key = (ts, "FOMC Decision")
            if key in existing:
                continue
            rows.append(
                {
                    "ts_utc": ts,
                    "country": "US",
                    "ccy": "USD",
                    "event": "FOMC Decision",
                    "category": "rates",
                    "importance": "high",
                    "source": "Federal Reserve (fomccalendars.htm)",
                    "notes": "Decision day = scheduled meeting end date; time not included (00:00Z)",
                }
            )

        if rows:
            with target.open("a", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                for r in rows:
                    w.writerow(r)
            print(f"Appended {len(rows)} rows to {target}")
        else:
            print("No new FOMC rows appended (already present)")


if __name__ == "__main__":
    main()

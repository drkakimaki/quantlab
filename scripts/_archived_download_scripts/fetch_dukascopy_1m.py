from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from quantlab.data.dukascopy import SYMBOL_SPECS, build_daily_1m, iter_days, latest_utc_date


def parse_day(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Dukascopy ticks and build 1-minute OHLC bars.")
    ap.add_argument("--symbols", nargs="+", default=["XAUUSD"], help="e.g. XAUUSD XAGUSD")
    ap.add_argument("--start", type=parse_day, default=dt.date(2022, 1, 1))
    ap.add_argument("--end", type=parse_day, default=None)
    ap.add_argument("--cache-dir", type=Path, default=Path("data/dukascopy_raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/dukascopy_1m"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    end = args.end or latest_utc_date()

    for sym in args.symbols:
        if sym not in SYMBOL_SPECS:
            raise SystemExit(f"Unknown symbol {sym}. Known: {sorted(SYMBOL_SPECS)}")

    total_days = (end - args.start).days + 1
    print(f"Building 1m bars for {args.symbols} from {args.start} to {end} ({total_days} days)")
    print(f"cache: {args.cache_dir} | out: {args.out_dir}")

    for sym in args.symbols:
        print(f"\n== {sym} ==")
        for i, day in enumerate(iter_days(args.start, end), start=1):
            out_path = build_daily_1m(
                symbol=sym,
                day=day,
                cache_dir=args.cache_dir,
                out_dir=args.out_dir,
                overwrite=args.overwrite,
            )
            if i % 10 == 0:
                print(f"{sym}: {i}/{total_days} -> {out_path}")

    print("Done")


if __name__ == "__main__":
    main()

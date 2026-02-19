from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from quantlab.data.resample import resample_dukascopy_1s_to_bars


def parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def main() -> None:
    ap = argparse.ArgumentParser(description="Resample Dukascopy 1s parquet (daily) to 5m bars (daily parquet).")
    ap.add_argument("--symbol", type=str, required=True)
    ap.add_argument("--start", type=parse_date, required=True)
    ap.add_argument("--end", type=parse_date, required=True)
    ap.add_argument("--root-in", type=Path, default=Path("data/dukascopy_1s"))
    ap.add_argument("--root-out", type=Path, default=Path("data/dukascopy_5m"))
    ap.add_argument("--rule", type=str, default="5m", help="Polars duration string (default 5m)")
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    n = resample_dukascopy_1s_to_bars(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        rule=args.rule,
        root_in=args.root_in,
        root_out=args.root_out,
        overwrite=args.overwrite,
    )
    print(f"Wrote {n} files under {args.root_out}/{args.symbol}")


if __name__ == "__main__":
    main()

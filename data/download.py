#!/usr/bin/env python3
"""Download Dukascopy tick data and build 1s/5m/15m parquet files.

Uses resample.py for OHLC/mid generation.

Designed for cron jobs - runs incrementally, skips existing files,
handles failures gracefully.

Usage:
    # Download last 7 days (for daily cron)
    .venv/bin/python quantlab/data/download.py --symbol XAUUSD --days 7
    
    # Download a specific period
    .venv/bin/python quantlab/data/download.py --symbol XAUUSD --start 2020-01-01 --end 2020-12-31
    
    # Backfill multiple symbols
    .venv/bin/python quantlab/data/download.py --symbols XAUUSD XAGUSD EURUSD --start 2021-01-01

Cron example (download recent data daily at 6am UTC):
    0 6 * * * cd /path/to/workspace && .venv/bin/python quantlab/data/download.py --symbols XAUUSD XAGUSD EURUSD --days 3 >> /var/log/quantlab_download.log 2>&1
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from quantlab.data.dukascopy import build_daily_1s, iter_days, latest_utc_date
from quantlab.data.resample import resample_dukascopy_1s_to_bars, resample_dukascopy_1s_to_ohlc

# Anchor default paths to the workspace root (like the WebUI does).
# This avoids accidental writes to e.g. quantlab/quantlab/data when running from
# inside the quantlab/ folder.
WORKSPACE = Path(__file__).resolve().parents[2]
QUANTLAB = WORKSPACE / "quantlab"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download Dukascopy tick data and build parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--symbols", nargs="+", default=["XAUUSD"], help="Symbols to download")
    ap.add_argument("--start", type=dt.date.fromisoformat, default=None, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", type=dt.date.fromisoformat, default=None, help="End date (YYYY-MM-DD)")
    ap.add_argument("--days", type=int, default=None, help="Download last N days (alternative to --start/--end)")
    ap.add_argument("--cache-dir", type=Path, default=QUANTLAB / "data" / "dukascopy_raw", help="Raw tick cache")
    ap.add_argument("--out-1s", type=Path, default=QUANTLAB / "data" / "dukascopy_1s", help="1s output dir")
    ap.add_argument("--out-5m", type=Path, default=QUANTLAB / "data" / "dukascopy_5m", help="5m mid output dir")
    ap.add_argument("--out-5m-ohlc", type=Path, default=QUANTLAB / "data" / "dukascopy_5m_ohlc", help="5m OHLC output dir")
    ap.add_argument("--out-15m", type=Path, default=QUANTLAB / "data" / "dukascopy_15m", help="15m mid output dir")
    ap.add_argument("--out-15m-ohlc", type=Path, default=QUANTLAB / "data" / "dukascopy_15m_ohlc", help="15m OHLC output dir")
    ap.add_argument("--no-mid", action="store_true", help="Skip mid files (only OHLC)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    ap.add_argument("--max-days", type=int, default=None, help="Max days per run (for cron safety)")
    ap.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    args = ap.parse_args()
    
    # Determine date range
    end = args.end or latest_utc_date()
    
    if args.days is not None:
        start = end - dt.timedelta(days=args.days - 1)
    elif args.start is not None:
        start = args.start
    else:
        # Default: last 7 days
        start = end - dt.timedelta(days=6)
    
    # Limit days if requested (for cron safety)
    if args.max_days is not None:
        max_end = start + dt.timedelta(days=args.max_days - 1)
        end = min(end, max_end)
    
    total_days = (end - start).days + 1
    
    if not args.quiet:
        print(
            "DOWNLOAD_START "
            f"symbols={args.symbols} start={start} end={end} days={total_days} "
            f"out_5m_ohlc={args.out_5m_ohlc} out_15m_ohlc={args.out_15m_ohlc}"
        )
    
    errors = []
    
    for symbol in args.symbols:
        if not args.quiet:
            print(f"\n== {symbol} ==")
        
        # Step 1: Download tick data → build 1s parquet
        total_1s = 0
        for i, day in enumerate(iter_days(start, end), start=1):
            try:
                build_daily_1s(
                    symbol=symbol,
                    day=day,
                    cache_dir=args.cache_dir,
                    out_dir=args.out_1s,
                    overwrite=args.overwrite,
                )
                total_1s += 1
                
                if not args.quiet and i % 10 == 0:
                    print(f"{symbol}: {i}/{total_days} days downloaded")
            except Exception as e:
                errors.append((symbol, day, f"download: {e}"))
                if not args.quiet:
                    print(f"ERROR {symbol} {day}: {e}")
        
        if not args.quiet:
            print(f"{symbol}: downloaded {total_1s} days")
        
        # Step 2: Resample to 5m/15m mid and OHLC (uses resample.py)
        if not args.no_mid:
            n_5m = resample_dukascopy_1s_to_bars(
                symbol=symbol,
                start=start,
                end=end,
                rule="5m",
                root_in=args.out_1s,
                root_out=args.out_5m,
                overwrite=args.overwrite,
            )
            n_15m = resample_dukascopy_1s_to_bars(
                symbol=symbol,
                start=start,
                end=end,
                rule="15m",
                root_in=args.out_1s,
                root_out=args.out_15m,
                overwrite=args.overwrite,
            )
            if not args.quiet:
                print(f"{symbol}: resampled {n_5m} 5m mid, {n_15m} 15m mid")
        
        n_5m_ohlc = resample_dukascopy_1s_to_ohlc(
            symbol=symbol,
            start=start,
            end=end,
            rule="5m",
            root_in=args.out_1s,
            root_out=args.out_5m_ohlc,
            overwrite=args.overwrite,
        )
        n_15m_ohlc = resample_dukascopy_1s_to_ohlc(
            symbol=symbol,
            start=start,
            end=end,
            rule="15m",
            root_in=args.out_1s,
            root_out=args.out_15m_ohlc,
            overwrite=args.overwrite,
        )
        if not args.quiet:
            print(f"{symbol}: resampled {n_5m_ohlc} 5m OHLC, {n_15m_ohlc} 15m OHLC")
    
    # Summary
    if not args.quiet:
        print(f"\nDOWNLOAD_END errors={len(errors)}")
    
    if errors:
        for symbol, day, err in errors:
            print(f"FAILED {symbol} {day}: {err}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

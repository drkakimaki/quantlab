#!/usr/bin/env python3
"""Validate data integrity across all Dukascopy data folders.

Checks for:
- Missing dates (gaps in daily files)
- Date range coverage by symbol/folder
- File counts per year

Usage:
    PYTHONPATH=. .venv/bin/python quantlab/scripts/validate_data.py
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple


class DataFolder(NamedTuple):
    """Data folder configuration."""
    name: str
    path: Path
    pattern: str  # "daily_parquet" or "tick_bi5"
    symbols: list[str] | None = None  # None = auto-detect


def is_forex_trading_day(d: date) -> bool:
    """Check if a date is a forex trading day.
    
    Forex trades Monday-Friday, but with some holidays:
    - New Year's Day (Jan 1)
    - Christmas (Dec 25)
    
    Note: Forex actually HAS data on most holidays, just lower volume.
    We'll only exclude weekends for simplicity.
    """
    return d.weekday() < 5  # Monday=0, Friday=4


def find_gaps_daily_parquet(
    symbol_path: Path,
    year: int,
) -> tuple[list[date], int]:
    """Find missing dates in daily parquet folder.
    
    Structure: {symbol}/{YYYY}/{YYYY-MM-DD}.parquet
    
    Returns (missing_dates, file_count)
    """
    year_path = symbol_path / str(year)
    if not year_path.exists():
        return [], 0
    
    files = list(year_path.glob("*.parquet"))
    file_count = len(files)
    
    # Extract dates from filenames (YYYY-MM-DD.parquet)
    found_dates = set()
    for f in files:
        try:
            stem = f.stem  # YYYY-MM-DD
            d = date.fromisoformat(stem)
            found_dates.add(d)
        except ValueError:
            continue
    
    # Check for gaps (forex trading days only)
    missing = []
    cur = date(year, 1, 1)
    end = date(year, 12, 31)
    
    while cur <= end:
        if is_forex_trading_day(cur) and cur not in found_dates:
            missing.append(cur)
        cur += timedelta(days=1)
    
    return missing, file_count


def find_gaps_tick_bi5(
    symbol_path: Path,
    year: int,
) -> tuple[list[date], int]:
    """Find missing dates in tick data folder structure.
    
    Structure: {symbol}/{YYYY}/{MM_0IDX}/{DD}/{HH}h_ticks.bi5
    Note: Dukascopy uses 0-indexed months (00=January, 11=December)
    
    Returns (missing_dates, day_count)
    """
    year_path = symbol_path / str(year)
    if not year_path.exists():
        return [], 0
    
    found_dates = set()
    
    for month_dir in year_path.iterdir():
        if not month_dir.is_dir():
            continue
        try:
            month0 = int(month_dir.name)  # 0-indexed month
            if month0 < 0 or month0 > 11:
                continue
            month = month0 + 1  # Convert to 1-indexed for date
        except ValueError:
            continue
        
        for day_dir in month_dir.iterdir():
            if not day_dir.is_dir():
                continue
            try:
                day = int(day_dir.name)
                if day < 1 or day > 31:
                    continue
                # Check if directory has tick files
                tick_files = list(day_dir.glob("*h_ticks.bi5"))
                if tick_files:
                    found_dates.add(date(year, month, day))
            except ValueError:
                continue
    
    day_count = len(found_dates)
    
    # Check for gaps (forex trading days only)
    missing = []
    cur = date(year, 1, 1)
    end = date(year, 12, 31)
    
    while cur <= end:
        if is_forex_trading_day(cur) and cur not in found_dates:
            missing.append(cur)
        cur += timedelta(days=1)
    
    return missing, day_count


def validate_folder(
    folder: DataFolder,
    verbose: bool = False,
) -> dict:
    """Validate a data folder.
    
    Returns dict with:
        - symbol: {year: {"missing": [...], "files": N, "complete": bool}}
    """
    results = {}
    
    # Get symbols
    if folder.symbols:
        symbols = folder.symbols
    else:
        symbols = [d.name for d in folder.path.iterdir() if d.is_dir()]
    
    for symbol in sorted(symbols):
        symbol_path = folder.path / symbol
        if not symbol_path.exists():
            results[symbol] = {"error": "folder not found"}
            continue
        
        results[symbol] = {}
        
        # Get years
        years = []
        for yd in symbol_path.iterdir():
            try:
                y = int(yd.name)
                if 2000 <= y <= 2030:
                    years.append(y)
            except ValueError:
                continue
        
        for year in sorted(years):
            if folder.pattern == "tick_bi5":
                missing, count = find_gaps_tick_bi5(symbol_path, year)
            else:
                missing, count = find_gaps_daily_parquet(symbol_path, year)
            
            results[symbol][year] = {
                "missing": missing[:10] if missing else [],  # Limit output
                "missing_count": len(missing),
                "files": count,
                "complete": len(missing) == 0,
            }
            
            if verbose and missing:
                print(f"  {symbol} {year}: {len(missing)} missing days")
                if len(missing) <= 5:
                    print(f"    Missing: {[d.isoformat() for d in missing]}")
    
    return results


def print_summary(results: dict, folder_name: str) -> None:
    """Print summary of validation results."""
    print(f"\n{'='*60}")
    print(f" {folder_name}")
    print(f"{'='*60}")
    
    total_complete = 0
    total_incomplete = 0
    
    for symbol, years in sorted(results.items()):
        if "error" in years:
            print(f"\n{symbol}: {years['error']}")
            continue
        
        print(f"\n{symbol}:")
        for year, data in sorted(years.items()):
            if data["complete"]:
                print(f"  {year}: ✓ {data['files']} days (complete)")
                total_complete += 1
            else:
                print(f"  {year}: ✗ {data['files']} days, {data['missing_count']} missing")
                total_incomplete += 1
    
    print(f"\nSummary: {total_complete} complete, {total_incomplete} incomplete")


def main():
    ap = argparse.ArgumentParser(description="Validate Dukascopy data")
    ap.add_argument("--verbose", "-v", action="store_true", help="Show missing dates")
    ap.add_argument("--data-root", type=Path, default=Path("quantlab/data"), help="Data root directory")
    args = ap.parse_args()
    
    data_root = args.data_root
    
    # Define folders to check
    folders = [
        DataFolder("dukascopy_raw", data_root / "dukascopy_raw", "tick_bi5"),
        DataFolder("dukascopy_1s", data_root / "dukascopy_1s", "daily_parquet"),
        DataFolder("dukascopy_5m", data_root / "dukascopy_5m", "daily_parquet"),
        DataFolder("dukascopy_5m_ohlc", data_root / "dukascopy_5m_ohlc", "daily_parquet"),
        DataFolder("dukascopy_15m", data_root / "dukascopy_15m", "daily_parquet"),
        DataFolder("dukascopy_15m_ohlc", data_root / "dukascopy_15m_ohlc", "daily_parquet"),
    ]
    
    all_results = {}
    
    for folder in folders:
        if not folder.path.exists():
            print(f"\n{folder.name}: Folder not found")
            continue
        
        print(f"\nChecking {folder.name}...")
        results = validate_folder(folder, verbose=args.verbose)
        all_results[folder.name] = results
        print_summary(results, folder.name)
    
    # Overall summary
    print(f"\n{'='*60}")
    print(" OVERALL SUMMARY")
    print(f"{'='*60}")
    
    for folder_name, results in all_results.items():
        complete = 0
        incomplete = 0
        for symbol, years in results.items():
            if "error" in years:
                continue
            for year, data in years.items():
                if data["complete"]:
                    complete += 1
                else:
                    incomplete += 1
        print(f"{folder_name}: {complete} complete, {incomplete} incomplete")


if __name__ == "__main__":
    main()

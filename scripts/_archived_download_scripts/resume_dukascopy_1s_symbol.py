from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # Windows

from quantlab.data.dukascopy import build_daily_1s, iter_days, latest_utc_date


def _latest_built_day(
    out_dir: Path,
    symbol: str,
    *,
    min_day: dt.date | None = None,
    max_day: dt.date | None = None,
) -> dt.date | None:
    """Return latest built parquet day for symbol, optionally bounded.

    This is important for backfills: if you have newer years already downloaded
    (e.g. 2026), and you want to backfill 2021, you must *ignore* those newer
    files when deciding where to resume.
    """
    root = out_dir / symbol
    if not root.exists():
        return None

    latest: dt.date | None = None
    for p in root.rglob("*.parquet"):
        try:
            d = dt.date.fromisoformat(p.stem)
        except ValueError:
            continue
        if min_day is not None and d < min_day:
            continue
        if max_day is not None and d > max_day:
            continue
        if latest is None or d > latest:
            latest = d
    return latest


def _acquire_lock(lock_path: Path) -> object | None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "w")
    if fcntl is None:
        return f
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f
    except OSError:
        f.close()
        return None


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    ap = argparse.ArgumentParser(description="Resume Dukascopy tick->1s build for a single symbol in small chunks.")
    ap.add_argument("--symbol", required=True, help="e.g. XAUUSD")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/dukascopy_raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/dukascopy_1s"))
    ap.add_argument("--start", type=dt.date.fromisoformat, default=None, help="YYYY-MM-DD (optional; otherwise resumes)")
    ap.add_argument("--end", type=dt.date.fromisoformat, default=None, help="YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--max-days", type=int, default=14, help="Max days to process this run")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    run_id = f"{_now_utc()} pid={os.getpid()}"
    print(
        f"RUN_START {run_id} symbol={args.symbol} start_arg={args.start} end_arg={args.end} max_days={args.max_days} overwrite={args.overwrite}"
    )

    lock = _acquire_lock(Path(f".cache/dukascopy_resume_{args.symbol}.lock"))
    if lock is None:
        print(f"{args.symbol}: another resume job is running; exiting")
        print(f"RUN_END {run_id} status=locked")
        return

    end = args.end or latest_utc_date()

    if args.start is not None:
        # When backfilling an earlier period (e.g. 2021) while newer data exists,
        # resume *within* [args.start, end] instead of jumping to the global latest.
        latest_in_window = _latest_built_day(
            args.out_dir,
            args.symbol,
            min_day=args.start,
            max_day=end,
        )
        start = args.start if latest_in_window is None else (latest_in_window + dt.timedelta(days=1))
    else:
        latest = _latest_built_day(args.out_dir, args.symbol)
        if latest is None:
            # default baseline
            start = dt.date(2021, 1, 1)
        else:
            start = latest + dt.timedelta(days=1)

    if start > end:
        print(f"{args.symbol}: nothing to do (start={start} > end={end})")
        print(f"RUN_END {run_id} status=nothing_to_do resolved_start={start} resolved_end={end}")
        return

    max_days = max(1, args.max_days)
    run_end = min(end, start + dt.timedelta(days=max_days - 1))

    print(f"{args.symbol}: processing {start} -> {run_end} (end target {end})")
    print(f"RUN_WINDOW {run_id} resolved_start={start} run_end={run_end} end_target={end}")

    n = 0
    for day in iter_days(start, run_end):
        build_daily_1s(
            symbol=args.symbol,
            day=day,
            cache_dir=args.cache_dir,
            out_dir=args.out_dir,
            overwrite=args.overwrite,
        )
        n += 1
        if n % 3 == 0:
            print(f"{args.symbol}: wrote {n} days (latest={day})")

    print(f"{args.symbol}: chunk complete ({n} days)")
    print(f"RUN_END {run_id} status=ok days={n} resolved_start={start} run_end={run_end}")

    # IMPORTANT: In scheduled/cron usage we want *one invocation = one chunk*.
    # Exiting explicitly prevents any wrapper from accidentally re-entering main()
    # or continuing work in the same process.
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

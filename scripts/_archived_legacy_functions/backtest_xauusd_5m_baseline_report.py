from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from quantlab import report_periods_equity_only
from quantlab.strategies import (
    buy_and_hold,
    breakout_pullback_ohlc,
    mean_reversion_zscore,
    trend_following_ma_crossover,
)
from quantlab.data.resample import load_dukascopy_mid_resample_last


def load_dukascopy_5m_mid_from_daily_parquet(
    *,
    symbol: str,
    start: dt.date,
    end: dt.date,
    root: Path,
) -> pd.Series:
    """Load pre-resampled 5m mid prices from daily parquet files."""
    paths = []
    cur = start
    one = dt.timedelta(days=1)
    while cur <= end:
        p = root / symbol / str(cur.year) / f"{cur.isoformat()}.parquet"
        if p.exists():
            paths.append(str(p))
        cur += one

    if not paths:
        raise FileNotFoundError(f"No 5m parquet files found for {symbol} in {root} between {start} and {end}")

    import polars as pl  # local import to keep script startup light

    # polars >=1.25 deprecates `streaming=` in favor of `engine=`
    df = pl.scan_parquet(paths).select(["ts", "mid"]).sort("ts").collect(engine="streaming")
    s = df.to_pandas().set_index("ts")["mid"].sort_index()
    s.name = symbol
    return s


def load_dukascopy_5m_ohlc_from_daily_parquet(
    *,
    symbol: str,
    start: dt.date,
    end: dt.date,
    root: Path,
) -> pd.DataFrame:
    """Load pre-resampled 5m OHLC bars from daily parquet files."""
    paths = []
    cur = start
    one = dt.timedelta(days=1)
    while cur <= end:
        p = root / symbol / str(cur.year) / f"{cur.isoformat()}.parquet"
        if p.exists():
            paths.append(str(p))
        cur += one

    if not paths:
        raise FileNotFoundError(f"No 5m OHLC parquet files found for {symbol} in {root} between {start} and {end}")

    import polars as pl

    df = (
        pl.scan_parquet(paths)
        .select(["ts", "open", "high", "low", "close"])
        .sort("ts")
        .collect(engine="streaming")
    )
    out = df.to_pandas().set_index("ts").sort_index()
    return out


OZ_PER_LOT = 100.0  # common spot gold contract size


def parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def main() -> None:
    ap = argparse.ArgumentParser(description="XAUUSD baseline report (resample 1s -> 5m, buy&hold per period).")
    ap.add_argument("--root", type=Path, default=Path("data/dukascopy_1s"), help="Root for 1s dukascopy data (fallback)")
    ap.add_argument("--root-5m", type=Path, default=Path("data/dukascopy_5m"), help="Root for pre-resampled 5m mid (close-only) dukascopy data")
    ap.add_argument("--root-5m-ohlc", type=Path, default=Path("data/dukascopy_5m_ohlc"), help="Root for pre-resampled 5m OHLC dukascopy data")
    ap.add_argument("--out", type=Path, default=Path("reports/xauusd_baseline_5m.html"))
    ap.add_argument(
        "--use-ohlc-close",
        action="store_true",
        help="Load 5m OHLC bars and use their close as the price series (recommended for consistency)",
    )
    ap.add_argument(
        "--strategy",
        type=str,
        default="buy_and_hold",
        choices=["buy_and_hold", "trend", "mean_reversion", "breakout_pullback"],
        help="Which baseline strategy to run",
    )
    ap.add_argument("--fee-bps", type=float, default=0.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)

    # Strategy params
    ap.add_argument("--fast", type=int, default=20, help="Trend MA fast window")
    ap.add_argument("--slow", type=int, default=100, help="Trend MA slow window")
    ap.add_argument("--lookback", type=int, default=50, help="Mean reversion z-score lookback")
    ap.add_argument("--entry-z", type=float, default=1.0, help="Mean reversion entry z")
    ap.add_argument("--exit-z", type=float, default=0.0, help="Mean reversion exit z")

    # Breakout+pullback params
    ap.add_argument("--n-breakout", type=int, default=16)
    ap.add_argument("--ema-trend", type=int, default=200)
    ap.add_argument("--ema-pull-fast", type=int, default=20)
    ap.add_argument("--ema-pull-slow", type=int, default=50)
    ap.add_argument("--atr-n", type=int, default=14)
    ap.add_argument("--atr-rising-sma", type=int, default=50)
    ap.add_argument("--ema-slope-window", type=int, default=10)
    ap.add_argument("--ema-slope-atr-mult", type=float, default=0.10)
    ap.add_argument("--stop-atr", type=float, default=1.2)
    ap.add_argument("--tp1-atr", type=float, default=1.0)
    ap.add_argument("--tp2-atr", type=float, default=2.0)
    ap.add_argument("--pb-retrace-min", type=str, default="0.30", help="Pullback retrace min (e.g. 0.30) or 'None' to disable")
    ap.add_argument("--pb-retrace-max", type=str, default="0.60", help="Pullback retrace max (e.g. 0.60) or 'None' to disable")
    ap.add_argument("--pyramid-atr", type=float, default=0.7)
    ap.add_argument("--no-pyramid", action="store_true")
    ap.add_argument("--tz", type=str, default="UTC", help="Interpret start/end datetimes in this tz (default UTC)")

    # Periods
    ap.add_argument("--p0-start", type=parse_date, default=dt.date(2021, 1, 1))
    ap.add_argument("--p0-end", type=parse_date, default=dt.date(2021, 12, 31))
    ap.add_argument("--p1-start", type=parse_date, default=dt.date(2022, 1, 1))
    ap.add_argument("--p1-end", type=parse_date, default=dt.date(2022, 12, 31))
    ap.add_argument("--p2-start", type=parse_date, default=dt.date(2023, 1, 1))
    ap.add_argument("--p2-end", type=parse_date, default=dt.date(2025, 12, 31))
    ap.add_argument("--p3-start", type=parse_date, default=dt.date(2026, 1, 1))
    ap.add_argument("--p3-end", type=parse_date, default=None)

    args = ap.parse_args()

    def opt_float(s: str) -> float | None:
        if s is None:
            return None
        s2 = str(s).strip().lower()
        if s2 in {"none", "null", ""}:
            return None
        return float(s)

    pb_retrace_min = opt_float(args.pb_retrace_min)
    pb_retrace_max = opt_float(args.pb_retrace_max)

    p3_end = args.p3_end or dt.date.today()

    def load_5m_mid_prices(a: dt.date, b: dt.date) -> pd.Series:
        """Load 5m mid prices for [a,b]. Prefer precomputed daily 5m parquet; fallback to on-the-fly resample."""
        try:
            mid_5m = load_dukascopy_5m_mid_from_daily_parquet(
                symbol="XAUUSD",
                start=a,
                end=b,
                root=args.root_5m,
            )
        except FileNotFoundError:
            mid_5m = load_dukascopy_mid_resample_last(
                symbol="XAUUSD",
                start=a,
                end=b,
                rule="5m",
                root=args.root,
            )

        if mid_5m.index.tz is None:
            mid_5m.index = mid_5m.index.tz_localize("UTC")

        return mid_5m

    bars_2122: pd.DataFrame | None = None
    bars_2325: pd.DataFrame | None = None
    bars_2026: pd.DataFrame | None = None

    # Combine 2021+2022 into a single period.
    p2122_start = args.p0_start
    p2122_end = args.p1_end

    if args.use_ohlc_close:
        bars_2122 = load_dukascopy_5m_ohlc_from_daily_parquet(
            symbol="XAUUSD", start=p2122_start, end=p2122_end, root=args.root_5m_ohlc
        )
        bars_2325 = load_dukascopy_5m_ohlc_from_daily_parquet(
            symbol="XAUUSD", start=args.p2_start, end=args.p2_end, root=args.root_5m_ohlc
        )
        bars_2026 = load_dukascopy_5m_ohlc_from_daily_parquet(
            symbol="XAUUSD", start=args.p3_start, end=p3_end, root=args.root_5m_ohlc
        )

        px_2122 = bars_2122["close"].astype(float).copy()
        px_2325 = bars_2325["close"].astype(float).copy()
        px_2026 = bars_2026["close"].astype(float).copy()

        for s in (px_2122, px_2325, px_2026):
            s.index = pd.to_datetime(s.index)
            if s.index.tz is None:
                s.index = s.index.tz_localize("UTC")
    else:
        px_2122 = load_5m_mid_prices(p2122_start, p2122_end)
        px_2325 = load_5m_mid_prices(args.p2_start, args.p2_end)
        px_2026 = load_5m_mid_prices(args.p3_start, p3_end)

    # Baseline backtests (positions are 1 lot notionally; PnL scaling handled by report initial_capital)
    if px_2122.dropna().empty:
        raise SystemExit("No XAUUSD data found in period 2021-2022 after resampling")
    if px_2325.dropna().empty:
        raise SystemExit("No XAUUSD data found in period 2023-2025 after resampling")
    if px_2026.dropna().empty:
        raise SystemExit("No XAUUSD data found in period 2026 after resampling")

    def run_strategy(px: pd.Series):
        if args.strategy == "buy_and_hold":
            return buy_and_hold(px, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
        if args.strategy == "trend":
            return trend_following_ma_crossover(
                px,
                fast=args.fast,
                slow=args.slow,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
            )
        if args.strategy == "mean_reversion":
            return mean_reversion_zscore(
                px,
                lookback=args.lookback,
                entry_z=args.entry_z,
                exit_z=args.exit_z,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
            )
        if args.strategy == "breakout_pullback":
            # Use OHLC bars for proper breakout/ATR/stop evaluation.
            if args.use_ohlc_close:
                assert bars_2122 is not None and bars_2325 is not None and bars_2026 is not None
                # Choose the matching bars by identity of the incoming series.
                if px is px_2122:
                    bars = bars_2122
                elif px is px_2325:
                    bars = bars_2325
                else:
                    bars = bars_2026
            else:
                bars = load_dukascopy_5m_ohlc_from_daily_parquet(
                    symbol="XAUUSD",
                    start=px.index[0].date(),
                    end=px.index[-1].date(),
                    root=args.root_5m_ohlc,
                )

            return breakout_pullback_ohlc(
                bars,
                n_breakout=args.n_breakout,
                ema_trend=args.ema_trend,
                ema_pull_fast=args.ema_pull_fast,
                ema_pull_slow=args.ema_pull_slow,
                atr_n=args.atr_n,
                atr_rising_sma=args.atr_rising_sma,
                ema_slope_window=args.ema_slope_window,
                ema_slope_atr_mult=args.ema_slope_atr_mult,
                pullback_retrace_min=pb_retrace_min,
                pullback_retrace_max=pb_retrace_max,
                stop_atr=args.stop_atr,
                tp1_atr=args.tp1_atr,
                tp2_atr=args.tp2_atr,
                pyramid_atr=args.pyramid_atr,
                allow_pyramid=(not args.no_pyramid),
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
            )
        raise ValueError(f"Unknown strategy: {args.strategy}")

    bt_2122, pnl_2122, exec_2122 = run_strategy(px_2122)
    bt_2325, pnl_2325, exec_2325 = run_strategy(px_2325)
    bt_2026, pnl_2026, exec_2026 = run_strategy(px_2026)

    periods = {"2021-2022": bt_2122, "2023-2025": bt_2325, "2026": bt_2026}

    initial_capital = {
        "2021-2022": 1000.0,
        "2023-2025": 1000.0,
        "2026": 1000.0,
    }

    def round_trips(executions: int) -> int:
        # Round-trip = entry+exit (2 executions). If a position is left open at the end,
        # we *do not* count it as a completed round trip.
        return int(executions // 2)

    if args.strategy == "buy_and_hold":
        n_trades = {"2021-2022": 1, "2023-2025": 1, "2026": 1}
    else:
        n_trades = {
            "2021-2022": round_trips(exec_2122),
            "2023-2025": round_trips(exec_2325),
            "2026": round_trips(exec_2026),
        }

    # Match the "trend_variants" report style: a descriptive title that encodes parameters.
    base_src = "base=5m OHLC close" if args.use_ohlc_close else "base=5m mid (close-only)"

    if args.strategy == "buy_and_hold":
        title = f"XAUUSD buy&hold [{base_src}]"
    elif args.strategy == "trend":
        title = f"XAUUSD trend (5m MA {args.fast}/{args.slow}) [{base_src}]"
    elif args.strategy == "mean_reversion":
        title = (
            f"XAUUSD mean reversion (zscore lookback={args.lookback}, entry_z={args.entry_z}, exit_z={args.exit_z}) "
            f"[{base_src}]"
        )
    else:
        title = (
            "XAUUSD breakout+pullback "
            f"(n={args.n_breakout}, ema_trend={args.ema_trend}, ema_pull={args.ema_pull_fast}/{args.ema_pull_slow}, "
            f"ATR{args.atr_n}, stop={args.stop_atr}ATR, tp1={args.tp1_atr}ATR, tp2={args.tp2_atr}ATR, "
            f"retrace={pb_retrace_min}-{pb_retrace_max}, pyramid={'off' if args.no_pyramid else f'on@{args.pyramid_atr}ATR'}) "
            f"[{base_src}; uses 5m OHLC for stops/targets]"
        )
    report_periods_equity_only(
        periods=periods,
        out_path=args.out,
        title=title,
        freq="5MIN",  # Sharpe annualization bucket
        initial_capital=initial_capital,
        n_trades=n_trades,
    )

    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()

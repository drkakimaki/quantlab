from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from quantlab import report_periods_equity_only
from quantlab.data.resample import load_dukascopy_mid_resample_last
from quantlab.strategies import trend_following_ma_crossover_htf_confirm


def parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _load_daily_paths(root: Path, symbol: str, start: dt.date, end: dt.date) -> list[str]:
    paths: list[str] = []
    cur = start
    one = dt.timedelta(days=1)
    while cur <= end:
        p = root / symbol / str(cur.year) / f"{cur.isoformat()}.parquet"
        if p.exists():
            paths.append(str(p))
        cur += one
    return paths


def load_dukascopy_5m_mid_from_daily_parquet(*, symbol: str, start: dt.date, end: dt.date, root: Path) -> pd.Series:
    paths = _load_daily_paths(root, symbol, start, end)
    if not paths:
        raise FileNotFoundError(f"No 5m parquet files found for {symbol} in {root} between {start} and {end}")

    import polars as pl

    df = pl.scan_parquet(paths).select(["ts", "mid"]).sort("ts").collect(engine="streaming")
    s = df.to_pandas().set_index("ts")["mid"].sort_index()
    s.name = symbol
    return s


def load_dukascopy_ohlc_from_daily_parquet(*, symbol: str, start: dt.date, end: dt.date, root: Path) -> pd.DataFrame:
    paths = _load_daily_paths(root, symbol, start, end)
    if not paths:
        raise FileNotFoundError(f"No OHLC parquet files found for {symbol} in {root} between {start} and {end}")

    import polars as pl

    df = pl.scan_parquet(paths).select(["ts", "open", "high", "low", "close"]).sort("ts").collect(engine="streaming")
    out = df.to_pandas().set_index("ts").sort_index()
    return out


def load_5m_prices(*, symbol: str, start: dt.date, end: dt.date, root_5m: Path, root_1s: Path) -> pd.Series:
    """Prefer precomputed daily 5m parquet; fallback to on-the-fly resample from 1s."""
    try:
        px = load_dukascopy_5m_mid_from_daily_parquet(symbol=symbol, start=start, end=end, root=root_5m)
    except FileNotFoundError:
        px = load_dukascopy_mid_resample_last(symbol=symbol, start=start, end=end, rule="5m", root=root_1s)

    px.index = pd.to_datetime(px.index)
    if px.index.tz is None:
        px.index = px.index.tz_localize("UTC")
    return px.astype(float)


def main() -> None:
    ap = argparse.ArgumentParser(description="Trend strategy variants report (15m HTF confirm + filters).")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--beta-symbol", type=str, default=None, help="Optional confirmation symbol (e.g. XAGUSD)")

    ap.add_argument("--root-1s", type=Path, default=Path("data/dukascopy_1s"))

    # Base timeframe sources
    ap.add_argument("--root-5m", type=Path, default=Path("data/dukascopy_5m"))
    ap.add_argument("--root-5m-ohlc", type=Path, default=Path("data/dukascopy_5m_ohlc"))
    ap.add_argument(
        "--use-ohlc-base",
        action="store_true",
        help="Use 5m OHLC close as the base price series (recommended for OHLC-based variant)",
    )

    # HTF OHLC source (preferred)
    ap.add_argument("--root-15m-ohlc", type=Path, default=Path("data/dukascopy_15m_ohlc"))
    ap.add_argument("--use-htf-ohlc", action="store_true", help="Use 15m OHLC bars for HTF filters (true-range ATR)")

    ap.add_argument("--out-dir", type=Path, default=Path("reports/trend_variants"))
    ap.add_argument("--out-name", type=str, default="v01_htf_15m_agree.html")

    ap.add_argument("--fee-bps", type=float, default=0.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)

    # Base MA params (on 5m)
    ap.add_argument("--fast", type=int, default=20)
    ap.add_argument("--slow", type=int, default=100)

    # HTF confirmation params
    ap.add_argument("--htf-rule", type=str, default="15min", help="Higher timeframe resample rule (default 15min)")
    ap.add_argument("--htf-fast", type=int, default=None)
    ap.add_argument("--htf-slow", type=int, default=None)

    # Option A: EMA separation filter on HTF (15m)
    ap.add_argument("--ema-sep-filter", action="store_true", help="Enable EMA separation filter on HTF")
    ap.add_argument("--ema-fast", type=int, default=50)
    ap.add_argument("--ema-slow", type=int, default=200)
    ap.add_argument("--atr-n", type=int, default=14)
    ap.add_argument("--sep-k", type=float, default=0.20, help="Separation threshold multiplier k (suggested 0.15-0.30)")

    # Option B: "no-chop" filter on HTF using EMA20 touches
    ap.add_argument("--nochop-filter", action="store_true", help="Enable no-chop filter on HTF (EMA touches)")
    ap.add_argument("--nochop-ema", type=int, default=20)
    ap.add_argument("--nochop-lookback", type=int, default=20, help="HTF candles in lookback window")
    ap.add_argument("--nochop-min-closes", type=int, default=13, help="Min closes above EMA (suggested 12-14)")

    # HTF slope filter
    ap.add_argument("--htf-slope-filter", action="store_true", help="Require HTF EMA slope > threshold (trend direction filter)")
    ap.add_argument("--htf-slope-ema", type=int, default=200)
    ap.add_argument("--htf-slope-window", type=int, default=10)
    ap.add_argument("--htf-slope-eps", type=float, default=0.0, help="Fixed slope threshold (price units)")
    ap.add_argument("--htf-slope-eps-atr-mult", type=float, default=None, help="Extra threshold = mult * ATR(HTF)")

    # Base slope filter
    ap.add_argument("--base-slope-filter", action="store_true", help="Require base EMA slope > threshold")
    ap.add_argument("--base-slope-ema", type=int, default=200)
    ap.add_argument("--base-slope-window", type=int, default=20)
    ap.add_argument("--base-slope-eps", type=float, default=0.0)

    # Entry filter
    ap.add_argument("--retest-entry", action="store_true", help="Retest entry: wait for pullback below EMA and reclaim above it")
    ap.add_argument("--retest-ema", type=int, default=20)

    # Cross-asset confirmation (beta)
    ap.add_argument("--beta-confirm", action="store_true", help="Require beta-symbol confirmation (e.g. XAGUSD) for entries")
    ap.add_argument(
        "--beta-mode",
        type=str,
        default="ema_or_reclaim",
        choices=[
            "ema_only",
            "reclaim_only",
            "ema_and_reclaim",
            "ema_or_reclaim",
            "ema_or_momentum",
            "ema_and_momentum",
            "breakout",
            "breakout_or_ema",
        ],
    )
    ap.add_argument("--beta-ema-slow", type=int, default=200)
    ap.add_argument("--beta-ema-fast", type=int, default=20)
    ap.add_argument("--beta-reclaim-lookback", type=int, default=3, help="Lookback window (bars) for beta reclaim pattern")
    ap.add_argument("--beta-mom-lookback", type=int, default=6, help="Momentum lookback (bars) for beta")
    ap.add_argument("--beta-breakout-lookback", type=int, default=12, help="Breakout lookback (bars) for beta")

    # Regime stability (rolling corr) filter
    ap.add_argument("--corr-filter", action="store_true", help="Skip entries when rolling corr is near-zero/flipping")
    ap.add_argument("--corr-symbol", type=str, default=None, help="Symbol to correlate against (e.g. XAGUSD, EURUSD)")
    ap.add_argument("--corr-window", type=int, default=100)
    ap.add_argument("--corr-min-abs", type=float, default=0.10)
    ap.add_argument("--corr-flip-lookback", type=int, default=50)
    ap.add_argument("--corr-max-flips", type=int, default=6)

    ap.add_argument("--corr2-symbol", type=str, default=None, help="Optional second corr symbol")
    ap.add_argument("--corr-logic", type=str, default="and", choices=["and", "or"], help="How to combine corr filters")
    ap.add_argument("--corr2-window", type=int, default=None)
    ap.add_argument("--corr2-min-abs", type=float, default=None)
    ap.add_argument("--corr2-flip-lookback", type=int, default=None)
    ap.add_argument("--corr2-max-flips", type=int, default=None)

    # Entry-only time filter
    ap.add_argument("--entry-hours-utc", type=str, default=None, help="Comma-separated UTC hours allowed for entries, e.g. '7,8,9,13,14,15'")

    # Entry-only date exclusion (e.g. skip FOMC decision days)
    ap.add_argument("--exclude-fomc", action="store_true", help="Exclude entries on FOMC decision days (UTC date)")
    ap.add_argument(
        "--econ-calendar",
        type=Path,
        default=Path("data/econ_calendar/usd_important_events.csv"),
        help="Path to econ calendar CSV (normalized schema) used for exclusions",
    )

    # Position sizing
    ap.add_argument("--sizing-mode", type=str, default="none", choices=["none", "vol_target", "confirm"])
    ap.add_argument("--vol-target", type=float, default=0.10)
    ap.add_argument("--vol-window", type=int, default=100)
    ap.add_argument("--confirm-size-one", type=float, default=1.0)
    ap.add_argument("--confirm-size-both", type=float, default=1.5)

    # Exit filter
    ap.add_argument("--exit-ema-filter", action="store_true", help="Trail exit: flat when base close < EMA(exit_ema)")
    ap.add_argument("--exit-ema", type=int, default=20)

    # Lever 3: conditional runner
    ap.add_argument("--runner-enable", action="store_true", help="Enable conditional runner (adds convexity)")
    ap.add_argument("--runner-frac", type=float, default=0.25, help="Runner size as fraction of base position")
    ap.add_argument("--runner-trigger-r", type=float, default=1.0, help="Activate runner once trade >= trigger_r * R")
    ap.add_argument("--runner-r-atr-window", type=int, default=14, help="R = mean abs pct return over this window")
    ap.add_argument("--runner-require-htf-on", action="store_true", help="Require HTF gate still ON to activate runner")
    ap.add_argument("--runner-require-corr-both-on", action="store_true", help="Require both corr stabilities ON to activate runner")
    ap.add_argument("--runner-exit-sma", type=int, default=30, help="Exit runner on first close < SMA(N)")
    ap.add_argument("--runner-min-hold-bars", type=int, default=0, help="Minimum bars to hold runner after activation")

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

    p3_end = args.p3_end or dt.date.today()

    def load_base_px(a: dt.date, b: dt.date) -> pd.Series:
        if args.use_ohlc_base:
            bars_5m = load_dukascopy_ohlc_from_daily_parquet(symbol=args.symbol, start=a, end=b, root=args.root_5m_ohlc)
            px = bars_5m["close"].astype(float).copy()
        else:
            px = load_5m_prices(symbol=args.symbol, start=a, end=b, root_5m=args.root_5m, root_1s=args.root_1s)
        px.index = pd.to_datetime(px.index)
        if px.index.tz is None:
            px.index = px.index.tz_localize("UTC")
        return px

    def load_htf_bars(a: dt.date, b: dt.date) -> pd.DataFrame | None:
        if not args.use_htf_ohlc:
            return None
        bars_15m = load_dukascopy_ohlc_from_daily_parquet(symbol=args.symbol, start=a, end=b, root=args.root_15m_ohlc)
        bars_15m.index = pd.to_datetime(bars_15m.index)
        if bars_15m.index.tz is None:
            bars_15m.index = bars_15m.index.tz_localize("UTC")
        return bars_15m

    px_2021 = load_base_px(args.p0_start, args.p0_end)
    px_2022 = load_base_px(args.p1_start, args.p1_end)
    px_2325 = load_base_px(args.p2_start, args.p2_end)
    px_2026 = load_base_px(args.p3_start, p3_end)

    beta_2021 = beta_2022 = beta_2325 = beta_2026 = None
    if args.beta_confirm:
        if not args.beta_symbol:
            raise SystemExit("--beta-confirm requires --beta-symbol")
        # Use 5m OHLC close for beta symbol (consistent, and we have it for XAGUSD).
        beta_2021 = load_dukascopy_ohlc_from_daily_parquet(symbol=args.beta_symbol, start=args.p0_start, end=args.p0_end, root=args.root_5m_ohlc)["close"].astype(float)
        beta_2022 = load_dukascopy_ohlc_from_daily_parquet(symbol=args.beta_symbol, start=args.p1_start, end=args.p1_end, root=args.root_5m_ohlc)["close"].astype(float)
        beta_2325 = load_dukascopy_ohlc_from_daily_parquet(symbol=args.beta_symbol, start=args.p2_start, end=args.p2_end, root=args.root_5m_ohlc)["close"].astype(float)
        beta_2026 = load_dukascopy_ohlc_from_daily_parquet(symbol=args.beta_symbol, start=args.p3_start, end=p3_end, root=args.root_5m_ohlc)["close"].astype(float)
        for s in (beta_2021, beta_2022, beta_2325, beta_2026):
            s.index = pd.to_datetime(s.index)
            if s.index.tz is None:
                s.index = s.index.tz_localize("UTC")

    corr_2021 = corr_2022 = corr_2325 = corr_2026 = None
    corr2_2021 = corr2_2022 = corr2_2325 = corr2_2026 = None
    if args.corr_filter:
        if not args.corr_symbol:
            raise SystemExit("--corr-filter requires --corr-symbol")
        corr_2021 = load_dukascopy_ohlc_from_daily_parquet(symbol=args.corr_symbol, start=args.p0_start, end=args.p0_end, root=args.root_5m_ohlc)["close"].astype(float)
        corr_2022 = load_dukascopy_ohlc_from_daily_parquet(symbol=args.corr_symbol, start=args.p1_start, end=args.p1_end, root=args.root_5m_ohlc)["close"].astype(float)
        corr_2325 = load_dukascopy_ohlc_from_daily_parquet(symbol=args.corr_symbol, start=args.p2_start, end=args.p2_end, root=args.root_5m_ohlc)["close"].astype(float)
        corr_2026 = load_dukascopy_ohlc_from_daily_parquet(symbol=args.corr_symbol, start=args.p3_start, end=p3_end, root=args.root_5m_ohlc)["close"].astype(float)
        for s in (corr_2021, corr_2022, corr_2325, corr_2026):
            s.index = pd.to_datetime(s.index)
            if s.index.tz is None:
                s.index = s.index.tz_localize("UTC")

        if args.corr2_symbol:
            corr2_2021 = load_dukascopy_ohlc_from_daily_parquet(symbol=args.corr2_symbol, start=args.p0_start, end=args.p0_end, root=args.root_5m_ohlc)["close"].astype(float)
            corr2_2022 = load_dukascopy_ohlc_from_daily_parquet(symbol=args.corr2_symbol, start=args.p1_start, end=args.p1_end, root=args.root_5m_ohlc)["close"].astype(float)
            corr2_2325 = load_dukascopy_ohlc_from_daily_parquet(symbol=args.corr2_symbol, start=args.p2_start, end=args.p2_end, root=args.root_5m_ohlc)["close"].astype(float)
            corr2_2026 = load_dukascopy_ohlc_from_daily_parquet(symbol=args.corr2_symbol, start=args.p3_start, end=p3_end, root=args.root_5m_ohlc)["close"].astype(float)
            for s in (corr2_2021, corr2_2022, corr2_2325, corr2_2026):
                s.index = pd.to_datetime(s.index)
                if s.index.tz is None:
                    s.index = s.index.tz_localize("UTC")

    htf_2021 = load_htf_bars(args.p0_start, args.p0_end)
    htf_2022 = load_htf_bars(args.p1_start, args.p1_end)
    htf_2325 = load_htf_bars(args.p2_start, args.p2_end)
    htf_2026 = load_htf_bars(args.p3_start, p3_end)

    entry_hours = None
    if args.entry_hours_utc:
        entry_hours = {int(x.strip()) for x in args.entry_hours_utc.split(',') if x.strip() != ''}

    exclude_dates = None
    if args.exclude_fomc:
        if not args.econ_calendar.exists():
            raise SystemExit(f"econ calendar not found: {args.econ_calendar}")
        cal = pd.read_csv(args.econ_calendar)
        if "event" not in cal.columns or "ts_utc" not in cal.columns:
            raise SystemExit("econ calendar must have columns: ts_utc,event")
        fomc = cal[cal["event"].astype(str) == "FOMC Decision"].copy()
        if not fomc.empty:
            dts = pd.to_datetime(fomc["ts_utc"], utc=True, errors="coerce")
            exclude_dates = set(dts.dropna().dt.date.tolist())
        else:
            exclude_dates = set()

    def run(
        px: pd.Series,
        htf_bars: pd.DataFrame | None,
        beta_close: pd.Series | None,
        corr_close: pd.Series | None,
        corr_close2: pd.Series | None,
    ):
        return trend_following_ma_crossover_htf_confirm(
            px,
            fast=args.fast,
            slow=args.slow,
            htf_rule=args.htf_rule,
            htf_fast=args.htf_fast,
            htf_slow=args.htf_slow,
            htf_bars=htf_bars,
            ema_sep_filter=args.ema_sep_filter,
            ema_fast=args.ema_fast,
            ema_slow=args.ema_slow,
            atr_n=args.atr_n,
            sep_k=args.sep_k,
            nochop_filter=args.nochop_filter,
            nochop_ema=args.nochop_ema,
            nochop_lookback=args.nochop_lookback,
            nochop_min_closes=args.nochop_min_closes,
            htf_slope_filter=args.htf_slope_filter,
            htf_slope_ema=args.htf_slope_ema,
            htf_slope_window=args.htf_slope_window,
            htf_slope_eps=args.htf_slope_eps,
            htf_slope_eps_atr_mult=args.htf_slope_eps_atr_mult,
            base_slope_filter=args.base_slope_filter,
            base_slope_ema=args.base_slope_ema,
            base_slope_window=args.base_slope_window,
            base_slope_eps=args.base_slope_eps,
            retest_entry=args.retest_entry,
            retest_ema=args.retest_ema,
            exit_ema_filter=args.exit_ema_filter,
            exit_ema=args.exit_ema,
            beta_confirm=args.beta_confirm,
            beta_close=beta_close,
            beta_mode=args.beta_mode,
            beta_ema_slow=args.beta_ema_slow,
            beta_ema_fast=args.beta_ema_fast,
            beta_reclaim_lookback=args.beta_reclaim_lookback,
            beta_mom_lookback=args.beta_mom_lookback,
            beta_breakout_lookback=args.beta_breakout_lookback,
            corr_filter=args.corr_filter,
            corr_close=corr_close,
            corr_window=args.corr_window,
            corr_min_abs=args.corr_min_abs,
            corr_flip_lookback=args.corr_flip_lookback,
            corr_max_flips=args.corr_max_flips,
            corr_close2=corr_close2,
            corr2_window=args.corr2_window,
            corr2_min_abs=args.corr2_min_abs,
            corr2_flip_lookback=args.corr2_flip_lookback,
            corr2_max_flips=args.corr2_max_flips,
            corr_logic=args.corr_logic,
            entry_hours_utc=entry_hours,
            exclude_entry_dates_utc=exclude_dates,
            sizing_mode=args.sizing_mode,
            vol_target=args.vol_target,
            vol_window=args.vol_window,
            confirm_size_one=args.confirm_size_one,
            confirm_size_both=args.confirm_size_both,
            runner_enable=args.runner_enable,
            runner_frac=args.runner_frac,
            runner_trigger_r=args.runner_trigger_r,
            runner_r_atr_window=args.runner_r_atr_window,
            runner_require_htf_on=args.runner_require_htf_on,
            runner_require_corr_both_on=args.runner_require_corr_both_on,
            runner_exit_sma=args.runner_exit_sma,
            runner_min_hold_bars=args.runner_min_hold_bars,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            long_only=True,
        )

    # Combine 2021+2022 into a single period.
    px_2122 = load_base_px(args.p0_start, args.p1_end)
    htf_2122 = load_htf_bars(args.p0_start, args.p1_end)

    beta_2122 = None
    if args.beta_confirm:
        beta_2122 = load_dukascopy_ohlc_from_daily_parquet(
            symbol=args.beta_symbol, start=args.p0_start, end=args.p1_end, root=args.root_5m_ohlc
        )["close"].astype(float)
        beta_2122.index = pd.to_datetime(beta_2122.index)
        if beta_2122.index.tz is None:
            beta_2122.index = beta_2122.index.tz_localize("UTC")

    corr_2122 = None
    corr2_2122 = None
    if args.corr_filter:
        corr_2122 = load_dukascopy_ohlc_from_daily_parquet(
            symbol=args.corr_symbol, start=args.p0_start, end=args.p1_end, root=args.root_5m_ohlc
        )["close"].astype(float)
        corr_2122.index = pd.to_datetime(corr_2122.index)
        if corr_2122.index.tz is None:
            corr_2122.index = corr_2122.index.tz_localize("UTC")

        if args.corr2_symbol:
            corr2_2122 = load_dukascopy_ohlc_from_daily_parquet(
                symbol=args.corr2_symbol, start=args.p0_start, end=args.p1_end, root=args.root_5m_ohlc
            )["close"].astype(float)
            corr2_2122.index = pd.to_datetime(corr2_2122.index)
            if corr2_2122.index.tz is None:
                corr2_2122.index = corr2_2122.index.tz_localize("UTC")

    bt_2122, _, exec_2122 = run(px_2122, htf_2122, beta_2122, corr_2122, corr2_2122)
    bt_2325, _, exec_2325 = run(px_2325, htf_2325, beta_2325, corr_2325, corr2_2325)
    bt_2026, _, exec_2026 = run(px_2026, htf_2026, beta_2026, corr_2026, corr2_2026)

    periods = {"2021-2022": bt_2122, "2023-2025": bt_2325, "2026": bt_2026}

    # USD account backtest now uses a fixed starting capital.
    initial_capital = {
        "2021-2022": 1000.0,
        "2023-2025": 1000.0,
        "2026": 1000.0,
    }

    def round_trips(executions: int) -> int:
        return int(executions // 2)

    n_trades = {
        "2021-2022": round_trips(exec_2122),
        "2023-2025": round_trips(exec_2325),
        "2026": round_trips(exec_2026),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.out_name

    title = f"{args.symbol} trend (5m MA {args.fast}/{args.slow})"
    if args.use_ohlc_base:
        title += " [base=5m OHLC close]"
    if args.use_htf_ohlc:
        title += f" + HTF({args.htf_rule}) [bars=15m OHLC]"
    else:
        title += f" + HTF({args.htf_rule})"

    if args.ema_sep_filter:
        atr_kind = "TR-ATR" if args.use_htf_ohlc else "close-ATR"
        title += f" + EMAsep(HTF EMA{args.ema_fast}/{args.ema_slow}, {atr_kind}{args.atr_n}, k={args.sep_k})"
    if args.nochop_filter:
        title += f" + NoChop(HTF EMA{args.nochop_ema}, lookback={args.nochop_lookback}, min={args.nochop_min_closes})"
    if args.htf_slope_filter:
        thr = f"eps={args.htf_slope_eps}"
        if args.htf_slope_eps_atr_mult is not None:
            thr += f"+{args.htf_slope_eps_atr_mult}*ATR"
        title += f" + HTFSlope(EMA{args.htf_slope_ema}, win={args.htf_slope_window}, {thr})"
    if args.base_slope_filter:
        title += f" + BaseSlope(EMA{args.base_slope_ema}, win={args.base_slope_window}, eps={args.base_slope_eps})"
    if args.retest_entry:
        title += f" + RetestEntry(retestEMA{args.retest_ema})"
    if args.beta_confirm:
        title += (
            f" + BetaConfirm({args.beta_symbol}, mode={args.beta_mode}, "
            f"ema{args.beta_ema_fast}/{args.beta_ema_slow}, reclaimN={args.beta_reclaim_lookback}, "
            f"momN={args.beta_mom_lookback}, brN={args.beta_breakout_lookback})"
        )
    if args.corr_filter:
        extra = ""
        if args.corr2_symbol:
            w2 = args.corr2_window if args.corr2_window is not None else args.corr_window
            a2 = args.corr2_min_abs if args.corr2_min_abs is not None else args.corr_min_abs
            flb2 = args.corr2_flip_lookback if args.corr2_flip_lookback is not None else args.corr_flip_lookback
            mf2 = args.corr2_max_flips if args.corr2_max_flips is not None else args.corr_max_flips
            extra = f" {args.corr_logic.upper()} {args.corr2_symbol}(win={w2}, abs>={a2}, flips<={mf2}/{flb2})"
        title += f" + CorrStability({args.corr_symbol}, win={args.corr_window}, abs>={args.corr_min_abs}, flips<={args.corr_max_flips}/{args.corr_flip_lookback}){extra}"
    if args.sizing_mode != "none":
        if args.sizing_mode == "vol_target":
            title += f" + Size(vol_target={args.vol_target}, win={args.vol_window})"
        else:
            title += f" + Size(confirm one={args.confirm_size_one}, both={args.confirm_size_both})"
    if args.exit_ema_filter:
        title += f" + ExitTrail(base close<EMA{args.exit_ema})"

    report_periods_equity_only(
        periods=periods,
        out_path=out_path,
        title=title,
        freq="5MIN",
        initial_capital=initial_capital,
        n_trades=n_trades,
    )

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

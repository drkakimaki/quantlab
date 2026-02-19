from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.metrics import performance_summary
from quantlab.strategies import trend_following_ma_crossover_htf_confirm


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


def load_ohlc_from_daily_parquet(*, symbol: str, start: dt.date, end: dt.date, root: Path) -> pd.DataFrame:
    paths = _load_daily_paths(root, symbol, start, end)
    if not paths:
        raise FileNotFoundError(f"No OHLC parquet files found for {symbol} in {root} between {start} and {end}")

    import polars as pl

    df = pl.scan_parquet(paths).select(["ts", "open", "high", "low", "close"]).sort("ts").collect(engine="streaming")
    out = df.to_pandas().set_index("ts").sort_index()
    out.index = pd.to_datetime(out.index)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    return out


def concat_returns(*bts: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    rets = []
    for bt in bts:
        if bt is None or bt.empty:
            continue
        rets.append(bt["returns_net"].copy())
    if not rets:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    r_all = pd.concat(rets).sort_index()
    e_all = (1.0 + r_all.fillna(0.0)).cumprod()
    return r_all, e_all


def score_summary(s) -> float:
    # max_drawdown is negative; higher is better.
    return float(s.sharpe) + 2.0 * float(s.max_drawdown)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extensive (random) hyperparam search for OHLC trend variant.")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--root-5m-ohlc", type=Path, default=Path("data/dukascopy_5m_ohlc"))
    ap.add_argument("--root-15m-ohlc", type=Path, default=Path("data/dukascopy_15m_ohlc"))
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-csv", type=Path, default=Path("reports/trend_variants/extensive_search_results.csv"))

    # Periods
    ap.add_argument("--p1-start", type=lambda s: dt.date.fromisoformat(s), default=dt.date(2022, 1, 1))
    ap.add_argument("--p1-end", type=lambda s: dt.date.fromisoformat(s), default=dt.date(2022, 12, 31))
    ap.add_argument("--p2-start", type=lambda s: dt.date.fromisoformat(s), default=dt.date(2023, 1, 1))
    ap.add_argument("--p2-end", type=lambda s: dt.date.fromisoformat(s), default=dt.date(2025, 12, 31))
    ap.add_argument("--p3-start", type=lambda s: dt.date.fromisoformat(s), default=dt.date(2026, 1, 1))
    ap.add_argument("--p3-end", type=lambda s: dt.date.fromisoformat(s), default=None)

    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))

    p3_end = args.p3_end or dt.date.today()

    # Load data once
    bars5_2022 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=args.p1_start, end=args.p1_end, root=args.root_5m_ohlc)
    bars5_2325 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=args.p2_start, end=args.p2_end, root=args.root_5m_ohlc)
    bars5_2026 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=args.p3_start, end=p3_end, root=args.root_5m_ohlc)

    bars15_2022 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=args.p1_start, end=args.p1_end, root=args.root_15m_ohlc)
    bars15_2325 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=args.p2_start, end=args.p2_end, root=args.root_15m_ohlc)
    bars15_2026 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=args.p3_start, end=p3_end, root=args.root_15m_ohlc)

    px_2022 = bars5_2022["close"].astype(float)
    px_2325 = bars5_2325["close"].astype(float)
    px_2026 = bars5_2026["close"].astype(float)

    # Search spaces (kept sensible to avoid absurd combos)
    fast_choices = np.array([10, 15, 20, 25, 30], dtype=int)
    slow_choices = np.array([50, 75, 100, 125, 150, 200], dtype=int)

    sep_k_choices = np.array([0.15, 0.20, 0.25, 0.27, 0.30, 0.33, 0.35, 0.40], dtype=float)

    nochop_min_choices = np.array([10, 11, 12, 13, 14, 15], dtype=int)
    nochop_lookback_choices = np.array([15, 20, 25, 30], dtype=int)
    nochop_ema_choices = np.array([15, 20, 30], dtype=int)

    ema_fast_choices = np.array([20, 30, 50, 80], dtype=int)
    ema_slow_choices = np.array([120, 150, 200, 250], dtype=int)
    atr_n_choices = np.array([10, 14, 20], dtype=int)

    # Slope thresholded HTF slope (best so far) — include both on/off.
    htf_slope_on_choices = np.array([0, 1], dtype=int)
    htf_slope_ema_choices = np.array([30, 50, 100, 200], dtype=int)
    htf_slope_window_choices = np.array([3, 4, 6, 8, 10], dtype=int)
    htf_slope_atr_mult_choices = np.array([None, 0.01, 0.015, 0.02, 0.03, 0.04], dtype=object)

    # Base slope (optional)
    base_slope_on_choices = np.array([0, 1], dtype=int)
    base_slope_ema_choices = np.array([50, 100, 200], dtype=int)
    base_slope_window_choices = np.array([10, 20, 30], dtype=int)

    rows = []

    def run_one(px: pd.Series, htf_bars: pd.DataFrame, params: dict) -> pd.DataFrame:
        bt, _, _ = trend_following_ma_crossover_htf_confirm(px, htf_bars=htf_bars, **params)
        return bt

    for i in range(int(args.iters)):
        fast = int(rng.choice(fast_choices))
        slow = int(rng.choice(slow_choices))
        if fast >= slow:
            continue

        ema_fast = int(rng.choice(ema_fast_choices))
        ema_slow = int(rng.choice(ema_slow_choices))
        if ema_fast >= ema_slow:
            continue

        atr_n = int(rng.choice(atr_n_choices))
        sep_k = float(rng.choice(sep_k_choices))

        nochop_ema = int(rng.choice(nochop_ema_choices))
        nochop_lookback = int(rng.choice(nochop_lookback_choices))
        nochop_min = int(rng.choice(nochop_min_choices))
        if nochop_min > nochop_lookback:
            continue

        htf_slope_on = bool(int(rng.choice(htf_slope_on_choices)))
        htf_slope_ema = int(rng.choice(htf_slope_ema_choices))
        htf_slope_window = int(rng.choice(htf_slope_window_choices))
        htf_slope_atr_mult = rng.choice(htf_slope_atr_mult_choices)

        base_slope_on = bool(int(rng.choice(base_slope_on_choices)))
        base_slope_ema = int(rng.choice(base_slope_ema_choices))
        base_slope_window = int(rng.choice(base_slope_window_choices))

        params = dict(
            fast=fast,
            slow=slow,
            htf_rule="15min",
            htf_fast=None,
            htf_slow=None,
            # Option A
            ema_sep_filter=True,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            atr_n=atr_n,
            sep_k=sep_k,
            # Option B
            nochop_filter=True,
            nochop_ema=nochop_ema,
            nochop_lookback=nochop_lookback,
            nochop_min_closes=nochop_min,
            # HTF slope
            htf_slope_filter=htf_slope_on,
            htf_slope_ema=htf_slope_ema,
            htf_slope_window=htf_slope_window,
            htf_slope_eps=0.0,
            htf_slope_eps_atr_mult=(None if htf_slope_atr_mult is None else float(htf_slope_atr_mult)),
            # base slope
            base_slope_filter=base_slope_on,
            base_slope_ema=base_slope_ema,
            base_slope_window=base_slope_window,
            base_slope_eps=0.0,
            # retest/exit disabled for this search (they hurt in our tests)
            retest_entry=False,
            exit_ema_filter=False,
            fee_bps=0.0,
            slippage_bps=0.0,
            long_only=True,
        )

        bt1 = run_one(px_2022, bars15_2022, params)
        bt2 = run_one(px_2325, bars15_2325, params)
        bt3 = run_one(px_2026, bars15_2026, params)

        r_all, e_all = concat_returns(bt1, bt2, bt3)
        summ = performance_summary(r_all, e_all, freq="5MIN")
        score = score_summary(summ)

        rows.append(
            {
                "iter": i,
                "score": score,
                "final_equity": float(e_all.iloc[-1]) if len(e_all) else float("nan"),
                **asdict(summ),
                # params
                "fast": fast,
                "slow": slow,
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "atr_n": atr_n,
                "sep_k": sep_k,
                "nochop_ema": nochop_ema,
                "nochop_lookback": nochop_lookback,
                "nochop_min": nochop_min,
                "htf_slope_on": htf_slope_on,
                "htf_slope_ema": htf_slope_ema,
                "htf_slope_window": htf_slope_window,
                "htf_slope_atr_mult": (None if htf_slope_atr_mult is None else float(htf_slope_atr_mult)),
                "base_slope_on": base_slope_on,
                "base_slope_ema": base_slope_ema,
                "base_slope_window": base_slope_window,
            }
        )

        if (i + 1) % 100 == 0:
            print(f"{i+1}/{args.iters} done")

    df = pd.DataFrame(rows).sort_values(["score"], ascending=False).reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    with pd.option_context("display.max_columns", 80, "display.width", 220):
        print(df.head(20).to_string(index=False))

    best = df.iloc[0].to_dict() if len(df) else None
    print("\nBEST:")
    print(best)


if __name__ == "__main__":
    main()

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
    return float(s.sharpe) + 2.0 * float(s.max_drawdown)


def main() -> None:
    ap = argparse.ArgumentParser(description="Search beta-confirmation params (XAGUSD) for XAUUSD strategy.")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--beta-symbol", type=str, default="XAGUSD")

    ap.add_argument("--root-5m-ohlc", type=Path, default=Path("data/dukascopy_5m_ohlc"))
    ap.add_argument("--root-15m-ohlc", type=Path, default=Path("data/dukascopy_15m_ohlc"))

    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--seed", type=int, default=17)

    ap.add_argument("--out-csv", type=Path, default=Path("reports/trend_variants/beta_confirm_search.csv"))

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

    # Load base and beta data once (5m OHLC close for both, plus 15m OHLC for HTF)
    bars5_2022 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=args.p1_start, end=args.p1_end, root=args.root_5m_ohlc)
    bars5_2325 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=args.p2_start, end=args.p2_end, root=args.root_5m_ohlc)
    bars5_2026 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=args.p3_start, end=p3_end, root=args.root_5m_ohlc)

    beta5_2022 = load_ohlc_from_daily_parquet(symbol=args.beta_symbol, start=args.p1_start, end=args.p1_end, root=args.root_5m_ohlc)
    beta5_2325 = load_ohlc_from_daily_parquet(symbol=args.beta_symbol, start=args.p2_start, end=args.p2_end, root=args.root_5m_ohlc)
    beta5_2026 = load_ohlc_from_daily_parquet(symbol=args.beta_symbol, start=args.p3_start, end=p3_end, root=args.root_5m_ohlc)

    bars15_2022 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=args.p1_start, end=args.p1_end, root=args.root_15m_ohlc)
    bars15_2325 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=args.p2_start, end=args.p2_end, root=args.root_15m_ohlc)
    bars15_2026 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=args.p3_start, end=p3_end, root=args.root_15m_ohlc)

    px_2022 = bars5_2022["close"].astype(float)
    px_2325 = bars5_2325["close"].astype(float)
    px_2026 = bars5_2026["close"].astype(float)

    bc_2022 = beta5_2022["close"].astype(float)
    bc_2325 = beta5_2325["close"].astype(float)
    bc_2026 = beta5_2026["close"].astype(float)

    # Fixed strategy params = v12 extensive best
    fixed = dict(
        fast=30,
        slow=75,
        htf_rule="15min",
        htf_fast=None,
        htf_slow=None,
        ema_sep_filter=True,
        ema_fast=50,
        ema_slow=250,
        atr_n=14,
        sep_k=0.15,
        nochop_filter=True,
        nochop_ema=15,
        nochop_lookback=20,
        nochop_min_closes=12,
        # slope
        base_slope_filter=True,
        base_slope_ema=100,
        base_slope_window=30,
        base_slope_eps=0.0,
        htf_slope_filter=False,
        htf_slope_ema=50,
        htf_slope_window=4,
        htf_slope_eps=0.0,
        htf_slope_eps_atr_mult=None,
        # retest/exit off
        retest_entry=False,
        exit_ema_filter=False,
        fee_bps=0.0,
        slippage_bps=0.0,
        long_only=True,
        # beta (enabled; beta_close passed per-period)
        beta_confirm=True,
    )

    # Search space for beta filter
    ema_slow_choices = np.array([150, 200, 250, 300], dtype=int)
    ema_fast_choices = np.array([10, 15, 20, 30, 50], dtype=int)
    reclaim_choices = np.array([0, 1, 2, 3, 5], dtype=int)

    rows = []

    def run_one(px: pd.Series, htf_bars: pd.DataFrame, beta_close: pd.Series, params: dict) -> pd.DataFrame:
        bt, _, _ = trend_following_ma_crossover_htf_confirm(px, htf_bars=htf_bars, beta_close=beta_close, **params)
        return bt

    for i in range(int(args.iters)):
        beta_ema_slow = int(rng.choice(ema_slow_choices))
        beta_ema_fast = int(rng.choice(ema_fast_choices))
        if beta_ema_fast >= beta_ema_slow:
            continue
        beta_reclaim = int(rng.choice(reclaim_choices))

        params = dict(fixed)
        params.update(
            beta_ema_slow=beta_ema_slow,
            beta_ema_fast=beta_ema_fast,
            beta_reclaim_lookback=beta_reclaim,
        )

        bt1 = run_one(px_2022, bars15_2022, bc_2022, params)
        bt2 = run_one(px_2325, bars15_2325, bc_2325, params)
        bt3 = run_one(px_2026, bars15_2026, bc_2026, params)

        r_all, e_all = concat_returns(bt1, bt2, bt3)
        summ = performance_summary(r_all, e_all, freq="5MIN")
        score = score_summary(summ)

        rows.append(
            {
                "iter": i,
                "score": score,
                "final_equity": float(e_all.iloc[-1]) if len(e_all) else float("nan"),
                **asdict(summ),
                "beta_ema_slow": beta_ema_slow,
                "beta_ema_fast": beta_ema_fast,
                "beta_reclaim_lookback": beta_reclaim,
            }
        )

    df = pd.DataFrame(rows).sort_values(["score"], ascending=False).reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    with pd.option_context("display.max_columns", 50, "display.width", 200):
        print(df.head(20).to_string(index=False))

    best = df.iloc[0].to_dict() if len(df) else None
    print("\nBEST:")
    print(best)


if __name__ == "__main__":
    main()

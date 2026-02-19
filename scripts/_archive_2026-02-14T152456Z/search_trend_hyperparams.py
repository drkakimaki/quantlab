from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.data.resample import load_dukascopy_mid_resample_last
from quantlab.metrics import performance_summary
from quantlab.strategies import trend_following_ma_crossover_htf_confirm


def parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def load_dukascopy_5m_mid_from_daily_parquet(*, symbol: str, start: dt.date, end: dt.date, root: Path) -> pd.Series:
    paths: list[str] = []
    cur = start
    one = dt.timedelta(days=1)
    while cur <= end:
        p = root / symbol / str(cur.year) / f"{cur.isoformat()}.parquet"
        if p.exists():
            paths.append(str(p))
        cur += one

    if not paths:
        raise FileNotFoundError

    import polars as pl

    df = pl.scan_parquet(paths).select(["ts", "mid"]).sort("ts").collect(engine="streaming")
    s = df.to_pandas().set_index("ts")["mid"].sort_index()
    s.name = symbol
    s.index = pd.to_datetime(s.index)
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    return s.astype(float)


def load_5m_prices(*, symbol: str, start: dt.date, end: dt.date, root_5m: Path, root_1s: Path) -> pd.Series:
    try:
        return load_dukascopy_5m_mid_from_daily_parquet(symbol=symbol, start=start, end=end, root=root_5m)
    except FileNotFoundError:
        px = load_dukascopy_mid_resample_last(symbol=symbol, start=start, end=end, rule="5m", root=root_1s)
        px.index = pd.to_datetime(px.index)
        if px.index.tz is None:
            px.index = px.index.tz_localize("UTC")
        return px.astype(float)


def concat_returns(*bts: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    rets = []
    eq = []
    for bt in bts:
        if bt is None or bt.empty:
            continue
        r = bt["returns_net"].copy()
        e = bt["equity"].copy()
        rets.append(r)
        eq.append(e)
    if not rets:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    r_all = pd.concat(rets).sort_index()

    # For equity, rebuild from concatenated returns to avoid discontinuities between periods.
    e_all = (1.0 + r_all.fillna(0.0)).cumprod()
    return r_all, e_all


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid search for (k, M) on the trend variant.")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--root-1s", type=Path, default=Path("data/dukascopy_1s"))
    ap.add_argument("--root-5m", type=Path, default=Path("data/dukascopy_5m"))

    ap.add_argument("--fast", type=int, default=20)
    ap.add_argument("--slow", type=int, default=100)

    ap.add_argument("--htf-rule", type=str, default="15min")

    # Option A/B search ranges
    ap.add_argument("--k", type=str, default="0.15,0.20,0.25,0.30")
    ap.add_argument("--m", type=str, default="12,13,14")

    ap.add_argument("--ema-fast", type=int, default=50)
    ap.add_argument("--ema-slow", type=int, default=200)
    ap.add_argument("--atr-n", type=int, default=14)

    ap.add_argument("--nochop-ema", type=int, default=20)
    ap.add_argument("--nochop-lookback", type=int, default=20)

    ap.add_argument("--fee-bps", type=float, default=0.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)

    # Periods
    ap.add_argument("--p1-start", type=parse_date, default=dt.date(2022, 1, 1))
    ap.add_argument("--p1-end", type=parse_date, default=dt.date(2022, 12, 31))
    ap.add_argument("--p2-start", type=parse_date, default=dt.date(2023, 1, 1))
    ap.add_argument("--p2-end", type=parse_date, default=dt.date(2025, 12, 31))
    ap.add_argument("--p3-start", type=parse_date, default=dt.date(2026, 1, 1))
    ap.add_argument("--p3-end", type=parse_date, default=None)

    args = ap.parse_args()

    ks = [float(x.strip()) for x in args.k.split(",") if x.strip()]
    ms = [int(x.strip()) for x in args.m.split(",") if x.strip()]

    p3_end = args.p3_end or dt.date.today()

    px_2022 = load_5m_prices(symbol=args.symbol, start=args.p1_start, end=args.p1_end, root_5m=args.root_5m, root_1s=args.root_1s)
    px_2325 = load_5m_prices(symbol=args.symbol, start=args.p2_start, end=args.p2_end, root_5m=args.root_5m, root_1s=args.root_1s)
    px_2026 = load_5m_prices(symbol=args.symbol, start=args.p3_start, end=p3_end, root_5m=args.root_5m, root_1s=args.root_1s)

    rows = []
    for k in ks:
        for m in ms:
            bt1, _, _ = trend_following_ma_crossover_htf_confirm(
                px_2022,
                fast=args.fast,
                slow=args.slow,
                htf_rule=args.htf_rule,
                ema_sep_filter=True,
                ema_fast=args.ema_fast,
                ema_slow=args.ema_slow,
                atr_n=args.atr_n,
                sep_k=k,
                nochop_filter=True,
                nochop_ema=args.nochop_ema,
                nochop_lookback=args.nochop_lookback,
                nochop_min_closes=m,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
                long_only=True,
            )
            bt2, _, _ = trend_following_ma_crossover_htf_confirm(
                px_2325,
                fast=args.fast,
                slow=args.slow,
                htf_rule=args.htf_rule,
                ema_sep_filter=True,
                ema_fast=args.ema_fast,
                ema_slow=args.ema_slow,
                atr_n=args.atr_n,
                sep_k=k,
                nochop_filter=True,
                nochop_ema=args.nochop_ema,
                nochop_lookback=args.nochop_lookback,
                nochop_min_closes=m,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
                long_only=True,
            )
            bt3, _, _ = trend_following_ma_crossover_htf_confirm(
                px_2026,
                fast=args.fast,
                slow=args.slow,
                htf_rule=args.htf_rule,
                ema_sep_filter=True,
                ema_fast=args.ema_fast,
                ema_slow=args.ema_slow,
                atr_n=args.atr_n,
                sep_k=k,
                nochop_filter=True,
                nochop_ema=args.nochop_ema,
                nochop_lookback=args.nochop_lookback,
                nochop_min_closes=m,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
                long_only=True,
            )

            r_all, e_all = concat_returns(bt1, bt2, bt3)
            summ = performance_summary(r_all, e_all, freq="5MIN")

            # Score: prioritize Sharpe, then smaller drawdown.
            # max_drawdown is negative; higher is better.
            score = float(summ.sharpe) + 2.0 * float(summ.max_drawdown)

            rows.append(
                {
                    "k": k,
                    "m": m,
                    "score": score,
                    **asdict(summ),
                    "final_equity": float(e_all.iloc[-1]) if len(e_all) else float("nan"),
                    "n_bars": int(len(r_all)),
                }
            )

    df = pd.DataFrame(rows).sort_values(["score"], ascending=False).reset_index(drop=True)

    # Print top results
    with pd.option_context("display.max_rows", 50, "display.max_columns", 50, "display.width", 200):
        print(df.head(15).to_string(index=False))

    best = df.iloc[0].to_dict() if len(df) else None
    if best is None:
        raise SystemExit("No results")

    print("\nBEST:")
    print(best)


if __name__ == "__main__":
    main()

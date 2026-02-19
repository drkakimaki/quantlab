from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from quantlab.metrics import performance_summary
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


def main() -> None:
    ap = argparse.ArgumentParser(description="OHLC-based grid search for (k, M) on the trend variant.")
    ap.add_argument("--symbol", type=str, default="XAUUSD")

    ap.add_argument("--root-5m-ohlc", type=Path, default=Path("data/dukascopy_5m_ohlc"))
    ap.add_argument("--root-15m-ohlc", type=Path, default=Path("data/dukascopy_15m_ohlc"))

    ap.add_argument("--fast", type=int, default=20)
    ap.add_argument("--slow", type=int, default=100)

    ap.add_argument("--htf-rule", type=str, default="15min")

    # Option A/B search ranges
    ap.add_argument("--k", type=str, default="0.25,0.27,0.30,0.33,0.35")
    ap.add_argument("--m", type=str, default="11,12,13,14")

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

    bars5_2022 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=args.p1_start, end=args.p1_end, root=args.root_5m_ohlc)
    bars5_2325 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=args.p2_start, end=args.p2_end, root=args.root_5m_ohlc)
    bars5_2026 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=args.p3_start, end=p3_end, root=args.root_5m_ohlc)

    bars15_2022 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=args.p1_start, end=args.p1_end, root=args.root_15m_ohlc)
    bars15_2325 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=args.p2_start, end=args.p2_end, root=args.root_15m_ohlc)
    bars15_2026 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=args.p3_start, end=p3_end, root=args.root_15m_ohlc)

    px_2022 = bars5_2022["close"].astype(float)
    px_2325 = bars5_2325["close"].astype(float)
    px_2026 = bars5_2026["close"].astype(float)

    rows = []
    for k in ks:
        for m in ms:
            bt1, _, _ = trend_following_ma_crossover_htf_confirm(
                px_2022,
                fast=args.fast,
                slow=args.slow,
                htf_rule=args.htf_rule,
                htf_bars=bars15_2022,
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
                htf_bars=bars15_2325,
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
                htf_bars=bars15_2026,
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

    with pd.option_context("display.max_rows", 50, "display.max_columns", 50, "display.width", 200):
        print(df.head(20).to_string(index=False))

    best = df.iloc[0].to_dict() if len(df) else None
    if best is None:
        raise SystemExit("No results")

    print("\nBEST:")
    print(best)


if __name__ == "__main__":
    main()

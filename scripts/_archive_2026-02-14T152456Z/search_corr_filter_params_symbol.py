from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import asdict
from pathlib import Path

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


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid search corr-stability filter params vs a chosen symbol.")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--corr-symbol", type=str, default="EURUSD")
    ap.add_argument("--root-5m-ohlc", type=Path, default=Path("data/dukascopy_5m_ohlc"))
    ap.add_argument("--root-15m-ohlc", type=Path, default=Path("data/dukascopy_15m_ohlc"))
    ap.add_argument("--out", type=Path, default=Path("reports/trend_variants/eurusd_filters/corr_search.csv"))

    args = ap.parse_args()

    root_5m_ohlc = args.root_5m_ohlc
    root_15m_ohlc = args.root_15m_ohlc

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    # Periods
    p1_start, p1_end = dt.date(2022, 1, 1), dt.date(2022, 12, 31)
    p2_start, p2_end = dt.date(2023, 1, 1), dt.date(2025, 12, 31)
    p3_start, p3_end = dt.date(2026, 1, 1), dt.date.today()

    xau5_2022 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=p1_start, end=p1_end, root=root_5m_ohlc)
    xau5_2325 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=p2_start, end=p2_end, root=root_5m_ohlc)
    xau5_2026 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=p3_start, end=p3_end, root=root_5m_ohlc)

    corr5_2022 = load_ohlc_from_daily_parquet(symbol=args.corr_symbol, start=p1_start, end=p1_end, root=root_5m_ohlc)
    corr5_2325 = load_ohlc_from_daily_parquet(symbol=args.corr_symbol, start=p2_start, end=p2_end, root=root_5m_ohlc)
    corr5_2026 = load_ohlc_from_daily_parquet(symbol=args.corr_symbol, start=p3_start, end=p3_end, root=root_5m_ohlc)

    xau15_2022 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=p1_start, end=p1_end, root=root_15m_ohlc)
    xau15_2325 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=p2_start, end=p2_end, root=root_15m_ohlc)
    xau15_2026 = load_ohlc_from_daily_parquet(symbol=args.symbol, start=p3_start, end=p3_end, root=root_15m_ohlc)

    px_2022 = xau5_2022["close"].astype(float)
    px_2325 = xau5_2325["close"].astype(float)
    px_2026 = xau5_2026["close"].astype(float)

    cc_2022 = corr5_2022["close"].astype(float)
    cc_2325 = corr5_2325["close"].astype(float)
    cc_2026 = corr5_2026["close"].astype(float)

    # Baseline v12 params (fixed) w/out beta confirm.
    base_params = dict(
        fast=30,
        slow=75,
        htf_rule="15min",
        ema_sep_filter=True,
        ema_fast=50,
        ema_slow=250,
        atr_n=14,
        sep_k=0.15,
        nochop_filter=True,
        nochop_ema=15,
        nochop_lookback=20,
        nochop_min_closes=12,
        base_slope_filter=True,
        base_slope_ema=100,
        base_slope_window=30,
        base_slope_eps=0.0,
        # disable other experiments
        retest_entry=False,
        exit_ema_filter=False,
        beta_confirm=False,
        corr_filter=True,
        fee_bps=0.0,
        slippage_bps=0.0,
        long_only=True,
    )

    windows = [50, 75, 100, 150]
    min_abs = [0.05, 0.10, 0.15, 0.20]
    flip_lb = [25, 50, 75]
    max_flips = [2, 4, 6, 8]

    rows = []
    for w in windows:
        for a in min_abs:
            for flb in flip_lb:
                for mf in max_flips:
                    params = dict(base_params)
                    params.update(
                        corr_window=w,
                        corr_min_abs=a,
                        corr_flip_lookback=flb,
                        corr_max_flips=mf,
                    )

                    bt1, _, _ = trend_following_ma_crossover_htf_confirm(px_2022, htf_bars=xau15_2022, corr_close=cc_2022, **params)
                    bt2, _, _ = trend_following_ma_crossover_htf_confirm(px_2325, htf_bars=xau15_2325, corr_close=cc_2325, **params)
                    bt3, _, _ = trend_following_ma_crossover_htf_confirm(px_2026, htf_bars=xau15_2026, corr_close=cc_2026, **params)

                    r_all, e_all = concat_returns(bt1, bt2, bt3)
                    summ = performance_summary(r_all, e_all, freq="5MIN")

                    score = float(100.0 * summ.cagr) + 0.5 * float(summ.sharpe) + 0.5 * float(summ.max_drawdown * 100.0)

                    rows.append(
                        {
                            "score": score,
                            "final_equity": float(e_all.iloc[-1]) if len(e_all) else float("nan"),
                            **asdict(summ),
                            "corr_window": w,
                            "corr_min_abs": a,
                            "corr_flip_lookback": flb,
                            "corr_max_flips": mf,
                        }
                    )

    df = pd.DataFrame(rows).sort_values(["score"], ascending=False).reset_index(drop=True)
    df.to_csv(out, index=False)

    with pd.option_context("display.max_columns", 50, "display.width", 220):
        print(df.head(15).to_string(index=False))

    best = df.iloc[0].to_dict()
    print("\nBEST:")
    print(best)


if __name__ == "__main__":
    main()

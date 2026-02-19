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


def main() -> None:
    ap = argparse.ArgumentParser(description="Random search over combined corr-stability filters (XAGUSD + EURUSD).")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("reports/trend_variants/eurusd_filters/combined_corr_search.csv"))
    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))

    # Roots
    root_5m_ohlc = Path("data/dukascopy_5m_ohlc")
    root_15m_ohlc = Path("data/dukascopy_15m_ohlc")

    # Periods
    p1_start, p1_end = dt.date(2022, 1, 1), dt.date(2022, 12, 31)
    p2_start, p2_end = dt.date(2023, 1, 1), dt.date(2025, 12, 31)
    p3_start, p3_end = dt.date(2026, 1, 1), dt.date.today()

    # Load base XAU and HTF bars
    xau5_2022 = load_ohlc_from_daily_parquet(symbol="XAUUSD", start=p1_start, end=p1_end, root=root_5m_ohlc)
    xau5_2325 = load_ohlc_from_daily_parquet(symbol="XAUUSD", start=p2_start, end=p2_end, root=root_5m_ohlc)
    xau5_2026 = load_ohlc_from_daily_parquet(symbol="XAUUSD", start=p3_start, end=p3_end, root=root_5m_ohlc)

    xau15_2022 = load_ohlc_from_daily_parquet(symbol="XAUUSD", start=p1_start, end=p1_end, root=root_15m_ohlc)
    xau15_2325 = load_ohlc_from_daily_parquet(symbol="XAUUSD", start=p2_start, end=p2_end, root=root_15m_ohlc)
    xau15_2026 = load_ohlc_from_daily_parquet(symbol="XAUUSD", start=p3_start, end=p3_end, root=root_15m_ohlc)

    # Corr symbols
    xag5_2022 = load_ohlc_from_daily_parquet(symbol="XAGUSD", start=p1_start, end=p1_end, root=root_5m_ohlc)
    xag5_2325 = load_ohlc_from_daily_parquet(symbol="XAGUSD", start=p2_start, end=p2_end, root=root_5m_ohlc)
    xag5_2026 = load_ohlc_from_daily_parquet(symbol="XAGUSD", start=p3_start, end=p3_end, root=root_5m_ohlc)

    eur5_2022 = load_ohlc_from_daily_parquet(symbol="EURUSD", start=p1_start, end=p1_end, root=root_5m_ohlc)
    eur5_2325 = load_ohlc_from_daily_parquet(symbol="EURUSD", start=p2_start, end=p2_end, root=root_5m_ohlc)
    eur5_2026 = load_ohlc_from_daily_parquet(symbol="EURUSD", start=p3_start, end=p3_end, root=root_5m_ohlc)

    px_2022 = xau5_2022["close"].astype(float)
    px_2325 = xau5_2325["close"].astype(float)
    px_2026 = xau5_2026["close"].astype(float)

    cc1_2022 = xag5_2022["close"].astype(float)
    cc1_2325 = xag5_2325["close"].astype(float)
    cc1_2026 = xag5_2026["close"].astype(float)

    cc2_2022 = eur5_2022["close"].astype(float)
    cc2_2325 = eur5_2325["close"].astype(float)
    cc2_2026 = eur5_2026["close"].astype(float)

    # Fixed baseline v12 parameters
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
        fee_bps=0.0,
        slippage_bps=0.0,
        long_only=True,
        # corr enabled
        corr_filter=True,
    )

    # Sampling spaces
    windows = [25, 50, 75, 100, 150, 200]
    min_abs = [0.05, 0.10, 0.15, 0.20, 0.25]
    flip_lb = [25, 50, 75, 100]
    max_flips = [0, 1, 2, 4, 6, 8]
    logic = ["and", "or"]

    rows = []

    for i in range(int(args.iters)):
        w1 = int(rng.choice(windows))
        a1 = float(rng.choice(min_abs))
        flb1 = int(rng.choice(flip_lb))
        mf1 = int(rng.choice(max_flips))

        w2 = int(rng.choice(windows))
        a2 = float(rng.choice(min_abs))
        flb2 = int(rng.choice(flip_lb))
        mf2 = int(rng.choice(max_flips))

        lg = str(rng.choice(logic))

        params = dict(base_params)
        params.update(
            corr_window=w1,
            corr_min_abs=a1,
            corr_flip_lookback=flb1,
            corr_max_flips=mf1,
            corr2_window=w2,
            corr2_min_abs=a2,
            corr2_flip_lookback=flb2,
            corr2_max_flips=mf2,
            corr_logic=lg,
        )

        bt1, _, _ = trend_following_ma_crossover_htf_confirm(px_2022, htf_bars=xau15_2022, corr_close=cc1_2022, corr_close2=cc2_2022, **params)
        bt2, _, _ = trend_following_ma_crossover_htf_confirm(px_2325, htf_bars=xau15_2325, corr_close=cc1_2325, corr_close2=cc2_2325, **params)
        bt3, _, _ = trend_following_ma_crossover_htf_confirm(px_2026, htf_bars=xau15_2026, corr_close=cc1_2026, corr_close2=cc2_2026, **params)

        r_all, e_all = concat_returns(bt1, bt2, bt3)
        summ = performance_summary(r_all, e_all, freq="5MIN")

        # score focused on CAGR/Sharpe with DD penalty
        score = float(100.0 * summ.cagr) + 0.5 * float(summ.sharpe) + 0.5 * float(summ.max_drawdown * 100.0)

        rows.append(
            {
                "iter": i,
                "score": score,
                "final_equity": float(e_all.iloc[-1]) if len(e_all) else float("nan"),
                **asdict(summ),
                "logic": lg,
                "w1": w1,
                "a1": a1,
                "flb1": flb1,
                "mf1": mf1,
                "w2": w2,
                "a2": a2,
                "flb2": flb2,
                "mf2": mf2,
            }
        )

        if (i + 1) % 100 == 0:
            print(f"{i+1}/{args.iters} done")

    df = pd.DataFrame(rows).sort_values(["score"], ascending=False).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    with pd.option_context("display.max_columns", 50, "display.width", 220):
        print(df.head(15).to_string(index=False))

    best = df.iloc[0].to_dict()
    print("\nBEST:")
    print(best)


if __name__ == "__main__":
    main()

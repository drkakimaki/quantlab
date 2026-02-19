from __future__ import annotations

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


def score_summary(s) -> float:
    # max_drawdown is negative; higher is better.
    return float(s.sharpe) + 2.0 * float(s.max_drawdown)


def run_variant(
    *,
    name: str,
    px_2022: pd.Series,
    px_2325: pd.Series,
    px_2026: pd.Series,
    htf_2022: pd.DataFrame,
    htf_2325: pd.DataFrame,
    htf_2026: pd.DataFrame,
    k: float,
    m: int,
    common_kwargs: dict,
    extra_kwargs: dict,
) -> dict:
    bt1, _, _ = trend_following_ma_crossover_htf_confirm(
        px_2022,
        htf_bars=htf_2022,
        sep_k=k,
        nochop_min_closes=m,
        **common_kwargs,
        **extra_kwargs,
    )
    bt2, _, _ = trend_following_ma_crossover_htf_confirm(
        px_2325,
        htf_bars=htf_2325,
        sep_k=k,
        nochop_min_closes=m,
        **common_kwargs,
        **extra_kwargs,
    )
    bt3, _, _ = trend_following_ma_crossover_htf_confirm(
        px_2026,
        htf_bars=htf_2026,
        sep_k=k,
        nochop_min_closes=m,
        **common_kwargs,
        **extra_kwargs,
    )

    r_all, e_all = concat_returns(bt1, bt2, bt3)
    summ = performance_summary(r_all, e_all, freq="5MIN")

    return {
        "variant": name,
        "score": score_summary(summ),
        **asdict(summ),
        "final_equity": float(e_all.iloc[-1]) if len(e_all) else float("nan"),
        "n_bars": int(len(r_all)),
    }


def main() -> None:
    symbol = "XAUUSD"

    root_5m_ohlc = Path("data/dukascopy_5m_ohlc")
    root_15m_ohlc = Path("data/dukascopy_15m_ohlc")

    p1_start, p1_end = dt.date(2022, 1, 1), dt.date(2022, 12, 31)
    p2_start, p2_end = dt.date(2023, 1, 1), dt.date(2025, 12, 31)
    p3_start, p3_end = dt.date(2026, 1, 1), dt.date.today()

    bars5_2022 = load_ohlc_from_daily_parquet(symbol=symbol, start=p1_start, end=p1_end, root=root_5m_ohlc)
    bars5_2325 = load_ohlc_from_daily_parquet(symbol=symbol, start=p2_start, end=p2_end, root=root_5m_ohlc)
    bars5_2026 = load_ohlc_from_daily_parquet(symbol=symbol, start=p3_start, end=p3_end, root=root_5m_ohlc)

    bars15_2022 = load_ohlc_from_daily_parquet(symbol=symbol, start=p1_start, end=p1_end, root=root_15m_ohlc)
    bars15_2325 = load_ohlc_from_daily_parquet(symbol=symbol, start=p2_start, end=p2_end, root=root_15m_ohlc)
    bars15_2026 = load_ohlc_from_daily_parquet(symbol=symbol, start=p3_start, end=p3_end, root=root_15m_ohlc)

    px_2022 = bars5_2022["close"].astype(float)
    px_2325 = bars5_2325["close"].astype(float)
    px_2026 = bars5_2026["close"].astype(float)

    # Baseline best params
    k = 0.25
    m = 12

    common = dict(
        fast=20,
        slow=100,
        htf_rule="15min",
        ema_sep_filter=True,
        ema_fast=50,
        ema_slow=200,
        atr_n=14,
        nochop_filter=True,
        nochop_ema=20,
        nochop_lookback=20,
        fee_bps=0.0,
        slippage_bps=0.0,
        long_only=True,
    )

    variants = [
        (
            "A_fast_htf_slope",
            dict(htf_slope_filter=True, htf_slope_ema=50, htf_slope_window=4, htf_slope_eps=0.0, htf_slope_eps_atr_mult=None),
        ),
        (
            "B_thr_htf_slope",
            dict(htf_slope_filter=True, htf_slope_ema=50, htf_slope_window=4, htf_slope_eps=0.0, htf_slope_eps_atr_mult=0.02),
        ),
        (
            "C_base_slope",
            dict(base_slope_filter=True, base_slope_ema=200, base_slope_window=20, base_slope_eps=0.0),
        ),
    ]

    rows = []
    for nm, extra in variants:
        rows.append(
            run_variant(
                name=nm,
                px_2022=px_2022,
                px_2325=px_2325,
                px_2026=px_2026,
                htf_2022=bars15_2022,
                htf_2325=bars15_2325,
                htf_2026=bars15_2026,
                k=k,
                m=m,
                common_kwargs=common,
                extra_kwargs=extra,
            )
        )

    df = pd.DataFrame(rows).sort_values(["score"], ascending=False).reset_index(drop=True)
    with pd.option_context("display.max_columns", 50, "display.width", 200):
        print(df.to_string(index=False))

    best = df.iloc[0].to_dict()
    print("\nBEST:")
    print(best)


if __name__ == "__main__":
    main()

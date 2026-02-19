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


def main() -> None:
    out_dir = Path("reports/trend_variants/position_sizing")
    out_dir.mkdir(parents=True, exist_ok=True)

    root_5m_ohlc = Path("data/dukascopy_5m_ohlc")
    root_15m_ohlc = Path("data/dukascopy_15m_ohlc")

    # Periods
    p1_start, p1_end = dt.date(2022, 1, 1), dt.date(2022, 12, 31)
    p2_start, p2_end = dt.date(2023, 1, 1), dt.date(2025, 12, 31)
    p3_start, p3_end = dt.date(2026, 1, 1), dt.date.today()

    xau5_2022 = load_ohlc_from_daily_parquet(symbol="XAUUSD", start=p1_start, end=p1_end, root=root_5m_ohlc)
    xau5_2325 = load_ohlc_from_daily_parquet(symbol="XAUUSD", start=p2_start, end=p2_end, root=root_5m_ohlc)
    xau5_2026 = load_ohlc_from_daily_parquet(symbol="XAUUSD", start=p3_start, end=p3_end, root=root_5m_ohlc)

    xau15_2022 = load_ohlc_from_daily_parquet(symbol="XAUUSD", start=p1_start, end=p1_end, root=root_15m_ohlc)
    xau15_2325 = load_ohlc_from_daily_parquet(symbol="XAUUSD", start=p2_start, end=p2_end, root=root_15m_ohlc)
    xau15_2026 = load_ohlc_from_daily_parquet(symbol="XAUUSD", start=p3_start, end=p3_end, root=root_15m_ohlc)

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

    # New baseline = combined corr OR best
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
        beta_confirm=False,
        retest_entry=False,
        exit_ema_filter=False,
        fee_bps=0.0,
        slippage_bps=0.0,
        long_only=True,
        corr_filter=True,
        corr_window=50,
        corr_min_abs=0.25,
        corr_flip_lookback=100,
        corr_max_flips=1,
        corr2_window=75,
        corr2_min_abs=0.20,
        corr2_flip_lookback=50,
        corr2_max_flips=4,
        corr_logic="or",
    )

    levels = (0.0, 0.5, 1.0, 1.5)

    variants = []

    # Vol-target variants
    for tgt in [0.08, 0.10, 0.12, 0.15]:
        for win in [50, 100, 150, 200]:
            variants.append(
                dict(
                    name=f"vol_t{tgt}_w{win}",
                    sizing_mode="vol_target",
                    vol_target=tgt,
                    vol_window=win,
                    confirm_size_one=1.0,
                    confirm_size_both=1.5,
                )
            )

    # Confirmation sizing variants
    for one in [0.5, 1.0]:
        for both in [1.0, 1.5]:
            variants.append(
                dict(
                    name=f"confirm_one{one}_both{both}",
                    sizing_mode="confirm",
                    vol_target=0.10,
                    vol_window=100,
                    confirm_size_one=one,
                    confirm_size_both=both,
                )
            )

    rows = []

    for v in variants:
        params = dict(base_params)
        params.update(
            sizing_mode=v["sizing_mode"],
            sizing_levels=levels,
            vol_target=float(v["vol_target"]),
            vol_window=int(v["vol_window"]),
            confirm_size_one=float(v["confirm_size_one"]),
            confirm_size_both=float(v["confirm_size_both"]),
        )

        bt1, _, _ = trend_following_ma_crossover_htf_confirm(px_2022, htf_bars=xau15_2022, corr_close=cc1_2022, corr_close2=cc2_2022, **params)
        bt2, _, _ = trend_following_ma_crossover_htf_confirm(px_2325, htf_bars=xau15_2325, corr_close=cc1_2325, corr_close2=cc2_2325, **params)
        bt3, _, _ = trend_following_ma_crossover_htf_confirm(px_2026, htf_bars=xau15_2026, corr_close=cc1_2026, corr_close2=cc2_2026, **params)

        r_all, e_all = concat_returns(bt1, bt2, bt3)
        summ = performance_summary(r_all, e_all, freq="5MIN")

        # Score: emphasize CAGR; penalize DD; include Sharpe lightly.
        score = float(100.0 * summ.cagr) + 0.3 * float(summ.sharpe) + 0.7 * float(summ.max_drawdown * 100.0)

        rows.append(
            {
                "name": v["name"],
                "score": score,
                "final_equity": float(e_all.iloc[-1]) if len(e_all) else float("nan"),
                **asdict(summ),
                **v,
            }
        )

    df = pd.DataFrame(rows).sort_values(["score"], ascending=False).reset_index(drop=True)
    out_csv = out_dir / "sizing_search.csv"
    df.to_csv(out_csv, index=False)

    with pd.option_context("display.max_columns", 80, "display.width", 220):
        print(df.head(20).to_string(index=False))

    best = df.iloc[0].to_dict()
    print("\nBEST:")
    print(best)


if __name__ == "__main__":
    main()

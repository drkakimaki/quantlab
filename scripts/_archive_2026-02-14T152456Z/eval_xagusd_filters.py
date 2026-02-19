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


def trade_returns_from_position(bt: pd.DataFrame) -> pd.Series:
    if bt is None or bt.empty:
        return pd.Series(dtype=float)
    pos = bt["position"].fillna(0.0).astype(float)
    r = bt["returns_net"].fillna(0.0).astype(float)

    in_pos = pos != 0.0
    if not bool(in_pos.any()):
        return pd.Series(dtype=float)

    prev = in_pos.shift(1, fill_value=False)
    entry = in_pos & (~prev)
    tid = entry.cumsum()

    df = pd.DataFrame({"tid": tid, "in_pos": in_pos, "r": r})
    df = df[df["in_pos"]]
    if df.empty:
        return pd.Series(dtype=float)

    log_r = (1.0 + df["r"]).clip(lower=1e-12)
    import numpy as np

    df["log1p"] = np.log(log_r.astype(float))
    tr_log = df.groupby("tid")["log1p"].sum()
    return np.expm1(tr_log).astype(float)


def win_rate(bt: pd.DataFrame) -> float:
    tr = trade_returns_from_position(bt)
    if tr.empty:
        return float("nan")
    return float(100.0 * (tr > 0.0).mean())


def concat_bt(*bts: pd.DataFrame) -> pd.DataFrame:
    parts = [bt for bt in bts if bt is not None and not bt.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts).sort_index()
    return out


def main() -> None:
    out_dir = Path("reports/trend_variants/xagusd_filters")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data roots
    root_5m_ohlc = Path("data/dukascopy_5m_ohlc")
    root_15m_ohlc = Path("data/dukascopy_15m_ohlc")

    # Periods
    p1_start, p1_end = dt.date(2022, 1, 1), dt.date(2022, 12, 31)
    p2_start, p2_end = dt.date(2023, 1, 1), dt.date(2025, 12, 31)
    p3_start, p3_end = dt.date(2026, 1, 1), dt.date.today()

    # Load base (XAUUSD) and beta (XAGUSD)
    xau5_2022 = load_ohlc_from_daily_parquet(symbol="XAUUSD", start=p1_start, end=p1_end, root=root_5m_ohlc)
    xau5_2325 = load_ohlc_from_daily_parquet(symbol="XAUUSD", start=p2_start, end=p2_end, root=root_5m_ohlc)
    xau5_2026 = load_ohlc_from_daily_parquet(symbol="XAUUSD", start=p3_start, end=p3_end, root=root_5m_ohlc)

    xag5_2022 = load_ohlc_from_daily_parquet(symbol="XAGUSD", start=p1_start, end=p1_end, root=root_5m_ohlc)
    xag5_2325 = load_ohlc_from_daily_parquet(symbol="XAGUSD", start=p2_start, end=p2_end, root=root_5m_ohlc)
    xag5_2026 = load_ohlc_from_daily_parquet(symbol="XAGUSD", start=p3_start, end=p3_end, root=root_5m_ohlc)

    xau15_2022 = load_ohlc_from_daily_parquet(symbol="XAUUSD", start=p1_start, end=p1_end, root=root_15m_ohlc)
    xau15_2325 = load_ohlc_from_daily_parquet(symbol="XAUUSD", start=p2_start, end=p2_end, root=root_15m_ohlc)
    xau15_2026 = load_ohlc_from_daily_parquet(symbol="XAUUSD", start=p3_start, end=p3_end, root=root_15m_ohlc)

    px_2022 = xau5_2022["close"].astype(float)
    px_2325 = xau5_2325["close"].astype(float)
    px_2026 = xau5_2026["close"].astype(float)

    bc_2022 = xag5_2022["close"].astype(float)
    bc_2325 = xag5_2325["close"].astype(float)
    bc_2026 = xag5_2026["close"].astype(float)

    # Baseline v12 params (fixed)
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
        retest_entry=False,
        exit_ema_filter=False,
        fee_bps=0.0,
        slippage_bps=0.0,
        long_only=True,
        beta_confirm=True,
    )

    # Define up to 25 variants of beta usage
    variants = []

    # 1) The tuned one we found
    variants.append(
        dict(name="v00_tuned", beta_mode="ema_or_reclaim", beta_ema_fast=10, beta_ema_slow=200, beta_reclaim_lookback=5, beta_mom_lookback=6, beta_breakout_lookback=12)
    )

    # 2-? Small grid around it
    for mode in ["ema_only", "reclaim_only", "ema_and_reclaim", "ema_or_reclaim", "ema_or_momentum", "ema_and_momentum", "breakout_or_ema"]:
        for ema_fast in [10, 15, 20]:
            for ema_slow in [150, 200, 250]:
                for reclaim in [0, 2, 5]:
                    variants.append(
                        dict(
                            name=f"{mode}_f{ema_fast}_s{ema_slow}_r{reclaim}",
                            beta_mode=mode,
                            beta_ema_fast=ema_fast,
                            beta_ema_slow=ema_slow,
                            beta_reclaim_lookback=reclaim,
                            beta_mom_lookback=6,
                            beta_breakout_lookback=12,
                        )
                    )

    # keep only first 25
    variants = variants[:25]

    rows = []

    for v in variants:
        params = dict(base_params)
        params.update(
            beta_mode=v["beta_mode"],
            beta_ema_fast=int(v["beta_ema_fast"]),
            beta_ema_slow=int(v["beta_ema_slow"]),
            beta_reclaim_lookback=int(v["beta_reclaim_lookback"]),
            beta_mom_lookback=int(v["beta_mom_lookback"]),
            beta_breakout_lookback=int(v["beta_breakout_lookback"]),
        )

        bt1, _, _ = trend_following_ma_crossover_htf_confirm(px_2022, htf_bars=xau15_2022, beta_close=bc_2022, **params)
        bt2, _, _ = trend_following_ma_crossover_htf_confirm(px_2325, htf_bars=xau15_2325, beta_close=bc_2325, **params)
        bt3, _, _ = trend_following_ma_crossover_htf_confirm(px_2026, htf_bars=xau15_2026, beta_close=bc_2026, **params)

        bt_all = concat_bt(bt1, bt2, bt3)
        r_all = bt_all["returns_net"] if not bt_all.empty else pd.Series(dtype=float)
        e_all = (1.0 + r_all.fillna(0.0)).cumprod() if len(r_all) else pd.Series(dtype=float)

        summ = performance_summary(r_all, e_all, freq="5MIN")
        wr = win_rate(bt_all)

        # Score emphasizing win-rate and CAGR, with mild DD penalty
        score = float(100.0 * summ.cagr) + 0.5 * float(wr) + 0.5 * float(summ.max_drawdown * 100.0)

        rows.append(
            {
                "name": v["name"],
                "score": score,
                "win_rate": wr,
                "final_equity": float(e_all.iloc[-1]) if len(e_all) else float("nan"),
                **asdict(summ),
                **{k: v[k] for k in v if k != "name"},
            }
        )

    df = pd.DataFrame(rows).sort_values(["score"], ascending=False).reset_index(drop=True)
    out_csv = out_dir / "xagusd_filter_variants.csv"
    df.to_csv(out_csv, index=False)

    with pd.option_context("display.max_columns", 80, "display.width", 220):
        print(df.head(15).to_string(index=False))

    best = df.iloc[0].to_dict()
    print("\nBEST:")
    print(best)


if __name__ == "__main__":
    main()

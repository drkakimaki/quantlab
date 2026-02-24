from __future__ import annotations

"""Resampling utilities (Polars-based).

This module is allowed to depend on Polars. Keep it out of the import path for
normal backtests (rnd/webui), which should only need parquet loaders.

Download/resample CLI: :mod:`quantlab.data.download`.
"""

import datetime as dt
from pathlib import Path

import pandas as pd
import polars as pl


def load_dukascopy_1s_mid(
    *,
    symbol: str,
    start: dt.date,
    end: dt.date,
    root: str | Path = Path("data/dukascopy_1s"),
) -> pd.Series:
    """Load mid quotes from daily 1s Dukascopy parquet files as a pandas Series."""
    root = Path(root)
    paths: list[str] = []
    cur = start
    one = dt.timedelta(days=1)
    while cur <= end:
        p = root / symbol / str(cur.year) / f"{cur.isoformat()}.parquet"
        if p.exists():
            paths.append(str(p))
        cur += one

    if not paths:
        raise FileNotFoundError(f"No parquet files found for {symbol} in {root} between {start} and {end}")

    lf = pl.scan_parquet(paths).select(["ts", "mid"]).sort("ts")
    df = lf.collect()

    s = df.to_pandas().set_index("ts")["mid"].sort_index()
    s.name = symbol
    return s


def load_dukascopy_mid_resample_last(
    *,
    symbol: str,
    start: dt.date,
    end: dt.date,
    rule: str = "5m",
    root: str | Path = Path("data/dukascopy_1s"),
) -> pd.Series:
    """Load Dukascopy 1s mid quotes and resample *in Polars* to reduce memory."""
    root = Path(root)
    paths: list[str] = []
    cur = start
    one = dt.timedelta(days=1)
    while cur <= end:
        p = root / symbol / str(cur.year) / f"{cur.isoformat()}.parquet"
        if p.exists():
            paths.append(str(p))
        cur += one

    if not paths:
        raise FileNotFoundError(f"No parquet files found for {symbol} in {root} between {start} and {end}")

    lf = (
        pl.scan_parquet(paths)
        .select(["ts", "mid"])
        .sort("ts")
        .group_by_dynamic(index_column="ts", every=rule, closed="left", label="right")
        .agg(pl.col("mid").last().alias("mid"))
        .drop_nulls()
        .sort("ts")
    )

    df = lf.collect(streaming=True)
    s = df.to_pandas().set_index("ts")["mid"].sort_index()
    s.name = symbol
    return s


def resample_dukascopy_1s_to_bars(
    *,
    symbol: str,
    start: dt.date,
    end: dt.date,
    rule: str = "5m",
    root_in: str | Path = Path("data/dukascopy_1s"),
    root_out: str | Path = Path("data/dukascopy_5m"),
    overwrite: bool = False,
) -> int:
    """Resample daily Dukascopy 1s parquet files into daily close-only bars."""
    root_in = Path(root_in)
    root_out = Path(root_out)

    n_written = 0
    cur = start
    one = dt.timedelta(days=1)

    while cur <= end:
        in_path = root_in / symbol / str(cur.year) / f"{cur.isoformat()}.parquet"
        out_path = root_out / symbol / str(cur.year) / f"{cur.isoformat()}.parquet"

        if not in_path.exists():
            cur += one
            continue

        if out_path.exists() and not overwrite:
            cur += one
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)

        df = pl.read_parquet(str(in_path)).select(["ts", "bid", "ask", "mid", "spread"]).sort("ts")

        bars = (
            df.group_by_dynamic(index_column="ts", every=rule, closed="left", label="right")
            .agg(
                [
                    pl.col("bid").last().alias("bid"),
                    pl.col("ask").last().alias("ask"),
                    pl.col("mid").last().alias("mid"),
                    pl.col("spread").last().alias("spread"),
                ]
            )
            .drop_nulls()
            .sort("ts")
        )

        bars.write_parquet(str(out_path))
        n_written += 1
        cur += one

    return n_written


def resample_dukascopy_1s_to_ohlc(
    *,
    symbol: str,
    start: dt.date,
    end: dt.date,
    rule: str = "5m",
    root_in: str | Path = Path("data/dukascopy_1s"),
    root_out: str | Path = Path("data/dukascopy_5m_ohlc"),
    price_col: str = "mid",
    overwrite: bool = False,
) -> int:
    """Resample daily Dukascopy 1s parquet to daily OHLC bars."""
    root_in = Path(root_in)
    root_out = Path(root_out)

    n_written = 0
    cur = start
    one = dt.timedelta(days=1)

    while cur <= end:
        in_path = root_in / symbol / str(cur.year) / f"{cur.isoformat()}.parquet"
        out_path = root_out / symbol / str(cur.year) / f"{cur.isoformat()}.parquet"

        if not in_path.exists():
            cur += one
            continue

        if out_path.exists() and not overwrite:
            cur += one
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)

        df = pl.read_parquet(str(in_path)).select(["ts", price_col]).sort("ts")

        bars = (
            df.group_by_dynamic(index_column="ts", every=rule, closed="left", label="right")
            .agg(
                [
                    pl.col(price_col).first().alias("open"),
                    pl.col(price_col).max().alias("high"),
                    pl.col(price_col).min().alias("low"),
                    pl.col(price_col).last().alias("close"),
                ]
            )
            .drop_nulls()
            .sort("ts")
        )

        bars.write_parquet(str(out_path))
        n_written += 1
        cur += one

    return n_written

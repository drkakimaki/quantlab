from __future__ import annotations

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
    """Load mid quotes from daily 1s Dukascopy parquet files as a pandas Series.

    Expects files:
      {root}/{symbol}/{YYYY}/{YYYY-MM-DD}.parquet

    Returns a pandas Series indexed by UTC timestamps (tz-aware if present in parquet).
    """
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

    # Polars is fast for many files.
    lf = pl.scan_parquet(paths).select(["ts", "mid"]).sort("ts")
    df = lf.collect()

    s = df.to_pandas().set_index("ts")["mid"].sort_index()
    s.name = symbol
    return s


def resample_last(prices: pd.Series, rule: str) -> pd.Series:
    """Resample a price series to a lower frequency using last-observation in each bin."""
    px = prices.dropna().copy()
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    out = px.resample(rule).last().dropna()
    out.name = prices.name
    return out


def load_dukascopy_mid_resample_last(
    *,
    symbol: str,
    start: dt.date,
    end: dt.date,
    rule: str = "5m",
    root: str | Path = Path("data/dukascopy_1s"),
) -> pd.Series:
    """Load Dukascopy 1s mid quotes and resample *in Polars* to reduce memory.

    This avoids materializing the full 1-second series in pandas for long ranges.

    Parameters
    ----------
    rule:
        Polars duration string (e.g. "5m", "1h").
    """
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
        .group_by_dynamic(
            index_column="ts",
            every=rule,
            closed="left",
            label="right",
        )
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
    """Resample daily Dukascopy 1s parquet files into daily lower-frequency parquet files.

    This produces *close-only* bars (last observation in each bucket) for bid/ask/mid/spread.

    Input files:
      {root_in}/{symbol}/{YYYY}/{YYYY-MM-DD}.parquet

    Output files:
      {root_out}/{symbol}/{YYYY}/{YYYY-MM-DD}.parquet

    Output schema matches input columns but sampled as last-observation in each bucket
    (ts is the bucket label, using label='right').
    """
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

        # Each input parquet is a single day; eager read is fine and keeps logic simple.
        df = pl.read_parquet(str(in_path)).select(["ts", "bid", "ask", "mid", "spread"]).sort("ts")

        bars = (
            df.group_by_dynamic(
                index_column="ts",
                every=rule,
                closed="left",
                label="right",
            )
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
    """Resample daily Dukascopy 1s parquet to daily OHLC bars.

    Input files:
      {root_in}/{symbol}/{YYYY}/{YYYY-MM-DD}.parquet

    Output files:
      {root_out}/{symbol}/{YYYY}/{YYYY-MM-DD}.parquet

    Output schema:
      ts, open, high, low, close

    Notes
    -----
    - OHLC is computed from `price_col` (default: mid).
    - `ts` is the bucket label (label='right') to align with other resamples.
    """
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
            df.group_by_dynamic(
                index_column="ts",
                every=rule,
                closed="left",
                label="right",
            )
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


def load_dukascopy_ohlc(
    *,
    symbol: str,
    start: dt.date,
    end: dt.date,
    root: str | Path = Path("data/dukascopy_5m_ohlc"),
) -> pd.DataFrame:
    """Load pre-resampled OHLC bars from daily parquet files.

    Expects files:
      {root}/{symbol}/{YYYY}/{YYYY-MM-DD}.parquet

    Schema:
      ts, open, high, low, close

    Returns:
        DataFrame indexed by timestamp with OHLC columns.
    """
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
        raise FileNotFoundError(f"No OHLC parquet files found for {symbol} in {root} between {start} and {end}")

    df = (
        pl.scan_parquet(paths)
        .select(["ts", "open", "high", "low", "close"])
        .sort("ts")
        .collect(engine="streaming")
    )
    out = df.to_pandas().set_index("ts").sort_index()
    out.index = pd.to_datetime(out.index)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    return out

from __future__ import annotations

# (dataclass not used)

import numpy as np
import pandas as pd


def _to_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, (pd.DataFrame,)):
        raise TypeError("Expected 1D series-like, got DataFrame")
    return pd.Series(x)


def daily_returns_from_equity(
    equity,
    *,
    tz: str | None = "UTC",
) -> pd.Series:
    """Compute daily close-to-close returns from an intraday equity curve.

    This is the canonical return series for Sharpe in Quantlab.

    Notes
    -----
    - Daily grouping is done by resampling to calendar days ("1D") on the equity
      index and taking the last value as the daily close.
    - If you need NY-5pm trading-day boundaries, implement that explicitly and
      keep this function as the single choke point.
    """
    e = _to_series(equity).dropna().astype(float)
    if e.empty:
        return pd.Series(dtype=float)

    if not isinstance(e.index, (pd.DatetimeIndex,)):
        raise TypeError("equity must be indexed by timestamps (DatetimeIndex)")

    # Normalize timestamps to a known timezone for resampling.
    idx = pd.to_datetime(e.index)
    if tz is not None:
        if idx.tz is None:
            idx = idx.tz_localize(tz)
        else:
            idx = idx.tz_convert(tz)
    e = e.copy()
    e.index = idx
    e = e.sort_index()

    daily_close = e.resample("1D").last().dropna()
    return daily_close.pct_change().dropna()


def sharpe(
    equity,
    *,
    annual_days: int = 252,
    tz: str | None = "UTC",
) -> float:
    """Annualized Sharpe ratio computed on **daily** returns.

    Canonical definition in this codebase:
      1) Aggregate equity to daily closes
      2) Compute daily returns (close-to-close)
      3) Sharpe = sqrt(252) * mean(daily_ret) / std(daily_ret)

    This intentionally avoids intraday annualization debates.
    """
    r = daily_returns_from_equity(equity, tz=tz)
    if r.empty:
        return float("nan")

    mu = float(r.mean())
    sig = float(r.std(ddof=1))
    if sig == 0.0 or not np.isfinite(sig):
        return float("nan")

    return float(np.sqrt(float(annual_days)) * mu / sig)


def max_drawdown(equity) -> float:
    e = _to_series(equity).dropna()
    if e.empty:
        return float("nan")
    peak = e.cummax()
    dd = e / peak - 1.0
    return float(dd.min())


# (removed) CAGR / volatility / performance_summary wrapper — kept metrics minimal.


# --- Trade-level metrics (canonical definition) ---
# Trade = contiguous segment where position != 0.


def trade_returns_from_position(
    bt: pd.DataFrame,
    *,
    pos_col: str = "position",
    returns_col: str = "returns_net",
) -> pd.Series:
    """Compute compounded return per trade (decimal, not %).

    Canonical trade definition:
      trade = contiguous segment where position != 0.

    Return per trade is computed by compounding per-bar returns_net:
      trade_return = prod(1 + r_t) - 1
    """
    if bt is None or len(bt) == 0:
        return pd.Series(dtype=float)
    if pos_col not in bt.columns or returns_col not in bt.columns:
        return pd.Series(dtype=float)

    pos = bt[pos_col].fillna(0.0).astype(float)
    r = bt[returns_col].fillna(0.0).astype(float)

    in_pos = pos != 0.0
    if not bool(in_pos.any()):
        return pd.Series(dtype=float)

    prev_in_pos = in_pos.shift(1, fill_value=False)
    entry = in_pos & (~prev_in_pos)
    trade_id = entry.cumsum()

    df = pd.DataFrame({"trade_id": trade_id, "in_pos": in_pos, "r": r})
    df = df[df["in_pos"]].copy()
    if df.empty:
        return pd.Series(dtype=float)

    # trade_return = exp(sum(log1p(r))) - 1
    log_r = (1.0 + df["r"]).clip(lower=1e-12)
    df["log1p_r"] = np.log(log_r)
    trade_log = df.groupby("trade_id")["log1p_r"].sum()
    return np.expm1(trade_log).astype(float)


def n_trades_from_position(bt: pd.DataFrame, *, pos_col: str = "position") -> int:
    if bt is None or len(bt) == 0 or pos_col not in bt.columns:
        return 0
    pos = bt[pos_col].fillna(0.0).astype(float)
    in_pos = pos != 0.0
    prev_in_pos = in_pos.shift(1, fill_value=False)
    entry = in_pos & (~prev_in_pos)
    return int(entry.sum())


def win_rate_from_position(
    bt: pd.DataFrame,
    *,
    pos_col: str = "position",
    returns_col: str = "returns_net",
) -> float:
    tr = trade_returns_from_position(bt, pos_col=pos_col, returns_col=returns_col)
    if tr.empty:
        return float("nan")
    return float(100.0 * (tr > 0.0).mean())


def profit_factor_from_position(
    bt: pd.DataFrame,
    *,
    pos_col: str = "position",
    returns_col: str = "returns_net",
) -> float:
    """Profit factor on per-trade compounded returns."""
    tr = trade_returns_from_position(bt, pos_col=pos_col, returns_col=returns_col)
    if tr.empty:
        return float("nan")

    gp = float(tr[tr > 0.0].sum())
    gl = float((-tr[tr < 0.0]).sum())
    if gl <= 0.0:
        return float("nan")
    return gp / gl


def avg_win_loss_from_position(
    bt: pd.DataFrame,
    *,
    pos_col: str = "position",
    returns_col: str = "returns_net",
) -> tuple[float, float]:
    """Return (avg_win%, avg_loss%) as percentages. avg_loss% is negative."""
    tr = trade_returns_from_position(bt, pos_col=pos_col, returns_col=returns_col)
    if tr.empty:
        return float("nan"), float("nan")

    wins = tr[tr > 0.0]
    losses = tr[tr < 0.0]

    avg_win = float(wins.mean() * 100.0) if len(wins) else float("nan")
    avg_loss = float(losses.mean() * 100.0) if len(losses) else float("nan")
    return avg_win, avg_loss

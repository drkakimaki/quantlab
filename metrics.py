from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _to_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, (pd.DataFrame,)):
        raise TypeError("Expected 1D series-like, got DataFrame")
    return pd.Series(x)


def annualization_factor(freq: str) -> float:
    """Return annualization factor for Sharpe/vol etc.

    Supported freq strings:
    - 'S'            : 1-second
    - 'T' / 'MIN'    : 1-minute
    - '5MIN'         : 5-minute
    - '15MIN'        : 15-minute
    - 'H'            : hourly
    - 'D'            : daily (calendar days)
    - 'B'            : business days
    - 'W'            : weekly
    - 'M'            : monthly

    Notes
    -----
    - For intraday, we annualize using **252 trading days/year** (not 365).
      This matches common finance convention and avoids overstating Sharpe/vol.
    """
    freq = freq.upper()

    # Common convention for FX-like markets (24h, 5d/week): ~260 trading days/year.
    TRADING_DAYS = 260
    table = {
        # Intraday: assume 24h * trading days (good approximation for FX/CFDs; weekends largely absent).
        "S": 60 * 60 * 24 * TRADING_DAYS,
        "T": 60 * 24 * TRADING_DAYS,
        "MIN": 60 * 24 * TRADING_DAYS,
        "5MIN": (60 // 5) * 24 * TRADING_DAYS,
        "15MIN": (60 // 15) * 24 * TRADING_DAYS,
        "H": 24 * TRADING_DAYS,

        # Daily calendars vs business days
        "D": 365,
        "B": 252,
        "W": 52,
        "M": 12,
    }
    if freq not in table:
        raise KeyError(f"Unknown freq {freq!r}. Known: {sorted(table)}")
    return table[freq]


def sharpe(returns, freq: str = "B") -> float:
    r = _to_series(returns).dropna()
    if r.empty:
        return float("nan")
    af = annualization_factor(freq)
    mu = r.mean()
    sig = r.std(ddof=1)
    if sig == 0:
        return float("nan")
    return float(np.sqrt(af) * mu / sig)


def volatility(returns, freq: str = "B") -> float:
    r = _to_series(returns).dropna()
    if r.empty:
        return float("nan")
    af = annualization_factor(freq)
    return float(np.sqrt(af) * r.std(ddof=1))


def max_drawdown(equity) -> float:
    e = _to_series(equity).dropna()
    if e.empty:
        return float("nan")
    peak = e.cummax()
    dd = e / peak - 1.0
    return float(dd.min())


def cagr(equity, freq: str = "B") -> float:
    e = _to_series(equity).dropna()
    if len(e) < 2:
        return float("nan")
    af = annualization_factor(freq)
    n_periods = len(e) - 1
    years = n_periods / af
    if years <= 0:
        return float("nan")
    return float(e.iloc[-1] ** (1.0 / years) - 1.0)


@dataclass(frozen=True)
class PerformanceSummary:
    cagr: float
    vol: float
    sharpe: float
    max_drawdown: float


def performance_summary(returns, equity, freq: str = "B") -> PerformanceSummary:
    return PerformanceSummary(
        cagr=cagr(equity, freq=freq),
        vol=volatility(returns, freq=freq),
        sharpe=sharpe(returns, freq=freq),
        max_drawdown=max_drawdown(equity),
    )


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

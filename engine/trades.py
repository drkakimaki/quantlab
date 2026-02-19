from __future__ import annotations

import numpy as np
import pandas as pd


def extract_trade_log(
    bt: pd.DataFrame,
    *,
    pos_col: str = "position",
    returns_col: str = "returns_net",
    equity_col: str = "equity",
) -> pd.DataFrame:
    """Extract a canonical trade log from a backtest dataframe.

    Canonical trade definition:
      trade = contiguous segment where position != 0.

    Trade return definition (Option A):
      trade_return = prod(1 + returns_net[t]) - 1 over bars while in trade

    Returns
    -------
    DataFrame with one row per trade:
      trade_id, entry_time, exit_time, bars,
      entry_equity, exit_equity, pnl,
      trade_return

    Notes
    -----
    - If a trade is still open at the end, exit_time is the last index and open=True.
    """
    if bt is None or len(bt) == 0:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "entry_time",
                "exit_time",
                "bars",
                "entry_equity",
                "exit_equity",
                "pnl",
                "trade_return",
                "open",
            ]
        )

    if pos_col not in bt.columns:
        raise KeyError(f"bt missing {pos_col!r}")
    if returns_col not in bt.columns:
        raise KeyError(f"bt missing {returns_col!r}")
    if equity_col not in bt.columns:
        raise KeyError(f"bt missing {equity_col!r}")

    bt = bt.copy()
    bt.index = pd.to_datetime(bt.index)

    pos = bt[pos_col].fillna(0.0).astype(float)
    r = bt[returns_col].fillna(0.0).astype(float)
    eq = bt[equity_col].astype(float)

    in_pos = pos != 0.0
    prev_in_pos = in_pos.shift(1, fill_value=False)
    entry = in_pos & (~prev_in_pos)

    if not bool(in_pos.any()):
        return pd.DataFrame(
            columns=[
                "trade_id",
                "entry_time",
                "exit_time",
                "bars",
                "entry_equity",
                "exit_equity",
                "pnl",
                "trade_return",
                "open",
            ]
        )

    trade_id = entry.cumsum().astype(int)

    df = pd.DataFrame({"trade_id": trade_id, "in_pos": in_pos, "r": r, "eq": eq}, index=bt.index)

    # Only keep in-trade bars for return compounding
    df_in = df[df["in_pos"]].copy()

    # Compounded return per trade via log1p
    log_r = (1.0 + df_in["r"]).clip(lower=1e-12)
    df_in["log1p_r"] = np.log(log_r)
    trade_log = df_in.groupby("trade_id")["log1p_r"].sum()
    trade_ret = np.expm1(trade_log).astype(float)

    # Entry/exit times
    entry_time = df.index[entry].to_series(index=df.index[entry])
    entry_time.index = trade_id[entry].values
    entry_time = entry_time.sort_index()

    # Exit = first bar where in_pos goes False after being True
    exit_flag = (~in_pos) & prev_in_pos
    exit_time = df.index[exit_flag].to_series(index=df.index[exit_flag])
    exit_time.index = trade_id[exit_flag].values
    exit_time = exit_time.sort_index()

    # Build rows
    rows = []
    for tid in sorted(trade_ret.index.astype(int).tolist()):
        et = pd.Timestamp(entry_time.loc[tid])
        if tid in exit_time.index:
            xt = pd.Timestamp(exit_time.loc[tid])
            open_trade = False
        else:
            xt = pd.Timestamp(df.index[-1])
            open_trade = True

        # bars in trade (count in_pos bars)
        bars = int(df_in[df_in["trade_id"] == tid].shape[0])

        # equity at entry/exit
        entry_eq = float(eq.loc[et])
        exit_eq = float(eq.loc[xt])
        pnl = exit_eq - entry_eq

        rows.append(
            {
                "trade_id": int(tid),
                "entry_time": et,
                "exit_time": xt,
                "bars": bars,
                "entry_equity": entry_eq,
                "exit_equity": exit_eq,
                "pnl": pnl,
                "trade_return": float(trade_ret.loc[tid]),
                "open": bool(open_trade),
            }
        )

    return pd.DataFrame(rows)

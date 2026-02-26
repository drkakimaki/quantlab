from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from quantlab.strategies.base import BacktestConfig


def extract_executions(
    df: pd.DataFrame,
    prices: pd.Series,
    config: "BacktestConfig",
) -> pd.DataFrame:
    """Extract execution log (one row per position change).

    Fields:
        time: Execution timestamp
        prev_size: Position size before execution
        new_size: Position size after execution
        d_size: Position size change (= new - prev)
        prev_contract_units: Contract units before execution
        new_contract_units: Contract units after execution
        d_contract_units: Contract unit change
        fill_price: Execution price (bar close)
        d_lots: Lot change
        notional: abs(d_contract_units) * fill_price
        fee_per_lot: Fee per lot
        spread_per_lot: Spread per lot
        costs: Total cost
        reason: Optional reason (later from gates/risk modules)

    Args:
        df: Backtest dataframe with 'position' column
        prices: Price series for fill prices
        config: BacktestConfig with cost parameters

    Returns:
        DataFrame with execution records.
    """
    pos = df["position"]
    lots = df["lots"]
    contract_units = df.get("contract_units")

    # Find position changes
    d_pos = pos.diff()
    change_mask = d_pos.abs() > 0

    columns = [
        "time",
        "prev_size", "new_size", "d_size",
        "prev_contract_units", "new_contract_units", "d_contract_units",
        "fill_price", "d_lots", "notional",
        "fee_per_lot", "spread_per_lot",
        "costs", "reason",
    ]

    if not change_mask.any():
        return pd.DataFrame(columns=columns)

    cost_per_lot = config.fee_per_lot + config.spread_per_lot

    records = []
    for t in pos.index[change_mask]:
        idx = pos.index.get_loc(t)
        if idx == 0:
            continue

        prev_size = float(pos.iloc[idx - 1])
        new_size = float(pos.iloc[idx])
        d_size = new_size - prev_size

        if contract_units is not None:
            prev_cu = float(contract_units.iloc[idx - 1])
            new_cu = float(contract_units.iloc[idx])
            d_cu = new_cu - prev_cu
        else:
            # Backward fallback (shouldn't happen in current engine)
            prev_cu = float(prev_size)
            new_cu = float(new_size)
            d_cu = float(d_size)

        prev_lots = float(lots.iloc[idx - 1])
        new_lots = float(lots.iloc[idx])
        d_lots = new_lots - prev_lots

        fill_price = float(prices.loc[t])
        notional = abs(d_cu) * fill_price

        costs = abs(d_lots) * cost_per_lot

        records.append({
            "time": t,
            "prev_size": prev_size,
            "new_size": new_size,
            "d_size": d_size,
            "prev_contract_units": prev_cu,
            "new_contract_units": new_cu,
            "d_contract_units": d_cu,
            "fill_price": fill_price,
            "d_lots": d_lots,
            "notional": notional,
            "fee_per_lot": config.fee_per_lot,
            "spread_per_lot": config.spread_per_lot,
            "costs": costs,
            "reason": None,  # Later from gates/risk modules
        })

    return pd.DataFrame(records)


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


def duration_bin(bars: pd.Series) -> pd.Categorical:
    """Stable duration buckets (in bars) used across trade analysis + reports."""
    bins = [-np.inf, 1, 3, 6, 12, 24, 48, 96, 192, np.inf]
    labels = [
        "1",
        "2-3",
        "4-6",
        "7-12",
        "13-24",
        "25-48",
        "49-96",
        "97-192",
        "193+",
    ]
    return pd.cut(bars.astype(float), bins=bins, labels=labels)


def profit_factor(pnl: pd.Series) -> float:
    """Profit factor = sum(wins) / sum(|losses|) on *trade* PnL."""
    pnl = pnl.astype(float)
    wins = float(pnl[pnl > 0].sum())
    losses = float((-pnl[pnl < 0]).sum())
    if losses <= 0:
        return float("inf") if wins > 0 else float("nan")
    return wins / losses


def agg_trade_table(trades: pd.DataFrame, key: str) -> pd.DataFrame:
    """Aggregate per-trade ledger into a bucket table.

    Expects columns:
      - win (bool)
      - trade_return (float)
      - pnl_net (float)
      - bars (int)
      - costs_total (float) (optional)

    Returns a stable superset of metrics. Callers can subset columns.
    """
    g = trades.groupby(key, dropna=False)

    out = pd.DataFrame(
        {
            "n_trades": g.size(),
            "win_rate": g["win"].mean(),
            "avg_return": g["trade_return"].mean(),
            "median_return": g["trade_return"].median(),
            "sum_pnl": g["pnl_net"].sum(),
            "avg_pnl": g["pnl_net"].mean(),
            "avg_bars": g["bars"].mean(),
        }
    )

    if "costs_total" in trades.columns:
        out["sum_costs"] = g["costs_total"].sum()
        out["avg_costs"] = g["costs_total"].mean()

    # Profit factor per bucket
    pf = []
    for val, sub in trades.groupby(key, dropna=False):
        pf.append((val, profit_factor(sub["pnl_net"])))
    pf_df = pd.DataFrame(pf, columns=[key, "profit_factor"])
    out = out.reset_index().merge(pf_df, on=key, how="left")

    return out.sort_values("sum_pnl", ascending=False)


def build_trade_ledger(
    bt: pd.DataFrame,
    *,
    pos_col: str = "position",
    returns_col: str = "returns_net",
    equity_col: str = "equity",
    costs_col: str = "costs",
) -> pd.DataFrame:
    """Create an enriched per-trade ledger from a backtest dataframe.

    Returns a DataFrame with one row per trade and columns:
      trade_id, entry_time, exit_time, bars, side,
      entry_equity, exit_equity, pnl_net, trade_return,
      costs_total, win, open
    """
    if bt is None or len(bt) == 0:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "entry_time",
                "exit_time",
                "bars",
                "side",
                "entry_equity",
                "exit_equity",
                "pnl_net",
                "trade_return",
                "costs_total",
                "win",
                "open",
            ]
        )

    bt = bt.copy()
    bt.index = pd.to_datetime(bt.index)
    bt = bt.sort_index()

    for c in (pos_col, returns_col, equity_col):
        if c not in bt.columns:
            raise KeyError(f"bt missing required column {c!r}")

    trades = extract_trade_log(bt, pos_col=pos_col, returns_col=returns_col, equity_col=equity_col)
    if trades.empty:
        return trades

    # Side at entry
    pos = bt[pos_col].fillna(0.0).astype(float)
    entry_pos = pos.reindex(pd.to_datetime(trades["entry_time"]).values).to_numpy(dtype=float)
    side = np.where(entry_pos > 0, "long", np.where(entry_pos < 0, "short", "flat"))

    # Costs per trade
    if costs_col in bt.columns:
        in_pos = pos != 0.0
        prev_in_pos = in_pos.shift(1, fill_value=False)
        entry = in_pos & (~prev_in_pos)
        trade_id = entry.cumsum().astype(int)
        df = pd.DataFrame({"trade_id": trade_id, "in_pos": in_pos, "costs": bt[costs_col].fillna(0.0).astype(float)})
        costs_total = df[df["in_pos"]].groupby("trade_id")["costs"].sum()
        trades["costs_total"] = trades["trade_id"].map(costs_total).fillna(0.0).astype(float)
    else:
        trades["costs_total"] = 0.0

    trades["side"] = side
    trades["pnl_net"] = trades["pnl"].astype(float)
    trades["win"] = trades["pnl_net"] > 0

    keep = [
        "trade_id",
        "entry_time",
        "exit_time",
        "bars",
        "side",
        "entry_equity",
        "exit_equity",
        "pnl_net",
        "trade_return",
        "costs_total",
        "win",
        "open",
    ]
    return trades[keep].sort_values("entry_time").reset_index(drop=True)

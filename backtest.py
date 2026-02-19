from __future__ import annotations

import numpy as np
import pandas as pd


def prices_to_returns(prices: pd.Series | pd.DataFrame, log: bool = False):
    """Convert prices to returns.

    - If DataFrame: returns per column.
    - If Series: returns per series.
    """
    if log:
        return np.log(prices).diff()
    return prices.pct_change()


def positions_from_signal(
    signal: pd.Series | pd.DataFrame,
    lag: int = 1,
    clip: float | None = 1.0,
    normalize_gross: bool = False,
):
    """Turn a (possibly raw) signal into positions.

    lag=1 means trade on next bar (avoids lookahead if signal uses close).
    normalize_gross: if True (DataFrame), scale so sum(abs(w)) = 1 each timestamp.

    NOTE: This is a helper for turning signals into position *sizes*.
    Execution/accounting should be handled by `backtest_positions_account_margin`.
    """
    pos = signal.shift(lag)
    if clip is not None:
        pos = pos.clip(-clip, clip)

    if normalize_gross and isinstance(pos, pd.DataFrame):
        gross = pos.abs().sum(axis=1)
        pos = pos.div(gross.replace(0.0, np.nan), axis=0).fillna(0.0)

    return pos.fillna(0.0)


def backtest_positions_account_margin(
    *,
    prices: pd.Series,
    positions_size: pd.Series,
    initial_capital: float = 1000.0,
    leverage: float | None = 20.0,
    lot_per_size: float = 0.01,
    contract_size_per_lot: float = 100.0,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    fee_per_lot: float = 0.0,
    spread_per_lot: float = 0.0,
    lag: int = 1,
    max_size: float | None = 2.0,
    discrete_sizes: tuple[float, ...] = (0.0, 1.0, 2.0),
    margin_policy: str = "skip_entry",  # skip_entry | allow_negative
) -> pd.DataFrame:
    """Backtest a 1D strategy using a simple *account + margin* model.

    Model
    -----
    - Strategy emits a *size* series (e.g. 0/1/2 or -1/0/+1). We map to:
        lots = size * lot_per_size
        units = lots * contract_size_per_lot
    - PnL per bar: units_prev * (price[t] - price[t-1]) - costs

    Costs
    -----
    Two modes:
    
    1. Absolute per-lot (recommended for real broker modeling):
        costs = abs(d_lots) * (fee_per_lot + spread_per_lot)
    
    2. BPS of notional (legacy):
        costs = abs(d_units) * price * (fee_bps + slippage_bps) * 1e-4

    Margin
    ------
    If leverage is not None:
      required_margin = abs(units * price) / leverage

    On entry (flat -> nonzero) if required_margin > equity:
      - skip_entry: remain flat (units=0)
      - allow_negative: allow (equity can go negative)

    Notes
    -----
    - lag=1 means the size decided at bar t is applied at t+1.
    - Long-only is not enforced here; provide negative sizes if desired.
    """
    if not isinstance(prices, pd.Series) or not isinstance(positions_size, pd.Series):
        raise TypeError("prices and positions_size must be Series")

    px = prices.dropna().astype(float).copy()
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()

    pos = positions_size.reindex(px.index).fillna(0.0).astype(float)
    pos.index = px.index

    if max_size is not None:
        pos = pos.clip(lower=-float(max_size), upper=float(max_size))

    # Engine lag to avoid lookahead.
    if lag:
        pos = pos.shift(int(lag)).fillna(0.0)

    # Snap to discrete sizes.
    ds = tuple(float(x) for x in discrete_sizes)
    if len(ds) == 0:
        raise ValueError("discrete_sizes must be non-empty")

    arr = pos.to_numpy(dtype=float)
    snapped = np.empty_like(arr)
    for i, v in enumerate(arr):
        snapped[i] = min(ds, key=lambda s: abs(v - s))
    pos_disc = pd.Series(snapped, index=pos.index, name="position")

    lots = pos_disc * float(lot_per_size)
    units_target = lots * float(contract_size_per_lot)

    # Cost calculation
    # Prefer absolute per-lot costs if specified, else use BPS
    use_abs_costs = (fee_per_lot > 0 or spread_per_lot > 0)
    cost_per_lot = float(fee_per_lot) + float(spread_per_lot)
    cost_rate_bps = (float(fee_bps) + float(slippage_bps)) * 1e-4

    eq = float(initial_capital)
    eq_series: list[float] = []
    pnl_series: list[float] = []
    costs_series: list[float] = []
    units_series: list[float] = []

    prev_price = float(px.iloc[0])
    prev_units = 0.0
    prev_lots = 0.0

    lev = None if leverage is None else float(leverage)

    for t, price in enumerate(px.to_numpy(dtype=float)):
        desired_units = float(units_target.iloc[t])
        desired_lots = float(lots.iloc[t])

        is_entry = (prev_units == 0.0) and (desired_units != 0.0)
        if lev is not None and is_entry and margin_policy == "skip_entry":
            req_margin = abs(desired_units * price) / lev
            if req_margin > eq:
                desired_units = 0.0
                desired_lots = 0.0

        d_units = desired_units - prev_units
        d_lots = desired_lots - prev_lots
        
        # Calculate costs
        if use_abs_costs:
            costs = abs(d_lots) * cost_per_lot
        else:
            costs = abs(d_units) * price * cost_rate_bps

        d_price = price - prev_price
        pnl = prev_units * d_price - costs

        eq = eq + pnl

        eq_series.append(eq)
        pnl_series.append(pnl)
        costs_series.append(costs)
        units_series.append(desired_units)

        prev_price = price
        prev_units = desired_units
        prev_lots = desired_lots

    eq_s = pd.Series(eq_series, index=px.index, name="equity")
    pnl_s = pd.Series(pnl_series, index=px.index, name="pnl")
    costs_s = pd.Series(costs_series, index=px.index, name="costs")
    units_s = pd.Series(units_series, index=px.index, name="units")

    eq_prev = eq_s.shift(1)
    ret = (pnl_s / eq_prev.replace(0.0, np.nan)).fillna(0.0)

    return pd.DataFrame(
        {
            "position": pos_disc.astype(float),
            "lots": lots.astype(float),
            "units": units_s.astype(float),
            "pnl": pnl_s.astype(float),
            "costs": costs_s.astype(float),
            "returns_net": ret.astype(float),
            "equity": eq_s.astype(float),
        }
    )

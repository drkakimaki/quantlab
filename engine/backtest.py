from __future__ import annotations

import numpy as np
import pandas as pd

# Optional acceleration
try:
    from numba import njit  # type: ignore

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    njit = None  # type: ignore
    _HAVE_NUMBA = False


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


def _bt_loop_python(
    px: np.ndarray,
    units_target: np.ndarray,
    lots_target: np.ndarray,
    *,
    initial_capital: float,
    leverage: float | None,
    fee_per_lot: float,
    spread_per_lot: float,
    margin_policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference loop (pure python). Returns (equity, pnl, costs, contract_units)."""

    cost_per_lot = float(fee_per_lot) + float(spread_per_lot)

    eq = float(initial_capital)

    n = px.shape[0]
    eq_arr = np.empty(n, dtype=np.float64)
    pnl_arr = np.empty(n, dtype=np.float64)
    costs_arr = np.empty(n, dtype=np.float64)
    units_arr = np.empty(n, dtype=np.float64)

    prev_price = float(px[0])
    prev_units = 0.0
    prev_lots = 0.0

    lev = None if leverage is None else float(leverage)

    for t in range(n):
        price = float(px[t])

        desired_units = float(units_target[t])
        desired_lots = float(lots_target[t])

        is_entry = (prev_units == 0.0) and (desired_units != 0.0)
        if lev is not None and is_entry and margin_policy == "skip_entry":
            req_margin = abs(desired_units * price) / lev
            if req_margin > eq:
                desired_units = 0.0
                desired_lots = 0.0

        d_units = desired_units - prev_units
        d_lots = desired_lots - prev_lots

        costs = abs(d_lots) * cost_per_lot

        d_price = price - prev_price
        pnl = prev_units * d_price - costs

        eq = eq + pnl

        eq_arr[t] = eq
        pnl_arr[t] = pnl
        costs_arr[t] = costs
        units_arr[t] = desired_units

        prev_price = price
        prev_units = desired_units
        prev_lots = desired_lots

    return eq_arr, pnl_arr, costs_arr, units_arr


if _HAVE_NUMBA:

    @njit(cache=True)
    def _bt_loop_numba(
        px,
        units_target,
        lots_target,
        initial_capital,
        leverage,
        fee_per_lot,
        spread_per_lot,
        margin_policy_skip_entry,
    ):
        """Numba-accelerated loop. Returns (equity, pnl, costs, contract_units)."""

        cost_per_lot = fee_per_lot + spread_per_lot

        n = px.shape[0]
        eq_arr = np.empty(n, dtype=np.float64)
        pnl_arr = np.empty(n, dtype=np.float64)
        costs_arr = np.empty(n, dtype=np.float64)
        units_arr = np.empty(n, dtype=np.float64)

        eq = initial_capital

        prev_price = px[0]
        prev_units = 0.0
        prev_lots = 0.0

        lev = leverage  # may be NaN sentinel

        for t in range(n):
            price = px[t]

            desired_units = units_target[t]
            desired_lots = lots_target[t]

            # Entry-only margin check (matches python reference)
            if margin_policy_skip_entry and lev > 0.0:
                if prev_units == 0.0 and desired_units != 0.0:
                    req_margin = abs(desired_units * price) / lev
                    if req_margin > eq:
                        desired_units = 0.0
                        desired_lots = 0.0

            d_units = desired_units - prev_units
            d_lots = desired_lots - prev_lots

            costs = abs(d_lots) * cost_per_lot

            d_price = price - prev_price
            pnl = prev_units * d_price - costs

            eq = eq + pnl

            eq_arr[t] = eq
            pnl_arr[t] = pnl
            costs_arr[t] = costs
            units_arr[t] = desired_units

            prev_price = price
            prev_units = desired_units
            prev_lots = desired_lots

        return eq_arr, pnl_arr, costs_arr, units_arr


def backtest_positions_account_margin(
    *,
    prices: pd.Series,
    positions_size: pd.Series,
    initial_capital: float = 1000.0,
    leverage: float | None = 20.0,
    lot_per_size: float = 0.01,
    contract_size_per_lot: float = 100.0,
    fee_per_lot: float = 0.0,
    spread_per_lot: float = 0.0,
    lag: int = 1,
    max_size: float | None = 2.0,
    margin_policy: str = "skip_entry",  # skip_entry | allow_negative
    use_numba: bool | None = None,
) -> pd.DataFrame:
    """Backtest a 1D strategy using a simple *account + margin* model.

    Model
    -----
    - Strategy emits a *size* series (e.g. 0/1/2 or -1/0/+1). We map to:
        lots = size * lot_per_size
        contract_units = lots * contract_size_per_lot
    - PnL per bar: contract_units_prev * (price[t] - price[t-1]) - costs

    Costs
    -----
    Absolute per-lot (simple + realistic):
        costs = abs(d_lots) * (fee_per_lot + spread_per_lot)

    Margin
    ------
    If leverage is not None:
      required_margin = abs(contract_units * price) / leverage

    On entry (flat -> nonzero) if required_margin > equity:
      - skip_entry: remain flat (contract_units=0)
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

    # Positions are expected to already be discrete (e.g. 0/1/2 or -1/0/+1).
    pos_disc = pos.rename("position").astype(float)

    lots = pos_disc * float(lot_per_size)
    contract_units_target = lots * float(contract_size_per_lot)

    # --- Engine loop (python reference or numba accelerated) ---

    px_arr = px.to_numpy(dtype=np.float64)
    units_arr_target = contract_units_target.to_numpy(dtype=np.float64)
    lots_arr_target = lots.to_numpy(dtype=np.float64)

    if use_numba is None:
        use_numba = _HAVE_NUMBA

    if use_numba and _HAVE_NUMBA:
        # Numba doesn't accept None; use -1.0 sentinel to disable leverage.
        lev = -1.0 if leverage is None else float(leverage)
        eq_arr, pnl_arr, costs_arr, units_arr = _bt_loop_numba(
            px_arr,
            units_arr_target,
            lots_arr_target,
            float(initial_capital),
            lev,
            float(fee_per_lot),
            float(spread_per_lot),
            margin_policy == "skip_entry",
        )
    else:
        eq_arr, pnl_arr, costs_arr, units_arr = _bt_loop_python(
            px_arr,
            units_arr_target,
            lots_arr_target,
            initial_capital=float(initial_capital),
            leverage=leverage,
            fee_per_lot=float(fee_per_lot),
            spread_per_lot=float(spread_per_lot),
            margin_policy=margin_policy,
        )

    eq_s = pd.Series(eq_arr, index=px.index, name="equity")
    pnl_s = pd.Series(pnl_arr, index=px.index, name="pnl")
    costs_s = pd.Series(costs_arr, index=px.index, name="costs")
    contract_units_s = pd.Series(units_arr, index=px.index, name="contract_units")

    eq_prev = eq_s.shift(1)
    ret = (pnl_s / eq_prev.replace(0.0, np.nan)).fillna(0.0)

    return pd.DataFrame(
        {
            "position": pos_disc,
            "lots": lots.astype(float),
            "contract_units": contract_units_s.astype(float),
            "pnl": pnl_s.astype(float),
            "costs": costs_s.astype(float),
            "returns_net": ret.astype(float),
            "equity": eq_s.astype(float),
        }
    )

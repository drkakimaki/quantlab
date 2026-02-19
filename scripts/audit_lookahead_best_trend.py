from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.backtest import backtest_positions_account_margin, prices_to_returns
from quantlab.metrics import sharpe, max_drawdown


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


def load_ohlc_daily(*, symbol: str, start: dt.date, end: dt.date, root: Path) -> pd.DataFrame:
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


@dataclass(frozen=True)
class Metrics:
    period: str
    variant: str
    pnl_pct: float
    maxdd_pct: float
    sharpe: float
    n_trades: int


def n_trades_from_position(bt: pd.DataFrame) -> int:
    if bt is None or bt.empty:
        return 0
    execs = int((bt["position"].fillna(0.0).diff().abs() > 0).sum())
    return int(execs // 2)


def compute_sign_flips(series: pd.Series) -> pd.Series:
    sign = series.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    flips = (sign != sign.shift(1)).astype(int)
    flips = flips.where(sign != 0.0, 0)
    return flips


def stable_mask(
    *,
    px_base: pd.Series,
    close_other: pd.Series,
    window: int,
    min_abs: float,
    flip_lb: int,
    max_flips: int,
    shift_bars: int,
) -> pd.Series:
    cc = pd.Series(close_other).dropna().copy()
    cc.index = pd.to_datetime(cc.index)
    cc = cc.sort_index().reindex(px_base.index).ffill()

    r_base = prices_to_returns(px_base).fillna(0.0)
    r_cc = prices_to_returns(cc).fillna(0.0)
    c = r_base.rolling(int(window), min_periods=int(window)).corr(r_cc)

    flips = compute_sign_flips(c)
    flip_cnt = flips.rolling(int(flip_lb), min_periods=int(flip_lb)).sum()

    ok = (c.abs() >= float(min_abs)) & (flip_cnt <= int(max_flips))
    if shift_bars:
        ok = ok.shift(int(shift_bars))
    return ok.fillna(False)


def best_trend_positions(
    *,
    prices_5m: pd.Series,
    htf_bars_15m: pd.DataFrame,
    corr_xag_5m: pd.Series,
    corr_eur_5m: pd.Series,
    # fixed params
    fast: int = 30,
    slow: int = 75,
    # filters
    ema_fast: int = 50,
    ema_slow: int = 250,
    atr_n: int = 14,
    sep_k: float = 0.15,
    nochop_ema: int = 15,
    nochop_lookback: int = 20,
    nochop_min_closes: int = 12,
    # corr params (current best)
    corr_logic: str = "or",
    corr_window_xag: int = 40,
    corr_min_abs_xag: float = 0.10,
    corr_flip_lookback_xag: int = 50,
    corr_max_flips_xag: int = 0,
    corr_window_eur: int = 75,
    corr_min_abs_eur: float = 0.10,
    corr_flip_lookback_eur: int = 75,
    corr_max_flips_eur: int = 5,
    # sizing
    confirm_size_one: float = 1.0,
    confirm_size_both: float = 2.0,
    # audit knobs
    htf_extra_shift: int = 0,
    corr_shift: int = 1,
    size_shift: int = 1,
) -> pd.Series:
    """Compute the *size* series (0/1/2) for best_trend with audit toggles.

    - htf_extra_shift: additional shift in HTF gates (in HTF bars) before forward-fill.
      If 1, uses previous closed HTF bar (more conservative).
    - corr_shift: shift in base bars applied to corr stability ok-mask.
      Production uses 1.
    - size_shift: shift in base bars for sizing at entry.
      Production uses 1.
    """

    corr_logic = corr_logic.strip().lower()
    if corr_logic not in {"or", "and"}:
        raise ValueError("corr_logic must be or/and")

    px = prices_5m.dropna().astype(float).copy()
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()

    # base
    sma_fast = px.rolling(int(fast), min_periods=int(fast)).mean()
    sma_slow = px.rolling(int(slow), min_periods=int(slow)).mean()
    base_signal = (sma_fast > sma_slow).astype(float)

    # HTF
    bars = htf_bars_15m.copy()
    bars.index = pd.to_datetime(bars.index)
    bars = bars.sort_index()
    bars = bars.rename(columns={c: c.lower() for c in bars.columns})
    for c in ("open", "high", "low", "close"):
        if c not in bars.columns:
            raise ValueError("htf_bars_15m must have OHLC")

    px_htf = bars["close"].astype(float).dropna()

    sma_fast_htf = px_htf.rolling(int(fast), min_periods=int(fast)).mean()
    sma_slow_htf = px_htf.rolling(int(slow), min_periods=int(slow)).mean()
    htf_on = (sma_fast_htf > sma_slow_htf)
    if htf_extra_shift:
        htf_on = htf_on.shift(int(htf_extra_shift))
    htf_on_base = htf_on.reindex(px.index).ffill().fillna(False)

    pos = (base_signal > 0.0) & htf_on_base

    # EMA sep
    ema_f = px_htf.ewm(span=int(ema_fast), adjust=False).mean()
    ema_s = px_htf.ewm(span=int(ema_slow), adjust=False).mean()
    h = bars["high"].astype(float).reindex(px_htf.index)
    l = bars["low"].astype(float).reindex(px_htf.index)
    prev_c = px_htf.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(int(atr_n), min_periods=int(atr_n)).mean()
    ema_sep_ok = (ema_f > ema_s) & ((ema_f - ema_s) > float(sep_k) * atr)
    if htf_extra_shift:
        ema_sep_ok = ema_sep_ok.shift(int(htf_extra_shift))
    ema_sep_ok_base = ema_sep_ok.reindex(px.index).ffill().fillna(False)
    pos = pos & ema_sep_ok_base

    # NoChop
    ema_nc = px_htf.ewm(span=int(nochop_ema), adjust=False).mean()
    above = (px_htf > ema_nc).astype(int)
    above_cnt = above.rolling(int(nochop_lookback), min_periods=int(nochop_lookback)).sum()
    nochop_ok = (above_cnt >= int(nochop_min_closes))
    if htf_extra_shift:
        nochop_ok = nochop_ok.shift(int(htf_extra_shift))
    nochop_ok_base = nochop_ok.reindex(px.index).ffill().fillna(False)
    pos = pos & nochop_ok_base

    # Corr stability
    stable_xag = stable_mask(
        px_base=px,
        close_other=corr_xag_5m,
        window=corr_window_xag,
        min_abs=corr_min_abs_xag,
        flip_lb=corr_flip_lookback_xag,
        max_flips=corr_max_flips_xag,
        shift_bars=corr_shift,
    )
    stable_eur = stable_mask(
        px_base=px,
        close_other=corr_eur_5m,
        window=corr_window_eur,
        min_abs=corr_min_abs_eur,
        flip_lb=corr_flip_lookback_eur,
        max_flips=corr_max_flips_eur,
        shift_bars=corr_shift,
    )

    if corr_logic == "or":
        stable_ok = stable_xag | stable_eur
    else:
        stable_ok = stable_xag & stable_eur

    gate_on = pos
    entry_bar = gate_on & (~gate_on.shift(1, fill_value=False))
    entry_ok = entry_bar & stable_ok
    seg = gate_on.ne(gate_on.shift(1, fill_value=False)).cumsum()
    seg_entry_ok = entry_ok.groupby(seg).transform("max")
    gate_final = gate_on & seg_entry_ok

    # sizing
    if corr_logic == "or":
        both_ok = stable_xag & stable_eur
        one_ok = stable_ok
    else:
        both_ok = stable_ok
        one_ok = stable_ok

    s = pd.Series(0.0, index=gate_final.index)
    s = s.where(~one_ok, float(confirm_size_one))
    s = s.where(~both_ok, float(confirm_size_both))

    size_entry = s.shift(int(size_shift)).fillna(0.0) if size_shift else s.fillna(0.0)
    size_on_entry = size_entry.where(entry_bar)
    size_in_seg = size_on_entry.groupby(seg).ffill().fillna(0.0)

    out = gate_final.astype(float) * size_in_seg
    return out.reindex(px.index).fillna(0.0).astype(float)


def bt_from_positions(px: pd.Series, pos_size: pd.Series, *, lag: int) -> pd.DataFrame:
    return backtest_positions_account_margin(
        prices=px,
        positions_size=pos_size,
        initial_capital=1000.0,
        leverage=20.0,
        lot_per_size=0.01,
        contract_size_per_lot=100.0,
        fee_per_lot=0.0,
        spread_per_lot=0.0,
        lag=lag,
        max_size=2.0,
        margin_policy="skip_entry",
    )


def metrics_for_bt(period: str, variant: str, bt: pd.DataFrame) -> Metrics:
    s = float(sharpe(bt["returns_net"], freq="5MIN"))
    dd = float(max_drawdown(bt["equity"]))
    pnl_pct = (float(bt["equity"].iloc[-1]) / 1000.0 - 1.0) * 100.0
    peak = bt["equity"].cummax()
    maxdd_pct = float((bt["equity"] / peak - 1.0).min()) * 100.0
    return Metrics(
        period=period,
        variant=variant,
        pnl_pct=pnl_pct,
        maxdd_pct=maxdd_pct,
        sharpe=float(summ.sharpe),
        n_trades=n_trades_from_position(bt),
    )


def htf_alignment_violations(px_index: pd.DatetimeIndex, htf_index: pd.DatetimeIndex) -> dict:
    """Check for any 5m timestamp that would be filled from an HTF timestamp > t.

    Uses merge_asof to find last HTF <= base.
    """
    base = pd.DataFrame({"ts": px_index}).sort_values("ts")
    htf = pd.DataFrame({"htf_ts": htf_index}).sort_values("htf_ts")
    m = pd.merge_asof(base, htf, left_on="ts", right_on="htf_ts", direction="backward")
    # violation would mean chosen htf_ts > ts (should never happen with backward)
    viol = (m["htf_ts"] > m["ts"]).fillna(False)
    return {
        "n_base": int(len(px_index)),
        "n_htf": int(len(htf_index)),
        "n_missing_htf": int(m["htf_ts"].isna().sum()),
        "n_future_violation": int(viol.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Lookahead / alignment audit for best_trend")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/trend_based/audits"))
    ap.add_argument("--root-5m-ohlc", type=Path, default=Path("data/dukascopy_5m_ohlc"))
    ap.add_argument("--root-15m-ohlc", type=Path, default=Path("data/dukascopy_15m_ohlc"))
    ap.add_argument("--p3-end", type=dt.date.fromisoformat, default=dt.date(2026, 2, 13))
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    periods = {
        "2021-2022": (dt.date(2021, 1, 1), dt.date(2022, 12, 31)),
        "2023-2025": (dt.date(2023, 1, 1), dt.date(2025, 12, 31)),
        "2026": (dt.date(2026, 1, 1), args.p3_end),
    }

    # Variants: try to detect lookahead by making model more/less conservative
    variants = [
        # (name, engine_lag, htf_extra_shift, corr_shift, size_shift)
        ("prod_like", 1, 0, 1, 1),
        ("engine_lag2", 2, 0, 1, 1),
        ("engine_lag0_DANGER", 0, 0, 1, 1),
        ("htf_extra_shift1", 1, 1, 1, 1),
        ("corr_shift0_DANGER", 1, 0, 0, 1),
        ("size_shift0_DANGER", 1, 0, 1, 0),
    ]

    all_metrics: list[dict] = []
    align_rows: list[dict] = []

    for pname, (start, end) in periods.items():
        bars5 = load_ohlc_daily(symbol="XAUUSD", start=start, end=end, root=args.root_5m_ohlc)
        bars15 = load_ohlc_daily(symbol="XAUUSD", start=start, end=end, root=args.root_15m_ohlc)
        px = bars5["close"].astype(float)

        # corr series
        xag = load_ohlc_daily(symbol="XAGUSD", start=start, end=end, root=args.root_5m_ohlc)["close"].astype(float)
        eur = load_ohlc_daily(symbol="EURUSD", start=start, end=end, root=args.root_5m_ohlc)["close"].astype(float)

        # alignment check
        align = htf_alignment_violations(px.index, bars15.index)
        align_rows.append({"period": pname, **align})

        for vname, eng_lag, htf_shift, corr_shift, size_shift in variants:
            pos_size = best_trend_positions(
                prices_5m=px,
                htf_bars_15m=bars15,
                corr_xag_5m=xag,
                corr_eur_5m=eur,
                htf_extra_shift=htf_shift,
                corr_shift=corr_shift,
                size_shift=size_shift,
            )
            bt = bt_from_positions(px, pos_size, lag=eng_lag)
            m = metrics_for_bt(pname, vname, bt)
            all_metrics.append(m.__dict__)

    dfm = pd.DataFrame(all_metrics)
    dfa = pd.DataFrame(align_rows)

    dfm.to_csv(out_dir / "lookahead_metrics.csv", index=False)
    dfa.to_csv(out_dir / "htf_alignment.csv", index=False)

    # Markdown report
    lines = []
    lines.append("# Lookahead / Fault-Confidence Audit — best_trend\n")
    lines.append("This audit stress-tests the strategy for common lookahead/alignment faults by comparing results under more/less conservative variants.\n")
    lines.append("\n## HTF alignment sanity (15m -> 5m)\n")
    lines.append(f"CSV: `{out_dir / 'htf_alignment.csv'}`\n")
    lines.append(dfa.to_string(index=False))

    lines.append("\n\n## Performance deltas under audit variants\n")
    lines.append(f"CSV: `{out_dir / 'lookahead_metrics.csv'}`\n")

    for pname in periods:
        sub = dfm[dfm.period == pname].copy()
        sub = sub.sort_values("variant")
        lines.append(f"\n### {pname}\n")
        lines.append(sub[["variant", "pnl_pct", "maxdd_pct", "sharpe", "n_trades"]].to_string(index=False))

    lines.append(
        "\n\n## How to interpret\n"
        "- `prod_like` should be the reference.\n"
        "- If `engine_lag0_DANGER` or `corr_shift0_DANGER` performs dramatically better, that indicates the strategy *would* benefit from lookahead, and we should be confident we are *not* using it in production.\n"
        "- If `htf_extra_shift1` changes results a lot, HTF alignment is a sensitive area; this is not necessarily bias, but it is a key porting risk (e.g. MQL).\n"
    )

    (out_dir / "lookahead_audit.md").write_text("\n".join(lines) + "\n")

    print(f"Wrote: {out_dir / 'lookahead_audit.md'}")
    print(f"Wrote: {out_dir / 'lookahead_metrics.csv'}")
    print(f"Wrote: {out_dir / 'htf_alignment.csv'}")


if __name__ == "__main__":
    main()

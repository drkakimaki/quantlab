from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from quantlab.strategies import trend_following_ma_crossover


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


def prices_to_returns(px: pd.Series) -> pd.Series:
    px = pd.Series(px).astype(float)
    return px.pct_change().fillna(0.0)


@dataclass
class DrawdownEpisode:
    peak_ts: pd.Timestamp
    trough_ts: pd.Timestamp
    recover_ts: pd.Timestamp
    depth: float  # negative
    duration_bars: int
    time_to_trough_bars: int


def compute_drawdown_episodes(equity: pd.Series) -> tuple[pd.Series, list[DrawdownEpisode]]:
    eq = pd.Series(equity).dropna().astype(float)
    eq = eq.sort_index()
    hwm = eq.cummax()
    dd = eq / hwm - 1.0

    in_dd = dd < 0
    episodes: list[DrawdownEpisode] = []
    if not bool(in_dd.any()):
        return dd, episodes

    # Identify contiguous underwater regions.
    grp = (in_dd != in_dd.shift(1, fill_value=False)).cumsum()
    for g, mask in in_dd.groupby(grp):
        if not bool(mask.iloc[0]):
            continue
        idx = mask.index
        start = idx[0]
        end = idx[-1]

        # Peak is last timestamp before start (or start itself if none).
        prev_idx = dd.index[dd.index.get_loc(start) - 1] if dd.index.get_loc(start) > 0 else start
        peak_ts = prev_idx

        dd_seg = dd.loc[start:end]
        trough_ts = dd_seg.idxmin()
        depth = float(dd_seg.min())

        # Recovery time: first time after end where dd==0 (or last index if never recovers)
        after = dd.loc[end:]
        rec_mask = after >= 0
        if bool(rec_mask.any()):
            recover_ts = rec_mask.idxmax()  # first True
        else:
            recover_ts = dd.index[-1]

        # duration from peak->recovery in bars
        peak_loc = dd.index.get_loc(peak_ts)
        rec_loc = dd.index.get_loc(recover_ts)
        trough_loc = dd.index.get_loc(trough_ts)
        episodes.append(
            DrawdownEpisode(
                peak_ts=peak_ts,
                trough_ts=trough_ts,
                recover_ts=recover_ts,
                depth=depth,
                duration_bars=int(rec_loc - peak_loc),
                time_to_trough_bars=int(trough_loc - peak_loc),
            )
        )

    return dd, episodes


@dataclass
class Segment:
    seg_id: int
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    size: float
    duration_bars: int
    ret_end: float
    mae: float
    mfe: float
    min_equity: float
    max_equity: float
    entry_equity: float
    exit_equity: float


def compute_segments(bt: pd.DataFrame) -> list[Segment]:
    pos = bt["position"].astype(float)
    eq = bt["equity"].astype(float)
    in_pos = pos > 0
    if not bool(in_pos.any()):
        return []

    seg = in_pos.ne(in_pos.shift(1, fill_value=False)).cumsum()
    entry_bar = in_pos & (~in_pos.shift(1, fill_value=False))

    segments: list[Segment] = []
    seg_ids = sorted(seg[in_pos].unique().tolist())
    for sid in seg_ids:
        m = (seg == sid) & in_pos
        idx = bt.index[m]
        entry_ts = idx[0]
        exit_ts = idx[-1]
        entry_eq = float(eq.loc[entry_ts])
        exit_eq = float(eq.loc[exit_ts])
        path = eq.loc[entry_ts:exit_ts]
        rel = path / entry_eq - 1.0
        mae = float(rel.min())
        mfe = float(rel.max())
        ret_end = float(exit_eq / entry_eq - 1.0)

        # Entry size: use position at entry bar (already lagged inside bt)
        size = float(pos.loc[entry_ts])
        segments.append(
            Segment(
                seg_id=int(sid),
                entry_ts=entry_ts,
                exit_ts=exit_ts,
                size=size,
                duration_bars=int(len(idx)),
                ret_end=ret_end,
                mae=mae,
                mfe=mfe,
                min_equity=float(path.min()),
                max_equity=float(path.max()),
                entry_equity=entry_eq,
                exit_equity=exit_eq,
            )
        )

    return segments


def daily_returns_while_in_position(bt: pd.DataFrame) -> pd.DataFrame:
    # Use end-of-day equity and whether we were in position at any point during that day.
    eq = bt["equity"].astype(float)
    pos = bt["position"].astype(float)

    daily_eq = eq.resample("1D").last().dropna()
    daily_ret = daily_eq.pct_change().fillna(0.0)

    in_pos_any = (pos > 0).resample("1D").max().reindex(daily_eq.index).fillna(False)

    # shock trigger condition (abs 5m return >= 0.006), shifted 1 like strategy
    # computed on underlying price proxy via equity? better use bt returns? Here we reconstruct from equity doesn't work.
    # We'll approximate using bt returns_net + costs? Instead, use returns from prices stored in bt if available (not).
    # Caller should pass separate price series if exact is needed.
    out = pd.DataFrame({"daily_return": daily_ret, "in_position_any": in_pos_any.astype(bool)})
    return out


def shock_trigger_series(px_5m: pd.Series, *, abs_thr: float = 0.006) -> pd.Series:
    r = prices_to_returns(px_5m).astype(float)
    shock = (r.abs() >= float(abs_thr)).shift(1).fillna(False)
    shock.index = px_5m.index
    return shock.astype(bool)


def period_summaries(name: str, episodes: list[DrawdownEpisode], bar_minutes: int = 5) -> dict:
    if not episodes:
        return {
            "period": name,
            "n_episodes": 0,
        }
    depths = np.array([e.depth for e in episodes], dtype=float)
    durs = np.array([e.duration_bars for e in episodes], dtype=float)
    ttt = np.array([e.time_to_trough_bars for e in episodes], dtype=float)

    def q(a: np.ndarray, p: float) -> float:
        return float(np.quantile(a, p))

    mins_per_bar = float(bar_minutes)

    return {
        "period": name,
        "n_episodes": int(len(episodes)),
        "depth_min": float(depths.min()),
        "depth_p50": q(depths, 0.50),
        "depth_p90": q(depths, 0.90),
        "depth_p95": q(depths, 0.95),
        "depth_p99": q(depths, 0.99),
        "duration_days_p50": q(durs * mins_per_bar / (60 * 24), 0.50),
        "duration_days_p90": q(durs * mins_per_bar / (60 * 24), 0.90),
        "duration_days_p95": q(durs * mins_per_bar / (60 * 24), 0.95),
        "duration_days_max": float((durs.max() * mins_per_bar) / (60 * 24)),
        "ttt_days_p50": q(ttt * mins_per_bar / (60 * 24), 0.50),
        "ttt_days_p90": q(ttt * mins_per_bar / (60 * 24), 0.90),
        "ttt_days_max": float((ttt.max() * mins_per_bar) / (60 * 24)),
    }


def main() -> None:
    symbol = "XAUUSD"
    corr_symbol = "XAGUSD"
    corr2_symbol = "EURUSD"

    root_5m = Path("data/dukascopy_5m_ohlc")
    root_15m = Path("data/dukascopy_15m_ohlc")

    p0_start = dt.date(2021, 1, 1)
    p1_end = dt.date(2022, 12, 31)
    p2_start = dt.date(2023, 1, 1)
    p2_end = dt.date(2025, 12, 31)
    p3_start = dt.date(2026, 1, 1)
    p3_end = dt.date.today()

    periods = {
        "2021-2022": (p0_start, p1_end),
        "2023-2025": (p2_start, p2_end),
        "2026": (p3_start, p3_end),
    }

    out_dir = Path("reports/trend_based/decisions/2026-02-15_loss_drawdown_deepdive")
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_dd_summary = []
    all_dd_summary_noshock = []

    seg_rows = []
    dd_ep_rows = []
    worst_daily_rows = []

    # Also compute baseline without shock to compare tail behavior
    for variant, shock_thr in [("shock_abs0p006", 0.006), ("baseline_no_shock", 0.0)]:
        for pname, (start, end) in periods.items():
            bars5 = load_ohlc_daily(symbol=symbol, start=start, end=end, root=root_5m)
            bars15 = load_ohlc_daily(symbol=symbol, start=start, end=end, root=root_15m)
            px = bars5["close"].astype(float)
            corr_xag = load_ohlc_daily(symbol=corr_symbol, start=start, end=end, root=root_5m)["close"].astype(float)
            corr_eur = load_ohlc_daily(symbol=corr2_symbol, start=start, end=end, root=root_5m)["close"].astype(float)

            bt, _, _ = trend_following_ma_crossover(
                px,
                fast=30,
                slow=75,
                fee_bps=0.0,
                slippage_bps=0.0,
                htf_confirm={"bars_15m": bars15, "rule": "15min"},
                ema_sep={"ema_fast": 40, "ema_slow": 300, "atr_n": 20, "sep_k": 0.05},
                nochop={"ema": 20, "lookback": 40, "min_closes": 24, "entry_held": False, "exit_bad_bars": 0},
                corr={
                    "logic": "or",
                    "xag": {"close": corr_xag, "window": 40, "min_abs": 0.10, "flip_lookback": 50, "max_flips": 0},
                    "eur": {"close": corr_eur, "window": 75, "min_abs": 0.10, "flip_lookback": 75, "max_flips": 5},
                },
                sizing={"confirm_size_one": 1.0, "confirm_size_both": 2.0},
                risk={
                    "shock_exit_abs_ret": float(shock_thr),
                    "shock_exit_sigma_k": 0.0,
                    "shock_exit_sigma_window": 96,
                    "shock_cooldown_bars": 0,
                    "segment_ttl_bars": 0,
                },
            )

            # Drawdowns
            dd, episodes = compute_drawdown_episodes(bt["equity"])
            summ = period_summaries(pname, episodes)
            summ["variant"] = variant
            if variant == "shock_abs0p006":
                all_dd_summary.append(summ)
            else:
                all_dd_summary_noshock.append(summ)

            for e in episodes:
                dd_ep_rows.append(
                    {
                        "variant": variant,
                        "period": pname,
                        "peak_ts": e.peak_ts,
                        "trough_ts": e.trough_ts,
                        "recover_ts": e.recover_ts,
                        "depth": e.depth,
                        "duration_bars": e.duration_bars,
                        "time_to_trough_bars": e.time_to_trough_bars,
                    }
                )

            # Segments
            segs = compute_segments(bt)
            for s in segs:
                seg_rows.append(
                    {
                        "variant": variant,
                        "period": pname,
                        "seg_id": s.seg_id,
                        "entry_ts": s.entry_ts,
                        "exit_ts": s.exit_ts,
                        "size": s.size,
                        "duration_bars": s.duration_bars,
                        "ret_end": s.ret_end,
                        "mae": s.mae,
                        "mfe": s.mfe,
                    }
                )

            # Worst daily returns while in-position + shock hit that day
            shock = shock_trigger_series(px, abs_thr=0.006)
            daily = daily_returns_while_in_position(bt)
            daily["shock_trigger_any"] = shock.resample("1D").max().reindex(daily.index).fillna(False)

            # Keep only in-position days
            d_in = daily[daily["in_position_any"]].copy()
            d_in = d_in.sort_values("daily_return")
            top_n = d_in.head(15)
            for ts, row in top_n.iterrows():
                worst_daily_rows.append(
                    {
                        "variant": variant,
                        "period": pname,
                        "date": ts.date().isoformat(),
                        "daily_return": float(row["daily_return"]),
                        "shock_trigger_any": bool(row["shock_trigger_any"]),
                    }
                )

            # Write heavy intermediates to raw for possible later inspection
            bt_path = raw_dir / f"bt_{variant}_{pname.replace(' ', '_')}.parquet"
            bt.to_parquet(bt_path)

    # Compact outputs
    dd_summary_df = pd.DataFrame(all_dd_summary).sort_values(["period"])
    dd_summary_noshock_df = pd.DataFrame(all_dd_summary_noshock).sort_values(["period"])
    dd_eps_df = pd.DataFrame(dd_ep_rows).sort_values(["variant", "period", "peak_ts"])
    seg_df = pd.DataFrame(seg_rows).sort_values(["variant", "period", "entry_ts"])
    worst_daily_df = pd.DataFrame(worst_daily_rows).sort_values(["variant", "period", "daily_return"])

    dd_summary_df.to_csv(out_dir / "drawdown_summary_shock_abs0p006.csv", index=False)
    dd_summary_noshock_df.to_csv(out_dir / "drawdown_summary_baseline_no_shock.csv", index=False)

    # Keep these compact (top 500 rows max)
    dd_eps_df.head(500).to_csv(out_dir / "drawdown_episodes_head.csv", index=False)
    seg_df.head(500).to_csv(out_dir / "segments_head.csv", index=False)
    worst_daily_df.to_csv(out_dir / "worst_daily_in_position.csv", index=False)

    print("Wrote outputs to:", out_dir)


if __name__ == "__main__":
    main()

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantlab.strategies import trend_following_ma_crossover_htf_confirm


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


def load_ohlc_from_daily_parquet(*, symbol: str, start: dt.date, end: dt.date, root: Path) -> pd.DataFrame:
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


def trade_table(bt: pd.DataFrame) -> pd.DataFrame:
    """Return per-trade table with entry ts, exit ts, trade return (decimal)."""
    if bt is None or bt.empty:
        return pd.DataFrame(columns=["entry_ts", "exit_ts", "trade_ret"])

    pos = bt["position"].fillna(0.0).astype(float)
    r = bt["returns_net"].fillna(0.0).astype(float)

    in_pos = pos != 0.0
    if not bool(in_pos.any()):
        return pd.DataFrame(columns=["entry_ts", "exit_ts", "trade_ret"])

    prev_in = in_pos.shift(1, fill_value=False)
    entry = in_pos & (~prev_in)
    trade_id = entry.cumsum()

    df = pd.DataFrame({"trade_id": trade_id, "in_pos": in_pos, "r": r}, index=bt.index)
    df_in = df[df["in_pos"]].copy()
    if df_in.empty:
        return pd.DataFrame(columns=["entry_ts", "exit_ts", "trade_ret"])

    # compound per trade
    log1p = np.log((1.0 + df_in["r"]).clip(lower=1e-12))
    trade_log = log1p.groupby(df_in["trade_id"]).sum()
    trade_ret = np.expm1(trade_log).astype(float)

    entry_ts = df_in.groupby("trade_id").apply(lambda x: x.index.min())
    exit_ts = df_in.groupby("trade_id").apply(lambda x: x.index.max())

    out = pd.DataFrame({"entry_ts": entry_ts, "exit_ts": exit_ts, "trade_ret": trade_ret})
    out = out.reset_index(drop=True)
    return out


def heatmap_png(mat: pd.DataFrame, *, title: str, fmt: str, cmap: str) -> str:
    """Save a heatmap to a PNG data URI."""
    import base64
    import io

    fig, ax = plt.subplots(figsize=(10.5, 3.2))

    data = mat.values
    im = ax.imshow(data, aspect="auto", cmap=cmap)

    ax.set_title(title)
    ax.set_yticks(range(mat.shape[0]), labels=[str(h).zfill(2) for h in mat.index])
    ax.set_xticks(range(mat.shape[1]), labels=list(mat.columns))
    ax.set_ylabel("Entry hour (UTC)")

    # annotate
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                txt = ""
            else:
                txt = format(v, fmt)
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()

    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(bio.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def main() -> None:
    out_path = Path("reports/trend_variants/time_of_day_heatmap.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    root_5m_ohlc = Path("data/dukascopy_5m_ohlc")
    root_15m_ohlc = Path("data/dukascopy_15m_ohlc")

    # Periods
    periods = {
        "2022": (dt.date(2022, 1, 1), dt.date(2022, 12, 31)),
        "2023-2025": (dt.date(2023, 1, 1), dt.date(2025, 12, 31)),
        "2026": (dt.date(2026, 1, 1), dt.date.today()),
    }

    # Baseline = combined corr OR best + confirm sizing (same as best_confirm_one1_both15_fx260)
    params = dict(
        fast=30,
        slow=75,
        htf_rule="15min",
        ema_sep_filter=True,
        ema_fast=50,
        ema_slow=250,
        atr_n=14,
        sep_k=0.15,
        nochop_filter=True,
        nochop_ema=15,
        nochop_lookback=20,
        nochop_min_closes=12,
        base_slope_filter=True,
        base_slope_ema=100,
        base_slope_window=30,
        base_slope_eps=0.0,
        corr_filter=True,
        corr_window=50,
        corr_min_abs=0.25,
        corr_flip_lookback=100,
        corr_max_flips=1,
        corr2_window=75,
        corr2_min_abs=0.20,
        corr2_flip_lookback=50,
        corr2_max_flips=4,
        corr_logic="or",
        sizing_mode="confirm",
        confirm_size_one=1.0,
        confirm_size_both=1.5,
        # disable other experiments
        beta_confirm=False,
        retest_entry=False,
        exit_ema_filter=False,
        fee_bps=0.0,
        slippage_bps=0.0,
        long_only=True,
    )

    all_trades = []

    for name, (a, b) in periods.items():
        xau5 = load_ohlc_from_daily_parquet(symbol="XAUUSD", start=a, end=b, root=root_5m_ohlc)
        xau15 = load_ohlc_from_daily_parquet(symbol="XAUUSD", start=a, end=b, root=root_15m_ohlc)
        xag5 = load_ohlc_from_daily_parquet(symbol="XAGUSD", start=a, end=b, root=root_5m_ohlc)
        eur5 = load_ohlc_from_daily_parquet(symbol="EURUSD", start=a, end=b, root=root_5m_ohlc)

        px = xau5["close"].astype(float)
        cc1 = xag5["close"].astype(float)
        cc2 = eur5["close"].astype(float)

        bt, _, _ = trend_following_ma_crossover_htf_confirm(px, htf_bars=xau15, corr_close=cc1, corr_close2=cc2, **params)
        tt = trade_table(bt)
        if tt.empty:
            continue
        tt["period"] = name
        tt["entry_hour"] = pd.to_datetime(tt["entry_ts"]).dt.hour.astype(int)
        tt["win"] = (tt["trade_ret"] > 0.0).astype(int)
        all_trades.append(tt)

    if not all_trades:
        raise SystemExit("No trades found to build heatmap")

    trades = pd.concat(all_trades, ignore_index=True)

    # Build matrices: rows=hours, cols=periods
    hours = list(range(24))
    cols = list(periods.keys())

    winrate = pd.DataFrame(index=hours, columns=cols, dtype=float)
    avgret = pd.DataFrame(index=hours, columns=cols, dtype=float)
    ntr = pd.DataFrame(index=hours, columns=cols, dtype=float)

    for p in cols:
        sub = trades[trades["period"] == p]
        for h in hours:
            ss = sub[sub["entry_hour"] == h]
            if len(ss) == 0:
                winrate.loc[h, p] = np.nan
                avgret.loc[h, p] = np.nan
                ntr.loc[h, p] = 0
            else:
                winrate.loc[h, p] = 100.0 * float(ss["win"].mean())
                avgret.loc[h, p] = 100.0 * float(ss["trade_ret"].mean())
                ntr.loc[h, p] = float(len(ss))

    uri_wr = heatmap_png(winrate, title="Win rate by entry hour (UTC) [%]", fmt=".0f", cmap="YlGn")
    uri_ar = heatmap_png(avgret, title="Avg trade return by entry hour (UTC) [%]", fmt=".2f", cmap="RdYlGn")
    uri_nt = heatmap_png(ntr, title="# Trades by entry hour (UTC)", fmt=".0f", cmap="Blues")

    html = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>Time-of-day heatmap (entry hour, UTC)</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; color:#111; }}
    h1 {{ margin: 0 0 6px 0; font-size: 20px; }}
    .sub {{ color:#666; margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
    .card {{ border:1px solid #e2e6ea; border-radius: 12px; padding: 10px; background: #fff; }}
    img {{ width: 100%; height: auto; display: block; }}
    code {{ background:#f5f7fa; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Time-of-day heatmap (entry-only analysis, UTC)</h1>
  <div class='sub'>Shows per-hour entry quality for the current baseline: combined corr stability (XAGUSD OR EURUSD) + confirm sizing (1.0/1.5). Values are grouped by <b>trade entry hour (UTC)</b>.</div>

  <div class='grid'>
    <div class='card'><img src='{uri_wr}' alt='winrate heatmap'/></div>
    <div class='card'><img src='{uri_ar}' alt='avgret heatmap'/></div>
    <div class='card'><img src='{uri_nt}' alt='ntrades heatmap'/></div>
  </div>

  <div class='sub' style='margin-top:14px'>Next step: choose a set of allowed entry hours (UTC) and rerun the backtest with an entry-only time filter.</div>
</body>
</html>
"""

    out_path.write_text(html, encoding='utf-8')
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

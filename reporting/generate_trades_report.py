from __future__ import annotations

"""Trade breakdown HTML report (multi-period).

This is the canonical trade report generator.

Input shape matches generate_bt_report.report_periods_equity_only:
  periods: dict[str, pd.DataFrame]

The report extracts canonical trades (contiguous position != 0) and renders:
- TOTAL summary + histogram
- breakdown tables (by period, calendar month-of-year aggregated, duration)
- per-period drilldown sections (top winners/losers stay here)
"""

import base64
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .trade_breakdown import build_trade_ledger
from .trade_common import duration_bin, agg_trade_table, profit_factor


def _fig_to_data_uri(fig) -> str:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=140)
    plt.close(fig)
    b64 = base64.b64encode(bio.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _duration_bin(bars: pd.Series) -> pd.Categorical:
    # Backward-compatible alias.
    return duration_bin(bars)


def _profit_factor(pnl: pd.Series) -> float:
    # Backward-compatible alias.
    return profit_factor(pnl)


def _agg(trades: pd.DataFrame, key: str) -> pd.DataFrame:
    # Backward-compatible alias.
    out = agg_trade_table(trades, key)
    # Match this report's expected column subset/order.
    keep = [key, "n_trades", "win_rate", "avg_return", "sum_pnl", "avg_bars", "profit_factor"]
    return out[keep]


_MONTH_MAP = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


def report_periods_trades_html(
    *,
    periods: dict[str, pd.DataFrame],
    out_path: str | Path,
    title: str = "Trade breakdown",
    pos_col: str = "position",
    returns_col: str = "returns_net",
    equity_col: str = "equity",
    costs_col: str = "costs",
) -> Path:
    """Single-file HTML report with trade breakdown per period + TOTAL."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build ledgers per period
    ledgers: list[pd.DataFrame] = []
    per_period: dict[str, pd.DataFrame] = {}
    for name, bt in periods.items():
        if bt is None or len(bt) == 0:
            per_period[name] = pd.DataFrame()
            continue
        tl = build_trade_ledger(
            bt,
            pos_col=pos_col,
            returns_col=returns_col,
            equity_col=equity_col,
            costs_col=costs_col,
        )
        if not tl.empty:
            tl = tl.copy()
            tl["period"] = name

            # Period conversion drops timezone info; month-of-year bucketing doesn't need tz.
            et = pd.to_datetime(tl["entry_time"])
            if getattr(et.dt, "tz", None) is not None:
                et = et.dt.tz_convert("UTC").dt.tz_localize(None)

            tl["entry_month_of_year"] = et.dt.month.map(_MONTH_MAP)
            tl["duration_bin"] = duration_bin(tl["bars"])

            ledgers.append(tl)
            per_period[name] = tl
        else:
            per_period[name] = pd.DataFrame()

    if not ledgers:
        html = f"""<!doctype html>
<html><head><meta charset='utf-8'/><title>{title}</title></head>
<body style='font-family: system-ui; background:#0b1220; color:#e5e7eb; padding:20px;'>
<h1>{title}</h1>
<p>No trades in any period.</p>
</body></html>"""
        out_path.write_text(html, encoding="utf-8")
        return out_path

    all_trades = pd.concat(ledgers, ignore_index=True)

    # Histogram for TOTAL
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    rr = all_trades["trade_return"].astype(float).to_numpy()
    rr = rr[np.isfinite(rr)]
    ax.hist(rr, bins=70, color="#60a5fa", alpha=0.85)
    ax.grid(True, alpha=0.25)
    ax.set_title("TOTAL trade return distribution", fontsize=11)
    ax.set_xlabel("trade_return")
    ax.set_ylabel("count")
    fig.tight_layout()
    hist_uri = _fig_to_data_uri(fig)

    # TOTAL stats
    n_trades = int(len(all_trades))
    win_rate = float(all_trades["win"].mean())
    sum_pnl = float(all_trades["pnl_net"].sum())
    # sum_costs removed from report (keep costs available in per-trade ledger)
    pf_total = _profit_factor(all_trades["pnl_net"])
    avg_bars = float(all_trades["bars"].mean())

    # TOTAL breakdowns
    by_period = _agg(all_trades, "period")
    by_month = _agg(all_trades, "entry_month_of_year")
    # Force calendar order (Jan..Dec) instead of sorting by sum_pnl.
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    by_month["entry_month_of_year"] = pd.Categorical(by_month["entry_month_of_year"], categories=month_order, ordered=True)
    by_month = by_month.sort_values("entry_month_of_year").reset_index(drop=True)

    by_duration = _agg(all_trades, "duration_bin")

    def _pct(x) -> str:
        try:
            x = float(x)
        except Exception:
            return "nan"
        if not np.isfinite(x):
            return "inf" if x > 0 else "nan"
        return f"{x*100:,.2f}%"

    def _num(x) -> str:
        try:
            x = float(x)
        except Exception:
            return "nan"
        if not np.isfinite(x):
            return "inf" if x > 0 else "nan"
        return f"{x:,.2f}"

    def _money(x) -> str:
        try:
            x = float(x)
        except Exception:
            return "nan"
        if not np.isfinite(x):
            return "inf" if x > 0 else "nan"
        return f"{x:,.2f}"

    def _df_to_html(df: pd.DataFrame, *, pct_cols: set[str] = set(), num_cols: set[str] = set()) -> str:
        df = df.copy()
        for c in df.columns:
            if c in pct_cols:
                df[c] = df[c].map(_pct)
            elif c in num_cols or c in {"avg_bars"}:
                df[c] = df[c].map(_num)
            elif c in {"sum_pnl", "avg_pnl", "pnl_net"}:
                df[c] = df[c].map(_money)
        return df.to_html(index=False, escape=False, classes="tbl")

    pct_cols = {"win_rate", "avg_return"}

    def _fmt_ts(x) -> str:
        try:
            ts = pd.to_datetime(x)
            # Drop tz suffix like "+00:00" for display.
            if getattr(ts, "tzinfo", None) is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            return ts.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return str(x)

    # Period sections (compact tables per period)
    period_sections = []
    for name, tl in per_period.items():
        if tl is None or tl.empty:
            period_sections.append(
                f"<details class='card'><summary><b>{name}</b> — no trades</summary><div class='box sub'>No trades.</div></details>"
            )
            continue

        tl_top = tl.sort_values("pnl_net", ascending=False).head(10).copy()
        tl_bot = tl.sort_values("pnl_net", ascending=True).head(10).copy()

        for _df in (tl_top, tl_bot):
            _df["entry_time"] = _df["entry_time"].map(_fmt_ts)
            _df["exit_time"] = _df["exit_time"].map(_fmt_ts)

        sec = []
        sec.append("<details class='card'>")
        sec.append(
            f"<summary><b>{name}</b> — {len(tl):,} trades, sum pnl { _money(float(tl['pnl_net'].sum())) }</summary>"
        )
        sec.append("<div class='box'>")
        sec.append("<div class='grid2'>")
        sec.append("<div>")
        sec.append("<h3>Top winners</h3>")
        sec.append(
            _df_to_html(
                tl_top[["trade_id", "entry_time", "exit_time", "bars", "pnl_net", "trade_return"]],
                pct_cols={"trade_return"},
            )
        )
        sec.append("</div>")
        sec.append("<div>")
        sec.append("<h3>Top losers</h3>")
        sec.append(
            _df_to_html(
                tl_bot[["trade_id", "entry_time", "exit_time", "bars", "pnl_net", "trade_return"]],
                pct_cols={"trade_return"},
            )
        )
        sec.append("</div>")
        sec.append("</div>")
        sec.append("</div>")
        sec.append("</details>")
        period_sections.append("\n".join(sec))

    html = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>{title}</title>
  <style>
    :root {{
      --fg:#e5e7eb; --muted:#9ca3af; --bg:#0b1220;
      --border:#223047; --card:#111a2b; --card2:#0f172a;
      --shadow: 0 18px 40px rgba(0, 0, 0, 0.45);
    }}
    * {{ box-sizing:border-box; }}
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; color:var(--fg); background:var(--bg); margin:0; line-height:1.45; }}
    .page {{ max-width:1180px; margin:0 auto; padding:26px 18px 34px; }}
    .header {{ background: radial-gradient(900px 420px at 20% -20%, rgba(96,165,250,0.18), rgba(0,0,0,0) 55%), linear-gradient(180deg, rgba(17,26,43,0.98), rgba(15,23,42,0.92)); border:1px solid var(--border); border-radius:20px; box-shadow:var(--shadow); padding:18px; margin-bottom:14px; }}
    h1 {{ margin:0 0 6px 0; font-size:30px; letter-spacing:-0.6px; }}
    .sub {{ color:var(--muted); font-size:12px; }}
    .summary-strip {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:10px; }}
    .pill {{ background: var(--card2); border:1px solid var(--border); border-radius:999px; padding:8px 10px; font-size:12px; color:var(--muted); }}
    .pill b {{ color:var(--fg); font-weight:700; font-variant-numeric: tabular-nums; }}

    .card {{ background: var(--card); border:1px solid var(--border); border-radius:18px; box-shadow:var(--shadow); overflow:hidden; }}
    details.card {{ padding:0; }}
    details.card summary {{ padding:12px 14px; cursor:pointer; border-bottom:1px solid var(--border); color:var(--fg); }}

    .box {{ padding:14px; }}
    h2 {{ margin:0 0 10px 0; font-size:14px; }}
    h3 {{ margin:0 0 10px 0; font-size:13px; color:var(--muted); }}

    .grid2 {{ display:grid; grid-template-columns: 1fr; gap:12px; }}
    @media (min-width: 900px) {{ .grid2 {{ grid-template-columns: 1fr 1fr; }} }}

    table.tbl {{ border-collapse:separate; border-spacing:0; width:100%; table-layout: fixed; }}
    .tbl {{ font-size: 11.5px; }}
    .tbl th, .tbl td {{
      border-bottom:1px solid var(--border);
      padding:6px 7px;
      text-align:left;
      vertical-align: top;
      /* Avoid horizontal scroll: wrap within cells */
      white-space: normal;
      overflow: hidden;
      text-overflow: ellipsis;
      word-break: break-word;
    }}
    .tbl th {{ font-size:11.5px; color:var(--muted); background: var(--card2); position: sticky; top:0; z-index:1; }}
    .tbl tbody tr:hover td {{ background: rgba(96,165,250,0.10); }}

    .chart {{ border:1px solid var(--border); border-radius:18px; background: var(--card); box-shadow:var(--shadow); padding:12px; }}
    .chart-title {{ font-size:12px; color:var(--muted); margin:0 0 8px 2px; }}
    .chart img {{ width:100%; height:auto; border:1px solid var(--border); border-radius:12px; background: var(--card2); display:block; }}

    .section {{ margin:14px 0; }}
  </style>
</head>
<body>
  <div class='page'>
    <div class='header'>
      <h1>{title}</h1>
      <div class='sub'>Per-trade breakdown across periods + per-period top winners/losers.</div>
      <div class='summary-strip'>
        <div class='pill'># Trades: <b>{n_trades:,}</b></div>
        <div class='pill'>Win rate: <b>{win_rate*100:,.2f}%</b></div>
        <div class='pill'>Profit factor: <b>{_num(pf_total)}</b></div>
        <div class='pill'>Sum PnL (net): <b>{_money(sum_pnl)}</b></div>
        <!-- Sum costs removed -->
        <div class='pill'>Avg bars/trade: <b>{avg_bars:,.2f}</b></div>
      </div>
    </div>

    <div class='section grid2'>
      <div class='chart'>
        <div class='chart-title'>TOTAL trade return distribution</div>
        <img alt='Trade return histogram' src='{hist_uri}' />
      </div>
      <div class='card'><div class='box'>
        <h2>Breakdown: by period</h2>
        {_df_to_html(by_period, pct_cols=pct_cols, num_cols={"profit_factor"})}
      </div></div>
    </div>

    <div class='section grid2'>
      <div class='card'><div class='box'><h2>Breakdown: by holding duration (bars)</h2>
        {_df_to_html(by_duration, pct_cols=pct_cols, num_cols={"profit_factor"})}
      </div></div>
      <div class='card'><div class='box'><h2>Breakdown: by calendar month (aggregated)</h2>
        {_df_to_html(by_month, pct_cols=pct_cols, num_cols={"profit_factor"})}
      </div></div>
    </div>

    <div class='section'>
      <h2 style='margin:0 0 10px 0;'>Per-period drilldown</h2>
      {"\n".join(period_sections)}
    </div>

    <div class='section sub'>Generated by quantlab.reporting.generate_trades_report.report_periods_trades_html</div>
  </div>
</body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")
    return out_path

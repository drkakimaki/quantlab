from __future__ import annotations

import base64
import io
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import PerformanceSummary, performance_summary


@dataclass(frozen=True)
class ReportConfig:
    title: str = "Backtest Report"
    # Annualization frequency for metrics. Use 'MIN' for 1-minute bars.
    # (Some pandas versions accept 'min' but not legacy 'T'.)
    freq: str = "MIN"
    returns_col: str = "returns_net"
    equity_col: str = "equity"
    turnover_col: str = "turnover"
    costs_col: str = "costs"
    max_plot_points: int = 20_000  # downsample for very long intraday runs


def _downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step].copy()


def _fig_to_data_uri(fig) -> str:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(bio.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return float("nan")


def write_html_report(
    bt: pd.DataFrame,
    out_path: str | Path,
    config: ReportConfig | None = None,
) -> Path:
    """Write a lightweight single-file HTML report.

    - Embeds plots as base64 PNGs (no external assets).
    - Expects bt to include at least config.returns_col and config.equity_col.

    Returns the written path.
    """
    cfg = config or ReportConfig()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg.returns_col not in bt.columns or cfg.equity_col not in bt.columns:
        raise KeyError(
            f"bt must include columns {cfg.returns_col!r} and {cfg.equity_col!r}; got {list(bt.columns)!r}"
        )

    bt = bt.copy()
    bt.index = pd.to_datetime(bt.index)
    bt = bt.sort_index()

    r = bt[cfg.returns_col].astype(float)
    eq = bt[cfg.equity_col].astype(float)

    summ = performance_summary(r, eq, freq=cfg.freq)
    summ_d = {k: _safe_float(v) for k, v in asdict(summ).items()}

    # Drawdown
    peak = eq.cummax()
    dd = (eq / peak) - 1.0

    # Downsample only for plotting
    bt_plot = _downsample(bt, cfg.max_plot_points)
    r_plot = bt_plot[cfg.returns_col].astype(float)
    eq_plot = bt_plot[cfg.equity_col].astype(float)
    dd_plot = (eq_plot / eq_plot.cummax()) - 1.0

    # --- Plots ---
    # Equity
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(eq_plot.index, eq_plot.values, lw=1.2)
    ax.set_title("Equity")
    ax.grid(True, alpha=0.3)
    equity_uri = _fig_to_data_uri(fig)

    # Drawdown
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.fill_between(dd_plot.index, dd_plot.values, 0.0, alpha=0.25)
    ax.plot(dd_plot.index, dd_plot.values, lw=1.0)
    ax.set_title("Drawdown")
    ax.grid(True, alpha=0.3)
    dd_uri = _fig_to_data_uri(fig)

    # Return histogram
    fig, ax = plt.subplots(figsize=(6, 3))
    rr = r.dropna()
    if len(rr) > 0:
        ax.hist(rr.values, bins=60, density=True, alpha=0.85)
    ax.set_title("Returns (per bar) histogram")
    ax.grid(True, alpha=0.3)
    hist_uri = _fig_to_data_uri(fig)

    # Turnover/costs (optional)
    turnover_uri = None
    if cfg.turnover_col in bt_plot.columns:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.plot(bt_plot.index, bt_plot[cfg.turnover_col].astype(float).values, lw=1.0)
        ax.set_title("Turnover")
        ax.grid(True, alpha=0.3)
        turnover_uri = _fig_to_data_uri(fig)

    costs_uri = None
    if cfg.costs_col in bt_plot.columns:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.plot(bt_plot.index, bt_plot[cfg.costs_col].astype(float).values, lw=1.0)
        ax.set_title("Costs")
        ax.grid(True, alpha=0.3)
        costs_uri = _fig_to_data_uri(fig)

    start = str(bt.index.min()) if len(bt) else "n/a"
    end = str(bt.index.max()) if len(bt) else "n/a"

    def fmt(x: float) -> str:
        if np.isnan(x):
            return "nan"
        return f"{x:,.6f}"

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\"/>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
  <title>{cfg.title}</title>
  <style>
    :root {{ --fg:#111; --muted:#666; --bg:#fff; --card:#f6f7f9; --border:#e2e6ea; }}
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; color:var(--fg); background:var(--bg); margin:24px; }}
    h1 {{ margin: 0 0 6px 0; font-size: 22px; }}
    .sub {{ color:var(--muted); margin-bottom: 18px; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; margin: 12px 0 18px 0; }}
    .card {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:12px; }}
    .k {{ color:var(--muted); font-size: 12px; margin-bottom: 6px; }}
    .v {{ font-size: 18px; font-variant-numeric: tabular-nums; }}
    img {{ max-width: 100%; height: auto; border:1px solid var(--border); border-radius:10px; background:#fff; }}
    .section {{ margin: 18px 0; }}
    .small {{ font-size: 12px; color:var(--muted); }}
  </style>
</head>
<body>
  <h1>{cfg.title}</h1>
  <div class=\"sub\">Period: {start} → {end} · Bars: {len(bt):,} · Freq: {cfg.freq}</div>

  <div class=\"grid\">
    <div class=\"card\"><div class=\"k\">CAGR</div><div class=\"v\">{fmt(summ_d['cagr'])}</div></div>
    <div class=\"card\"><div class=\"k\">Vol</div><div class=\"v\">{fmt(summ_d['vol'])}</div></div>
    <div class=\"card\"><div class=\"k\">Sharpe</div><div class=\"v\">{fmt(summ_d['sharpe'])}</div></div>
    <div class=\"card\"><div class=\"k\">Max drawdown</div><div class=\"v\">{fmt(summ_d['max_drawdown'])}</div></div>
  </div>

  <div class=\"section\">
    <div class=\"small\">Equity & drawdown</div>
    <img alt=\"Equity\" src=\"{equity_uri}\" />
    <div style=\"height:10px\"></div>
    <img alt=\"Drawdown\" src=\"{dd_uri}\" />
  </div>

  <div class=\"section\">
    <div class=\"small\">Returns distribution</div>
    <img alt=\"Return histogram\" src=\"{hist_uri}\" />
  </div>

  {"" if turnover_uri is None else f"""<div class='section'><div class='small'>Turnover</div><img alt='Turnover' src='{turnover_uri}' /></div>"""}
  {"" if costs_uri is None else f"""<div class='section'><div class='small'>Costs</div><img alt='Costs' src='{costs_uri}' /></div>"""}

  <div class=\"section small\">
    Generated by quantlab.report.write_html_report (single-file HTML).
  </div>
</body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")
    return out_path


@dataclass(frozen=True)
class PeriodResult:
    bt: pd.DataFrame
    pnl: float
    n_trades: int


def write_html_report_periods(
    periods: dict[str, PeriodResult],
    out_path: str | Path,
    config: ReportConfig | None = None,
) -> Path:
    """Write a lightweight single-file HTML report for multiple test periods.

    periods: mapping period_name -> PeriodResult(bt, pnl, n_trades)

    Output:
    - summary table: PnL, MaxDD, Sharpe, Number of Trades
    - ONE plot per period: equity curve
    """
    cfg = config or ReportConfig()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, float, float, float, int]] = []  # name, pnl, maxdd, sharpe, n_trades
    charts: list[tuple[str, str]] = []  # (name, equity_uri)

    for name, res in periods.items():
        bt = res.bt
        if bt is None or len(bt) == 0:
            continue
        if cfg.returns_col not in bt.columns or cfg.equity_col not in bt.columns:
            raise KeyError(f"Period {name!r} missing required columns")

        bt = bt.copy()
        bt.index = pd.to_datetime(bt.index)
        bt = bt.sort_index()

        r = bt[cfg.returns_col].astype(float)
        eq = bt[cfg.equity_col].astype(float)
        summ = performance_summary(r, eq, freq=cfg.freq)

        rows.append((name, float(res.pnl), float(summ.max_drawdown), float(summ.sharpe), int(res.n_trades)))

        # Plot (downsample): equity only
        bt_plot = _downsample(bt, cfg.max_plot_points)
        eq_plot = bt_plot[cfg.equity_col].astype(float)

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(eq_plot.index, eq_plot.values, lw=1.2)
        ax.set_title(f"Equity — {name}")
        ax.grid(True, alpha=0.3)
        equity_uri = _fig_to_data_uri(fig)

        charts.append((name, equity_uri))

    def fmt(x: float) -> str:
        x = _safe_float(x)
        if np.isnan(x):
            return "nan"
        return f"{x:,.6f}"

    def fmt_money(x: float) -> str:
        x = _safe_float(x)
        if np.isnan(x):
            return "nan"
        return f"{x:,.2f}"

    # Table rows
    tr = []
    for name, pnl, maxdd, shrp, n_trades in rows:
        tr.append(
            f"<tr>"
            f"<td>{name}</td>"
            f"<td class='num'>{fmt_money(pnl)}</td>"
            f"<td class='num'>{fmt(maxdd)}</td>"
            f"<td class='num'>{fmt(shrp)}</td>"
            f"<td class='num'>{n_trades:,}</td>"
            f"</tr>"
        )

    charts_html = []
    for name, eq_uri in charts:
        charts_html.append(
            f"<div class='section'><img alt='Equity {name}' src='{eq_uri}' /></div>"
        )

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\"/>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
  <title>{cfg.title}</title>
  <style>
    :root {{ --fg:#111; --muted:#666; --bg:#fff; --card:#f6f7f9; --border:#e2e6ea; }}
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; color:var(--fg); background:var(--bg); margin:24px; }}
    h1 {{ margin: 0 0 6px 0; font-size: 22px; }}
    .sub {{ color:var(--muted); margin-bottom: 18px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid var(--border); padding: 8px 8px; text-align: left; }}
    th {{ font-size: 12px; color: var(--muted); }}
    td.num {{ font-variant-numeric: tabular-nums; text-align: right; }}
    td.small {{ font-size: 12px; color: var(--muted); }}
    .section {{ margin: 18px 0; }}
    img {{ max-width: 100%; height: auto; border:1px solid var(--border); border-radius:10px; background:#fff; }}
  </style>
</head>
<body>
  <h1>{cfg.title}</h1>
  <div class=\"sub\">Multi-period report · Bars are 1-minute · Freq(annualization): {cfg.freq}</div>

  <div class='section'>
    <table>
      <thead>
        <tr>
          <th>Period</th>
          <th class='num'>PnL (USD)</th>
          <th class='num'>Max Drawdown</th>
          <th class='num'>Sharpe</th>
          <th class='num'># Trades</th>
        </tr>
      </thead>
      <tbody>
        {''.join(tr)}
      </tbody>
    </table>
  </div>

  {''.join(charts_html)}

  <div class=\"section\" style=\"font-size:12px; color:{'var(--muted)'}\">Generated by quantlab.report.write_html_report_periods</div>
</body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")
    return out_path

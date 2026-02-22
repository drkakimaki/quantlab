from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..engine.metrics import (
    sharpe,
    avg_win_loss_from_position,
    profit_factor_from_position,
    win_rate_from_position,
    n_trades_from_position,
)


@dataclass(frozen=True)
class PeriodRow:
    period: str
    pnl: float
    max_drawdown: float
    sharpe: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    n_trades: int
    start: str
    end: str


def _fig_to_data_uri(fig) -> str:
    bio = io.BytesIO()
    # Slightly lower DPI keeps files smaller and helps 3-up layout.
    # Avoid bbox_inches='tight' here: it can introduce inconsistent padding/cropping
    # across periods (and makes charts feel "flattened").
    fig.savefig(bio, format="png", dpi=130)
    plt.close(fig)
    b64 = base64.b64encode(bio.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return float("nan")


# (trade metric helpers live in quantlab.engine.metrics)


def _fmt_ts(ts: pd.Timestamp) -> str:
    """Compact timestamp for report tables.

    Prefer readability over full ISO. Drop timezone suffix like "+00:00".
    """
    try:
        ts = pd.to_datetime(ts)
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert("UTC")
        return ts.strftime("%Y-%m-%d %H:%M")
    except Exception:  # noqa: BLE001
        return str(ts)


def report_periods_equity_only(
    *,
    periods: dict[str, pd.DataFrame],
    out_path: str | Path,
    title: str,
    initial_capital: float | dict[str, float],
    returns_col: str = "returns_net",
    equity_col: str = "equity",
    n_trades: dict[str, int] | None = None,
    win_rate: dict[str, float] | None = None,
    score_exclude: list[str] | set[str] | None = None,
) -> Path:
    """Single-file HTML report for multiple periods.

    Table columns (as requested): PnL, Max Drawdown, Sharpe, Number of Trades.
    One plot per period: equity curve.

    PnL is computed as a *percent return* (equity_end - 1) * 100.

    initial_capital can be:
    - float: same capital for every period
    - dict[str,float]: per-period capital (recommended when sizing is 1 lot and
      entry price differs per period).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[PeriodRow] = []
    charts: list[tuple[str, str | None]] = []

    for name, bt in periods.items():
        if bt is None or len(bt) == 0:
            # Keep the slot so the page always shows all periods.
            rows.append(
                PeriodRow(
                    period=name,
                    pnl=float("nan"),
                    max_drawdown=float("nan"),
                    sharpe=float("nan"),
                    win_rate=float("nan"),
                    profit_factor=float("nan"),
                    avg_win=float("nan"),
                    avg_loss=float("nan"),
                    n_trades=int((n_trades or {}).get(name, 0)),
                    start="",
                    end="",
                )
            )
            charts.append((name, None))
            continue
        if returns_col not in bt.columns or equity_col not in bt.columns:
            raise KeyError(f"Period {name!r} missing required columns")

        bt = bt.copy()
        bt.index = pd.to_datetime(bt.index)
        bt = bt.sort_index()

        r = bt[returns_col].astype(float)
        eq = bt[equity_col].astype(float)

        cap0 = float(initial_capital[name] if isinstance(initial_capital, dict) else initial_capital)
        if cap0 <= 0:
            cap0 = float(eq.iloc[0]) if len(eq) else 1.0

        # Percent return for the period (relative to starting capital)
        pnl = (float(eq.iloc[-1]) / cap0 - 1.0) * 100.0

        peak = eq.cummax()
        dd = (eq / peak) - 1.0
        max_dd = float(dd.min()) * 100.0

        # Canonical Sharpe: computed on daily returns derived from equity.
        s = float(sharpe(eq))
        if n_trades is None:
            trades = n_trades_from_position(bt, pos_col="position")
        else:
            trades = int(n_trades.get(name, 0))

        wr = float(win_rate.get(name)) if (win_rate is not None and name in win_rate) else win_rate_from_position(bt)
        pf = profit_factor_from_position(bt, pos_col="position", returns_col=returns_col)
        avg_win, avg_loss = avg_win_loss_from_position(bt, pos_col="position", returns_col=returns_col)

        rows.append(
            PeriodRow(
                period=name,
                pnl=pnl,
                max_drawdown=max_dd,
                sharpe=s,
                win_rate=wr,
                profit_factor=pf,
                avg_win=avg_win,
                avg_loss=avg_loss,
                n_trades=trades,
                start=_fmt_ts(bt.index.min()),
                end=_fmt_ts(bt.index.max()),
            )
        )

        # Plot (compact). Reduce x-axis label clutter using ConciseDateFormatter.
        fig, ax = plt.subplots(figsize=(4.9, 3.6))
        ax.plot(eq.index, eq.values, lw=1.1)
        # Title is rendered in HTML (chart-title). Avoid matplotlib title to reduce top whitespace.
        ax.grid(True, alpha=0.25)

        # Y zoom: matplotlib autoscale + margins can still feel "too zoomed" on flat-ish periods
        # because the y-range is tiny. We enforce a minimum absolute padding based on equity level.
        ax.margins(x=0.01, y=0.05)
        try:
            y0 = float(np.nanmin(eq.values))
            y1 = float(np.nanmax(eq.values))
            if np.isfinite(y0) and np.isfinite(y1):
                center = 0.5 * (y0 + y1)
                span = y1 - y0
                # pad is max of:
                # - relative-to-span padding (for normal periods)
                # - relative-to-level padding (for flat-ish periods)
                # - an absolute floor (for small numbers)
                pad = max(0.20 * span, 0.02 * abs(center), 1.0)
                ax.set_ylim(y0 - pad, y1 + pad)
        except Exception:
            pass

        # Give y-axis labels more room (avoid clipping/squishing).
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        fig.subplots_adjust(left=0.16, right=0.99, top=0.995, bottom=0.14)

        try:
            import matplotlib.dates as mdates

            loc = mdates.AutoDateLocator(minticks=3, maxticks=5)
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        except Exception:
            pass

        charts.append((name, _fig_to_data_uri(fig)))

    def pct(x: float) -> str:
        x = _safe_float(x)
        if np.isnan(x):
            return "nan"
        return f"{x:,.2f}%"

    def num(x: float) -> str:
        x = _safe_float(x)
        if np.isnan(x):
            return "nan"
        return f"{x:,.2f}"

    def pf(x: float) -> str:
        x = _safe_float(x)
        if np.isnan(x):
            return "nan"
        if np.isinf(x):
            return "inf"
        return f"{x:,.2f}"

    tr = []
    for r in rows:
        tr.append(
            "<tr>"
            f"<td class='period'>{r.period}</td>"
            f"<td class='num mono'>{pct(r.pnl)}</td>"
            f"<td class='num mono'>{pct(r.max_drawdown)}</td>"
            f"<td class='num mono'>{num(r.sharpe)}</td>"
            f"<td class='num mono'>{pct(r.win_rate)}</td>"
            f"<td class='num mono'>{pf(r.profit_factor)}</td>"
            f"<td class='num mono'>{pct(r.avg_win)}</td>"
            f"<td class='num mono'>{pct(r.avg_loss)}</td>"
            f"<td class='num mono'>{r.n_trades}</td>"
            "</tr>"
        )

    # Top summary strip (scored periods only, if score_exclude is set)
    exclude = set(score_exclude or [])
    scored_rows = [r for r in rows if r.period not in exclude]

    total_trades = int(sum((r.n_trades or 0) for r in scored_rows))
    # Worst drawdown (most negative)
    worst_dd = float(min((r.max_drawdown for r in scored_rows if np.isfinite(r.max_drawdown)), default=float("nan")))
    # Average sharpe across periods (simple mean)
    sh_list = [r.sharpe for r in scored_rows if np.isfinite(r.sharpe)]
    avg_sharpe = float(np.mean(sh_list)) if sh_list else float("nan")

    holdout_note = ""
    if exclude:
        holdout_note = f"Holdout excluded from header stats: {sorted(exclude)}"

    # Charts HTML with show-all toggle
    charts_html = [
        "<div class='charts-head'>"
        "<button id='toggle-charts' class='btn-small' type='button'>Show all charts</button>"
        "</div>",
        "<div class='charts' id='charts'>",
    ]

    for i, (name, uri) in enumerate(charts):
        hidden_class = " hidden" if i >= 3 else ""
        if uri is None:
            charts_html.append(
                "".join(
                    [
                        "<div class='chart{hc}'>".format(hc=hidden_class),
                        f"<div class='chart-title'>Equity — {name}</div>",
                        "<div class='chart-missing'>No data for this period</div>",
                        "</div>",
                    ]
                )
            )
        else:
            charts_html.append(
                "".join(
                    [
                        "<div class='chart{hc}'>".format(hc=hidden_class),
                        f"<div class='chart-title'>Equity — {name}</div>",
                        f"<img alt='Equity {name}' src='{uri}' />",
                        "</div>",
                    ]
                )
            )

    charts_html.append("</div>")

    # (lightbox removed)

    archetype = title.split("(", 1)[0].strip() if title else "Report"
    parts = title.split(" + ") if title else []
    base_line = parts[0] if parts else title
    hyper = parts[1:] if len(parts) > 1 else []

    hyper_items = "".join([f"<li><span class='mono'>{p}</span></li>" for p in hyper])

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\"/>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
  <title>{archetype}</title>
  <style>
    :root {{
      /* Dark mode defaults */
      --fg:#e5e7eb;
      --muted:#9ca3af;
      --bg:#0b1220;
      --border:#223047;
      --card:#111a2b;
      --card2:#0f172a;
      --shadow: 0 18px 40px rgba(0, 0, 0, 0.45);
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      color:var(--fg);
      background:var(--bg);
      margin: 0;
      line-height: 1.45;
    }}

    .page {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 26px 18px 34px;
    }}

    .header {{
      background: radial-gradient(900px 420px at 20% -20%, rgba(96,165,250,0.18), rgba(0,0,0,0) 55%),
                  linear-gradient(180deg, rgba(17,26,43,0.98), rgba(15,23,42,0.92));
      border: 1px solid var(--border);
      border-radius: 20px;
      box-shadow: var(--shadow);
      padding: 18px 18px 16px;
      margin-bottom: 14px;
    }}

    h1 {{ margin: 0 0 6px 0; font-size: 30px; letter-spacing: -0.6px; }}
    .sub {{ color:var(--muted); font-size: 12px; }}
    .subtitle {{ color:var(--muted); font-size: 13px; margin-top: 2px; }}

    .section {{ margin: 14px 0; }}

    .summary-strip {{ display:flex; gap:10px; flex-wrap:wrap; margin-top: 10px; }}
    .pill {{ background: var(--card2); border:1px solid var(--border); border-radius: 999px; padding: 8px 10px; font-size: 12px; color: var(--muted); }}
    .pill b {{ color: var(--fg); font-weight: 700; font-variant-numeric: tabular-nums; }}

    .charts-head {{ display:flex; justify-content: space-between; align-items: center; gap: 12px; margin: 10px 0; }}
    .btn-small {{ font-size: 12px; padding: 6px 10px; border-radius: 10px; border:1px solid var(--border); background: var(--card2); color: var(--fg); cursor:pointer; }}
    .btn-small:hover {{ filter: brightness(1.08); }}

    .hidden {{ display:none; }}

    /* lightbox removed */

    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}

    .grid {{ display: grid; grid-template-columns: 1fr; gap: 12px; }}
    @media (min-width: 900px) {{ .grid {{ grid-template-columns: 1.2fr 0.8fr; }} }}

    .box {{ padding: 14px 14px; }}
    .box h2 {{ margin: 0 0 10px 0; font-size: 14px; color: var(--fg); letter-spacing: -0.2px; }}

    .kv {{
      background: var(--card2);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 10px 10px;
    }}
    .kv .mono {{ font-family: var(--mono); font-size: 12px; line-height: 1.45; }}
    ul.hyper {{ margin: 8px 0 0 18px; padding: 0; }}
    ul.hyper li {{ margin: 4px 0; }}

    table {{ border-collapse: separate; border-spacing: 0; width: 100%; }}
    th, td {{ border-bottom: 1px solid var(--border); padding: 10px 12px; text-align: left; }}
    th {{
      font-size: 12px;
      color: var(--muted);
      background: var(--card2);
      position: sticky;
      top: 0;
      z-index: 1;
    }}

    /* No zebra striping; keep uniform row background */
    tbody td {{ background: transparent; }}
    tbody tr:hover td {{ background: rgba(96,165,250,0.10); }}

    td.num {{ font-variant-numeric: tabular-nums; text-align: right; }}
    td.small, .small {{ font-size: 12px; color: var(--muted); }}
    .mono {{ font-family: var(--mono); font-variant-numeric: tabular-nums; }}
    td.period {{ font-weight: 600; }}

    /* Charts row: force 3 charts horizontally; allow horizontal scroll on small screens */
    .charts {{ display: flex; gap: 14px; flex-wrap: nowrap; align-items: stretch; overflow-x: auto; padding-bottom: 6px; }}
    .chart {{
      flex: 0 0 calc(33.333% - 10px);
      border: 1px solid var(--border);
      border-radius: 18px;
      background: var(--card);
      box-shadow: var(--shadow);
      padding: 12px;
      min-width: 340px;
    }}
    .chart-title {{ font-size: 12px; color: var(--muted); margin: 0 0 8px 2px; }}
    .chart img {{
      width: 100%;
      height: auto;
      max-height: 520px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--card2);
      display: block;
    }}
    .chart-missing {{
      height: 520px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      background: var(--card2);
      border: 1px dashed var(--border);
      border-radius: 12px;
      font-size: 12px;
    }}

    .footer {{ margin-top: 10px; }}
  </style>
</head>
<body>
  <div class='page'>
    <div class='header'>
      <h1>{archetype}</h1>
      <div class='subtitle'>{base_line}</div>
      <div class='summary-strip'>
        <div class='pill'># Trades: <b>{total_trades:,}</b></div>
        <div class='pill'>Avg Sharpe: <b>{num(avg_sharpe)}</b></div>
        <div class='pill'>Worst MaxDD: <b>{pct(worst_dd)}</b></div>
      </div>
      <div class='sub'>{holdout_note}</div>
    </div>

    <div class='section grid'>
      <div class='card'>
        <div class='box'>
          <h2>Performance summary</h2>
          <table id='perf-table'>
            <thead>
              <tr>
                <th data-sort='text'>Period</th>
                <th class='num' data-sort='num'>PnL</th>
                <th class='num' data-sort='num'>Max DD</th>
                <th class='num' data-sort='num'>Sharpe</th>
                <th class='num' data-sort='num'>Win Rate</th>
                <th class='num' data-sort='num'>Profit Factor</th>
                <th class='num' data-sort='num'>Avg Win</th>
                <th class='num' data-sort='num'>Avg Loss</th>
                <th class='num' data-sort='num'># Trades</th>
              </tr>
            </thead>
            <tbody>
              {''.join(tr)}
            </tbody>
          </table>
          <div class='small' style='padding:10px 12px;'>Tip: click headers to sort.</div>
        </div>
      </div>

      <div class='card'>
        <div class='box'>
          <h2>Hyperparameters / modules</h2>
          <div class='kv'>
            <div class='mono'>{archetype}</div>
            <div class='mono' style='margin-top:6px; color: var(--muted);'>Modules:</div>
            <ul class='hyper'>
              {hyper_items}
            </ul>
          </div>
        </div>
      </div>
    </div>

    <div class='section'>
      {''.join(charts_html)}
    </div>

    <script>
      // Sortable table
      (function() {{
        const table = document.getElementById('perf-table');
        if (!table) return;
        const tbody = table.querySelector('tbody');
        const getCell = (tr, idx) => tr.children[idx].innerText.trim();
        const parseNum = (s) => {{
          // handles 'nan', 'inf', '12.34%', '1,234.56'
          if (!s) return Number.NEGATIVE_INFINITY;
          const x = s.replace(/%/g,'').replace(/,/g,'');
          const v = parseFloat(x);
          return Number.isFinite(v) ? v : Number.NEGATIVE_INFINITY;
        }};

        let sortState = {{ idx: -1, asc: false }};

        table.querySelectorAll('thead th').forEach((th, idx) => {{
          th.style.cursor = 'pointer';
          th.addEventListener('click', () => {{
            const type = th.getAttribute('data-sort') || 'text';
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const asc = (sortState.idx === idx) ? !sortState.asc : false;
            sortState = {{ idx, asc }};

            rows.sort((a,b) => {{
              const A = getCell(a, idx);
              const B = getCell(b, idx);
              if (type === 'num') {{
                return (parseNum(A) - parseNum(B)) * (asc ? 1 : -1);
              }}
              return A.localeCompare(B) * (asc ? 1 : -1);
            }});

            rows.forEach(r => tbody.appendChild(r));
          }});
        }});
      }})();

      // Charts show-all toggle
      (function() {{
        const btn = document.getElementById('toggle-charts');
        const charts = document.getElementById('charts');
        if (!btn || !charts) return;
        btn.addEventListener('click', () => {{
          const hidden = charts.querySelectorAll('.chart.hidden');
          const anyHidden = hidden.length > 0;
          hidden.forEach(el => el.classList.remove('hidden'));
          if (anyHidden) {{
            btn.textContent = 'Hide extra charts';
            btn.dataset.state = 'shown';
          }} else {{
            // hide all after first 3
            const all = charts.querySelectorAll('.chart');
            all.forEach((el, i) => {{ if (i >= 3) el.classList.add('hidden'); }});
            btn.textContent = 'Show all charts';
            btn.dataset.state = 'hidden';
          }}
        }});
      }})();

      // (lightbox removed)
    </script>

    <div class='footer small'>Generated by quantlab.generate_bt_report.report_periods_equity_only</div>
  </div>
</body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")
    return out_path

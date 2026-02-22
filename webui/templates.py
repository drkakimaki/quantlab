"""HTML templates for the web UI."""

from .config import get_strategies
from .runner import report_exists, report_mtime


def render_index() -> str:
    """Render the main page."""
    strategies = get_strategies()
    strategy_options = "\n".join(
        f'<option value="{sid}">{s.name}</option>'
        for sid, s in strategies.items()
    )
    
    existing_reports = _render_existing_reports(strategies)
    
    # Embed costs for UI display
    import yaml
    import json
    from pathlib import Path

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "trend_based" / "current.yaml"
    costs = {}
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        costs = cfg.get("costs", {}) or {}
    except Exception:
        costs = {}

    return HTML_TEMPLATE.format(
        strategy_options=strategy_options,
        existing_reports=existing_reports,
        costs_json=json.dumps(costs),
    )


def _fmt_mtime(ts: float | None) -> str:
    if ts is None:
        return ""
    from datetime import datetime

    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


def _render_existing_reports(strategies: dict) -> str:
    """Render the existing reports list."""
    items = []

    for sid, s in strategies.items():
        if report_exists(sid):
            url = f"/report/{sid}"
            mt = _fmt_mtime(report_mtime(sid))

            links = [f'<a href="{url}" target="_blank" rel="noopener">Open</a>']

            # Trade breakdown report (if present)
            if report_exists(sid, kind="trades"):
                url_t = f"/trades/{sid}"
                links.append(f'<a href="{url_t}" target="_blank" rel="noopener">Trades</a>')

            if report_exists(sid, variant="yearly"):
                url_y = f"/report/{sid}?mode=yearly"
                links.append(f'<a href="{url_y}" target="_blank" rel="noopener">Open (Y)</a>')

            if report_exists(sid, variant="yearly", kind="trades"):
                url_ty = f"/trades/{sid}?mode=yearly"
                links.append(f'<a href="{url_ty}" target="_blank" rel="noopener">Trades (Y)</a>')

            items.append(
                f'<div class="strategy-item">'
                f'<div class="left"><span class="name">{s.name}</span>'
                f'<span class="meta">{mt}</span></div>'
                f'<div class="links">' + " ".join(links) + '</div>'
                f'</div>'
            )

    if not items:
        return "<p>No reports yet. Run a backtest first.</p>"

    return "\n".join(items)


def render_result(success: bool, output: str, report_url: str | None) -> str:
    """Render the result fragment for AJAX response."""
    import html

    if success:
        report_url = report_url or "#"
        result_html = (
            f'<div data-status="success"></div>'
            f'<a href="{report_url}" class="result-link" target="_blank" rel="noopener">Open Report</a>'
        )
    else:
        result_html = (
            f'<div data-status="error"></div>'
            f'<p style="color:var(--error);">Backtest failed. Check output above.</p>'
        )

    return f"{html.escape(output)}\n---RESULT---{result_html}"


# --- HTML Template ---

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Quantlab Backtest UI</title>
  <style>
    :root {{
      /* Dark mode defaults */
      --fg:#e5e7eb;
      --muted:#9ca3af;
      --bg:#0b1220;
      --border:#243044;
      --card:#111a2b;
      --card2:#0f172a;
      --accent:#60a5fa;
      --accent-hover:#93c5fd;
      --success:#34d399;
      --error:#f87171;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      color: var(--fg);
      background: var(--bg);
      margin: 0;
      padding: 24px;
      line-height: 1.5;
    }}
    .container {{ max-width: 1120px; margin: 0 auto; }}

    .layout {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
      align-items: start;
    }}
    @media (min-width: 980px) {{
      .layout {{
        grid-template-columns: 1.05fr 0.95fr;
      }}
    }}
    h1 {{ margin: 0 0 8px 0; font-size: 24px; }}
    .sub {{ color: var(--muted); margin-bottom: 24px; }}
    
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 12px 30px rgba(0,0,0,0.35);
    }}
    
    label {{ display: block; font-weight: 500; margin-bottom: 8px; }}
    select {{
      width: 100%;
      padding: 10px 12px;
      font-size: 15px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--card2);
      color: var(--fg);
      cursor: pointer;
    }}
    select:focus {{ outline: none; border-color: var(--accent); }}

    .row {{ margin-top: 10px; }}
    label.check {{ display:flex; align-items:center; gap:10px; font-weight:500; cursor:pointer; }}
    label.check input {{ width: 16px; height: 16px; }}
    .note {{ margin-top: 10px; font-size: 12px; color: var(--muted); }}

    button {{
      width: 100%;
      padding: 12px 16px;
      font-size: 15px;
      font-weight: 600;
      color: #07101d;
      background: var(--accent);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 10px;
      cursor: pointer;
      margin-top: 16px;
    }}
    button:hover {{ filter: brightness(1.05); }}
    button:disabled {{ background: #334155; color: #94a3b8; cursor: not-allowed; }}
    
    .status {{
      margin-top: 16px;
      padding: 12px;
      border-radius: 8px;
      font-size: 14px;
    }}
    .status.running {{ background: #eff6ff; color: var(--accent); }}
    .status.success {{ background: #ecfdf5; color: var(--success); }}
    .status.error {{ background: #fef2f2; color: var(--error); }}
    
    .output {{
      margin-top: 16px;
      background: #070b13;
      color: #a7f3d0;
      border: 1px solid rgba(255,255,255,0.08);
      padding: 12px;
      border-radius: 10px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      max-height: 300px;
      overflow: auto;
      white-space: pre-wrap;
      display: none;
    }}
    .output.visible {{ display: block; }}
    
    .result-link {{
      display: inline-block;
      margin-top: 12px;
      padding: 8px 16px;
      background: var(--success);
      color: white;
      text-decoration: none;
      border-radius: 6px;
      font-weight: 500;
    }}
    .result-link:hover {{ background: #047857; }}
    
    .strategies h2 {{ font-size: 16px; margin: 0 0 12px 0; }}
    .strategy-list {{ display: grid; gap: 8px; }}
    .strategy-item {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 10px 12px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    .strategy-item .left {{ display:flex; flex-direction:column; gap:2px; min-width:0; }}
    .strategy-item .name {{ font-size: 14px; font-weight: 500; }}
    .strategy-item .meta {{ font-size: 12px; color: var(--muted); }}
    .strategy-item .links {{ display:flex; gap: 10px; flex-wrap:wrap; justify-content:flex-end; }}
    .strategy-item a {{
      font-size: 13px;
      color: var(--accent);
      text-decoration: none;
    }}
    .strategy-item a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>📊 Quantlab Backtest</h1>
    <div class="sub">Run backtests and view reports</div>

    <div class="layout">
      <!-- LEFT: generator -->
      <div class="card">
        <form id="backtest-form" method="post" action="/run">
          <label for="strategy">Select Strategy</label>
          <select name="strategy" id="strategy">
            {strategy_options}
          </select>

          <div style="height:12px"></div>

          <label for="breakdown">Time setting</label>
          <select name="breakdown" id="breakdown">
            <option value="">Config default</option>
            <option value="three_block">3 blocks (20–22 / 23–25 / 26)</option>
            <option value="yearly">Yearly (2021+)</option>
          </select>

          <!-- record_executions checkbox removed (trade report is always generated) -->

          <div class="note" id="costs-note"></div>

          <button type="submit" id="run-btn">Run Backtest</button>
        </form>

        <div id="status" class="status" style="display:none;"></div>
        <div id="output" class="output"></div>
        <div id="result"></div>
      </div>

      <!-- RIGHT: existing reports -->
      <div class="strategies card">
        <h2>Existing Reports</h2>
        <div class="strategy-list">
          {existing_reports}
        </div>
      </div>
    </div>
  </div>

  <script>
    // Show current costs from config (embedded server-side at render time)
    const COSTS = {costs_json};
    const costsNote = document.getElementById('costs-note');
    if (COSTS && (COSTS.fee_per_lot || COSTS.spread_per_lot)) {{
      costsNote.textContent = `Costs: fee_per_lot=${{COSTS.fee_per_lot || 0}}, spread_per_lot=${{COSTS.spread_per_lot || 0}}`;
    }} else {{
      costsNote.textContent = 'Costs: fee_per_lot=0, spread_per_lot=0';
    }}

    document.getElementById('backtest-form').addEventListener('submit', async function(e) {{
      e.preventDefault();
      
      const btn = document.getElementById('run-btn');
      const status = document.getElementById('status');
      const output = document.getElementById('output');
      const result = document.getElementById('result');
      const strategy = document.getElementById('strategy').value;
      const breakdown = document.getElementById('breakdown').value;
      const recordExecutions = false;

      const t0 = Date.now();

      btn.disabled = true;
      btn.textContent = 'Running...';
      status.style.display = 'block';
      status.className = 'status running';
      status.textContent = 'Running backtest... (0s)';
      const tick = setInterval(() => {{
        const dt = Math.floor((Date.now() - t0) / 1000);
        status.textContent = 'Running backtest... (' + dt + 's)';
      }}, 500);
      output.style.display = 'none';
      output.textContent = '';
      result.innerHTML = '';
      
      try {{
        const resp = await fetch('/run', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/x-www-form-urlencoded' }},
          body: [
            'strategy=' + encodeURIComponent(strategy),
            'breakdown=' + encodeURIComponent(breakdown),
            'record_executions=0',
          ].join('&')
        }});
        
        const text = await resp.text();
        
        // Parse the response (output + result link)
        const parts = text.split('---RESULT---');
        const outputText = parts[0] || '';
        const resultHtml = parts[1] || '';
        
        output.textContent = outputText;
        output.style.display = 'block';
        output.classList.add('visible');
        
        clearInterval(tick);
        const dt = Math.floor((Date.now() - t0) / 1000);

        if (resultHtml.includes('data-status="success"')) {{
          status.className = 'status success';
          status.textContent = '✓ Done in ' + dt + 's';
          result.innerHTML = resultHtml;
        }} else {{
          status.className = 'status error';
          status.textContent = '✗ Failed (' + dt + 's)';
          result.innerHTML = resultHtml;
        }}
      }} catch (err) {{
        clearInterval(tick);
        status.className = 'status error';
        status.textContent = '✗ Error: ' + err.message;
      }}
      
      btn.disabled = false;
      btn.textContent = 'Run Backtest';
    }});
  </script>
</body>
</html>
"""

#!/usr/bin/env python3
"""
Serve the best_trend HTML report with a /regenerate endpoint.

Usage:
    python quantlab/scripts/serve_report.py [--port 8080] [--config quantlab/configs/trend_based/current.yaml]

Then open http://localhost:8080 in your browser.
Click "Regenerate" to rebuild the report.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from threading import Thread

# Try to use aiohttp for async streaming, fall back to http.server
try:
    from aiohttp import web
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    import urllib.parse


# Workspace root
WORKSPACE = Path(__file__).resolve().parents[2]
REPORT_PATH = WORKSPACE / "quantlab" / "reports" / "trend_based" / "best_trend.html"
CONFIG_DEFAULT = WORKSPACE / "quantlab" / "configs" / "trend_based" / "current.yaml"
FOMC_DAYS = WORKSPACE / "quantlab" / "data" / "econ_calendar" / "fomc_decision_days.csv"

# Global state
regenerate_status = {
    "running": False,
    "last_run": None,
    "last_output": "",
    "error": None
}


def run_regenerate(config_path: Path) -> tuple[bool, str]:
    """Run the report regeneration. Returns (success, output)."""
    import subprocess
    import sys
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE)
    
    cmd = [
        sys.executable,
        str(WORKSPACE / "quantlab" / "scripts" / "report_trend_variants.py"),
        "--config", str(config_path),
        "--fomc-days", str(FOMC_DAYS),
        "--out-dir", str(WORKSPACE / "quantlab" / "reports" / "trend_based"),
        "--out-name", "best_trend.html"
    ]
    
    result = subprocess.run(
        cmd,
        cwd=str(WORKSPACE),
        env=env,
        capture_output=True,
        text=True,
        timeout=300
    )
    
    output = result.stdout + result.stderr
    success = result.returncode == 0
    return success, output


if HAS_AIOHTTP:
    # Async version with streaming
    async def handle_index(request: web.Request) -> web.Response:
        """Serve the HTML report with regenerate button injected."""
        if not REPORT_PATH.exists():
            return web.Response(text="Report not found. Click Regenerate to create it.", content_type="text/html")
        
        html = REPORT_PATH.read_text()
        
        # Inject the regenerate button and script
        inject = """
    <div id="regen-controls" style="position:fixed;top:12px;right:12px;z-index:1000;background:var(--card);border:1px solid var(--border);border-radius:8px;padding:8px 12px;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
      <button id="regen-btn" onclick="regenerate()" style="cursor:pointer;background:#2563eb;color:white;border:none;padding:6px 14px;border-radius:6px;font-size:13px;font-weight:500;">
        Regenerate
      </button>
      <span id="regen-status" style="margin-left:10px;font-size:12px;color:var(--muted);"></span>
    </div>
    <pre id="regen-output" style="display:none;position:fixed;bottom:12px;right:12px;width:400px;max-height:200px;overflow:auto;background:#1e1e1e;color:#0f0;font-size:11px;padding:10px;border-radius:8px;z-index:1000;"></pre>
    <script>
    async function regenerate() {
      const btn = document.getElementById('regen-btn');
      const status = document.getElementById('regen-status');
      const output = document.getElementById('regen-output');
      
      btn.disabled = true;
      btn.textContent = 'Running...';
      status.textContent = 'Regenerating...';
      output.style.display = 'block';
      output.textContent = '';
      
      try {
        const resp = await fetch('/regenerate', { method: 'POST' });
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const text = decoder.decode(value);
          output.textContent += text;
          output.scrollTop = output.scrollHeight;
        }
        
        const result = JSON.parse(output.textContent.split('\\n').pop());
        if (result.success) {
          status.textContent = 'Done! Refreshing...';
          setTimeout(() => location.reload(), 500);
        } else {
          status.textContent = 'Error: ' + (result.error || 'unknown');
        }
      } catch (e) {
        status.textContent = 'Error: ' + e.message;
      }
      
      btn.disabled = false;
      btn.textContent = 'Regenerate';
    }
    </script>
"""
        # Insert before </body>
        html = html.replace("</body>", inject + "</body>")
        return web.Response(text=html, content_type="text/html")
    
    async def handle_regenerate(request: web.Request) -> web.StreamResponse:
        """Regenerate the report and stream output."""
        response = web.StreamResponse()
        response.headers["Content-Type"] = "text/plain; charset=utf-8"
        await response.prepare(request)
        
        def run():
            success, output = run_regenerate(CONFIG_DEFAULT)
            return success, output
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        success, output = await loop.run_in_executor(None, run)
        
        # Send output
        await response.write(output.encode())
        await response.write(f"\n\n{json.dumps({'success': success})}\n".encode())
        
        return response
    
    def main_async():
        app = web.Application()
        app.router.add_get("/", handle_index)
        app.router.add_post("/regenerate", handle_regenerate)
        web.run_app(app, port=args.port)
    
else:
    # Fallback sync version
    class ReportHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self.serve_report()
            else:
                super().do_GET()
        
        def do_POST(self):
            if self.path == "/regenerate":
                self.handle_regenerate()
            else:
                self.send_error(404)
        
        def serve_report(self):
            if not REPORT_PATH.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<h1>Report not found</h1><p>Click Regenerate to create it.</p>")
                return
            
            html = REPORT_PATH.read_text()
            
            # Inject regenerate button
            inject = """
    <div id="regen-controls" style="position:fixed;top:12px;right:12px;z-index:1000;background:var(--card);border:1px solid var(--border);border-radius:8px;padding:8px 12px;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
      <form action="/regenerate" method="post" style="display:inline;">
        <button type="submit" style="cursor:pointer;background:#2563eb;color:white;border:none;padding:6px 14px;border-radius:6px;font-size:13px;font-weight:500;">
          Regenerate
        </button>
      </form>
      <span id="regen-status" style="margin-left:10px;font-size:12px;color:var(--muted);"></span>
    </div>
"""
            html = html.replace("</body>", inject + "</body>")
            
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())
        
        def handle_regenerate(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            
            self.wfile.write(b"<html><body><h1>Regenerating...</h1><pre>")
            self.wfile.flush()
            
            success, output = run_regenerate(CONFIG_DEFAULT)
            
            # Escape HTML in output
            import html as html_mod
            self.wfile.write(html_mod.escape(output).encode())
            self.wfile.flush()
            
            if success:
                self.wfile.write(b"</pre><p style='color:green'>Success! <a href='/'>View report</a></p></body></html>")
            else:
                self.wfile.write(b"</pre><p style='color:red'>Failed. <a href='/'>Back</a></p></body></html>")
    
    def main_sync():
        server = HTTPServer(("", args.port), ReportHandler)
        print(f"Serving at http://localhost:{args.port}")
        server.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Serve report with regenerate endpoint")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--config", type=str, default=str(CONFIG_DEFAULT))
    args = ap.parse_args()
    
    CONFIG_DEFAULT = Path(args.config) if args.config else CONFIG_DEFAULT
    
    print(f"Report path: {REPORT_PATH}")
    print(f"Config: {CONFIG_DEFAULT}")
    
    if HAS_AIOHTTP:
        print("Using aiohttp (async with streaming)")
        main_async()
    else:
        print("Using http.server (sync fallback)")
        main_sync()
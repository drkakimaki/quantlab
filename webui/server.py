"""HTTP server for the backtest web UI."""

import argparse
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

from .config import get_strategies
from .runner import run_backtest, get_report_path
from .templates import render_index, render_result


class BacktestHandler(BaseHTTPRequestHandler):
    """HTTP handler for backtest UI."""
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/" or self.path == "/index.html":
            self._serve_html(render_index())
        elif self.path.startswith("/report/"):
            self._serve_report(kind="equity")
        elif self.path.startswith("/trades/"):
            self._serve_report(kind="trades")
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/run":
            self._handle_run()
        else:
            self.send_error(404)
    
    def _serve_html(self, html: str):
        """Serve HTML content."""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())
    
    def _serve_report(self, *, kind: str = "equity"):
        """Serve a report file."""
        parsed = urllib.parse.urlparse(self.path)
        strategy_id = parsed.path.split("/")[-1]
        qs = urllib.parse.parse_qs(parsed.query)
        mode = (qs.get("mode", [""])[0] or "").strip().lower()
        variant = "yearly" if mode in {"yearly", "y"} else None

        path = get_report_path(strategy_id, variant=variant, kind=kind)

        if path is None:
            self.send_error(404, "Unknown strategy")
        elif not path.exists():
            self.send_error(404, "Report not found")
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(path.read_bytes())
    
    def _handle_run(self):
        """Handle backtest run request."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode()
        params = urllib.parse.parse_qs(body)
        strategy_id = params.get("strategy", [""])[0]
        
        strategies = get_strategies()
        if strategy_id not in strategies:
            self.send_response(400)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Invalid strategy")
            return
        
        breakdown = params.get("breakdown", [""])[0] or None
        record_exec = params.get("record_executions", [""])[0] in {"1", "true", "on", "yes"}

        success, output, report_path = run_backtest(
            strategy_id,
            breakdown=breakdown,
            record_executions=record_exec,
        )
        # Point the user to the correct report variant
        if report_path.name.endswith("_y" + report_path.suffix):
            report_url = f"/report/{strategy_id}?mode=yearly"
        else:
            report_url = f"/report/{strategy_id}"

        response = render_result(success, output, report_url)
        
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(response.encode())


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    # Allow Ctrl+C / SIGTERM to exit even if a request handler is busy.
    daemon_threads = True
    allow_reuse_address = True


def main():
    """Run the backtest UI server."""
    ap = argparse.ArgumentParser(description="Backtest web UI")
    ap.add_argument("--port", type=int, default=8080, help="Port to listen on")
    ap.add_argument("--host", default="", help="Host to bind (default: all interfaces)")
    args = ap.parse_args()

    strategies = get_strategies()
    print(f"Starting backtest UI at http://localhost:{args.port}")
    print(f"Strategies: {list(strategies.keys())}")

    server = ThreadingHTTPServer((args.host, args.port), BacktestHandler)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            server.shutdown()
        except Exception:
            pass
        server.server_close()


if __name__ == "__main__":
    main()

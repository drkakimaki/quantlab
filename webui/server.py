"""HTTP server for the backtest web UI."""

import argparse
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

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
            self._serve_report()
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
    
    def _serve_report(self):
        """Serve a report file."""
        strategy_id = self.path.split("/")[-1]
        path = get_report_path(strategy_id)
        
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
        response = render_result(success, output, strategy_id)
        
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(response.encode())


def main():
    """Run the backtest UI server."""
    ap = argparse.ArgumentParser(description="Backtest web UI")
    ap.add_argument("--port", type=int, default=8080, help="Port to listen on")
    ap.add_argument("--host", default="", help="Host to bind (default: all interfaces)")
    args = ap.parse_args()
    
    strategies = get_strategies()
    print(f"Starting backtest UI at http://localhost:{args.port}")
    print(f"Strategies: {list(strategies.keys())}")
    
    server = HTTPServer((args.host, args.port), BacktestHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()

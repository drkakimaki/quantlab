#!/usr/bin/env python3
"""
Entry point for the backtest web UI.

Usage:
    PYTHONPATH=. .venv/bin/python quantlab/webui/backtest_ui.py --port 8080

Then open http://localhost:8080
"""

from quantlab.webui.server import main

if __name__ == "__main__":
    main()

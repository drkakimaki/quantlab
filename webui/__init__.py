"""Quantlab Web UI - Browser-based backtest runner."""

from .config import get_strategies, WORKSPACE, QUANTLAB
from .runner import run_backtest, report_exists, get_report_path
from .server import main, BacktestHandler

__all__ = [
    "get_strategies",
    "run_backtest",
    "report_exists",
    "get_report_path",
    "main",
    "BacktestHandler",
    "WORKSPACE",
    "QUANTLAB",
]

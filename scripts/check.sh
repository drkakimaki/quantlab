#!/usr/bin/env bash
set -euo pipefail

# Quantlab safety-check runner (agent-friendly)
# - runs fast unit tests
# - runs the golden series regression test
#
# Usage:
#   ./scripts/check.sh
#   ./scripts/check.sh --unit-only
#   ./scripts/check.sh --regression-only

MODE="all"
if [[ "${1-}" == "--unit-only" ]]; then
  MODE="unit"
elif [[ "${1-}" == "--regression-only" ]]; then
  MODE="regression"
elif [[ -n "${1-}" ]]; then
  echo "Unknown arg: ${1}" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Prefer the workspace venv (used by docs/TOOLS.md), then repo-local venv.
PY="$ROOT/../.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$ROOT/.venv/bin/python"
fi
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

banner() {
  echo ""
  echo "================================================================================"
  echo "$1"
  echo "================================================================================"
}

banner "Python"
"$PY" -V

banner "Import smoke test"
# Repo is used as a source checkout (not necessarily installed as a package).
# Keep this consistent with README (PYTHONPATH=.)
PYTHONPATH="${ROOT}/.." "$PY" -c "import quantlab; import quantlab.engine.backtest; import quantlab.strategies.trend_following; print('OK')"

if [[ "$MODE" == "all" || "$MODE" == "unit" ]]; then
  banner "Unit tests (fast)"
  # Skip regression tests here; they are run explicitly below.
  "$PY" -m pytest -q tests --ignore=tests/regression
fi

if [[ "$MODE" == "all" || "$MODE" == "regression" ]]; then
  banner "Regression: golden series (best_trend 2024–2025)"
  "$PY" -m pytest -q tests/regression/test_best_trend_2024_2025_series.py
fi

banner "DONE"

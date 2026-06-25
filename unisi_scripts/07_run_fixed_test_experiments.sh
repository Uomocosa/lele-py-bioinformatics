#!/usr/bin/env bash
# Run the fixed-test-set augmentation study (Phase 1: baseline vs each LLM source).
# Output goes to RESULTS/fixed_test_experiments/.
# Pass-through args, e.g.:
#   ./07_run_fixed_test_experiments.sh --seeds 42 123 7
#   ./07_run_fixed_test_experiments.sh --phase 2
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_DIR"
export PATH="$HOME/.pixi/bin:$PATH"

echo "[fixed-test] 1/2 smoke test (single arch, max-size 30)..."
pixi run -e cuda pytest -rFP -q -s bio/run_fixed_test_experiments.py::test_smoke -o "addopts="

echo "[fixed-test] 2/2 running full experiment sweep..."
pixi run -e cuda run_fixed_test_experiments "$@"

echo "[fixed-test] done — leaderboard at RESULTS/fixed_test_experiments/leaderboard.{csv,md}"

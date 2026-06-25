#!/usr/bin/env bash
# Run the fixed-test-set augmentation study across all phases.
# Results accumulate in RESULTS/fixed_test_experiments/all_runs.csv — run phases in order.
#
# Usage:
#   ./07_run_fixed_test_experiments.sh           # runs all 4 phases in sequence
#   ./07_run_fixed_test_experiments.sh --phase 2  # run a single phase
#   ./07_run_fixed_test_experiments.sh --seeds 42 123 7 999
#
# Phases:
#   1 — baseline + each single LLM         (5 experiments)
#   2 — all pairs of LLMs                  (6 experiments)
#   3 — all triples of LLMs                (4 experiments)
#   4 — all four LLMs combined             (1 experiment)
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_DIR"
export PATH="$HOME/.pixi/bin:$PATH"

# If --phase N is passed, run only that phase; otherwise run all 4.
if [[ "$*" == *"--phase"* ]]; then
  echo "[fixed-test] running single phase: $*"
  pixi run -e cuda run_fixed_test_experiments "$@"
else
  echo "[fixed-test] 1/4 smoke test (single arch, max-size 30)..."
  pixi run -e cuda pytest -rFP -q -s bio/run_fixed_test_experiments.py::test_smoke -o "addopts="

  for phase in 1 2 3 4; do
    echo "[fixed-test] phase ${phase}/4..."
    pixi run -e cuda run_fixed_test_experiments "$@" --phase "${phase}"
  done
fi

echo "[fixed-test] done — leaderboard at RESULTS/fixed_test_experiments/leaderboard.{csv,md}"

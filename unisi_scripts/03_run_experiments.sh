#!/usr/bin/env bash
# Run the thesis experiment sweep (datasets x architectures, grouped CV) on the GPU.
# Output goes to RESULTS/thesis_experiments/ (the old RESULTS/mlp_experiments/ is untouched).
# Pass-through args, e.g.:
#   ./03_run_experiments.sh --datasets original opus deepseek
#   ./03_run_experiments.sh --architectures hd_16_8_4_4_4 hd_64_32_16_8_4
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_DIR"
export PATH="$HOME/.pixi/bin:$PATH"

pixi run -e cuda run_integrated_paper_scraper_experiments "$@"

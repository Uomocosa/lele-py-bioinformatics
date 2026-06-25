#!/usr/bin/env bash
# Curated grid search for a good model config under grouped CV, on one dataset.
# Output -> RESULTS/config_search/<dataset>/ with its own leaderboard.
# NOTE: the top Q2 is optimistic (best-of-many); confirm the winner before quoting it.
# Pass-through args, e.g.:  ./06_search_config.sh --dataset old_plus_new
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_DIR"
export PATH="$HOME/.pixi/bin:$PATH"

pixi run -e cuda search_for_best_config "$@"

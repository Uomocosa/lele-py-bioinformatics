#!/usr/bin/env bash
# Rebuild the Q2/MAE/RMSE leaderboard from whatever experiments have run so far.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_DIR"
export PATH="$HOME/.pixi/bin:$PATH"

pixi run -e cuda python -c "from bio.run_all_integrated_paper_scraper_experiments import rank; rank()"
echo "[rank] leaderboard at RESULTS/thesis_experiments/q2_leaderboard.{csv,md}"

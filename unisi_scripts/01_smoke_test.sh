#!/usr/bin/env bash
# Fast end-to-end sanity check that every part of the pipeline works before
# committing GPU time to the full sweep. Run after 00_bootstrap.sh.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_DIR"
export PATH="$HOME/.pixi/bin:$PATH"

echo "[smoke] 1/3 fast unit tests (pytest)..."
pixi run -e cuda pytest -q

echo "[smoke] 2/3 integrate CLI on the original PDCC (max-size 8)..."
pixi run -e cuda integrate_paper_scraper \
  --max-size 8 \
  --output-csv RESULTS/thesis_experiments/_smoke/featurized_original.csv

echo "[smoke] 3/3 grouped-CV mini experiment (original+opus+deepseek, k=3, max-size 30)..."
pixi run -e cuda python -c "from bio.run_all_integrated_paper_scraper_experiments import test_smoke; test_smoke()"

echo "[smoke] OK — pipeline works."

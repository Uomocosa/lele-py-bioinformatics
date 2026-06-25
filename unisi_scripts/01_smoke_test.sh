#!/usr/bin/env bash
# Fast end-to-end sanity check that every part of the pipeline works before
# committing GPU time to the full sweep. Run after 00_bootstrap.sh.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_DIR"
export PATH="$HOME/.pixi/bin:$PATH"

echo "[smoke] 1/4 every module imports (pytest collection)..."
pixi run -e cuda pytest --co -q >/dev/null

echo "[smoke] 2/4 metadata-alignment unit test..."
pixi run -e cuda pytest -q -o "addopts=" \
  "bio/Dataset/TorchDataset/PDCCtorch.py::test_metadata_aligns_after_featurize_drops_rows"

echo "[smoke] 3/4 integrate CLI on the original PDCC (max-size 8)..."
pixi run -e cuda integrate_paper_scraper \
  --max-size 8 \
  --output-csv RESULTS/thesis_experiments/_smoke/featurized_original.csv

echo "[smoke] 4/4 grouped-CV mini experiment (original+opus+deepseek, k=3, max-size 30)..."
pixi run -e cuda python -c "from bio.run_all_integrated_paper_scraper_experiments import test_smoke; test_smoke()"

echo "[smoke] OK — pipeline works."

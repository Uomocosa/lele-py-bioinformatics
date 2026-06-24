#!/usr/bin/env bash
# Optional: pre-build / inspect the combined old+new featurized dataset.
# The experiment runner (03) already materializes combined datasets internally, so
# this is mainly a sanity check that all CSVs + dicts resolve and featurize cleanly.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_DIR"
export PATH="$HOME/.pixi/bin:$PATH"
PS=DATASETS/PDCC/paper_scraper

echo "[build] old + new combined -> $PS/featurized_old_plus_new.csv"
pixi run -e cuda integrate_paper_scraper \
  --pdcc-datasets DATASETS/PDCC/polymer_drug_concentration_capacity.csv \
                  "$PS/pdcc_opus_without_conflicts.csv" \
                  "$PS/pdcc_deepseek_without_conflicts.csv" \
                  "$PS/pdcc_kimi_without_conflicts.csv" \
                  "$PS/pdcc_gemma4_image_without_conflicts.csv" \
  --psmiles-dicts builtin "$PS/paper_scraper_complete_psmiles.json" \
  --smiles-dicts  builtin "$PS/paper_scraper_complete_smiles.json" \
  --output-csv "$PS/featurized_old_plus_new.csv"
echo "[build] done."

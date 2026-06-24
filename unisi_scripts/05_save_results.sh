#!/usr/bin/env bash
# Archive the thesis results so you can scp them off before the server is reset.
# (Results are NOT auto-pushed to git; copy the archive yourself.)
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_DIR"

if [ ! -d RESULTS/thesis_experiments ]; then
  echo "[save] nothing to archive: RESULTS/thesis_experiments does not exist yet." >&2
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT="thesis_results_${TS}.tar.gz"
tar czf "$OUT" RESULTS/thesis_experiments
echo "[save] wrote $REPO_DIR/$OUT"
echo "[save] copy it off the server, e.g.:"
echo "       scp <user>@<unisi-host>:$REPO_DIR/$OUT ."

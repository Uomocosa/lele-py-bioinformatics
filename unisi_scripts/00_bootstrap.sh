#!/usr/bin/env bash
# Bootstrap the lele-py-bioinformatics CUDA environment on a fresh UniSI box.
# The server is periodically reset, so this starts from installing pixi and is
# safe to re-run. Run it from anywhere inside the cloned repo.
set -euo pipefail

REPO_URL="https://github.com/Uomocosa/lele-py-bioinformatics"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
echo "[bootstrap] repo: $REPO_DIR  (clone from $REPO_URL if you don't have it)"

# 1. Install pixi if missing.
if ! command -v pixi >/dev/null 2>&1 && [ ! -x "$HOME/.pixi/bin/pixi" ]; then
  echo "[bootstrap] installing pixi..."
  curl -fsSL https://pixi.sh/install.sh | bash
fi
export PATH="$HOME/.pixi/bin:$PATH"
pixi --version

# 2. Materialize the CUDA environment (brings in torch+cuda, rdkit, polymetrix, ...).
echo "[bootstrap] pixi install -e cuda (can take several minutes)..."
pixi install -e cuda

# 3. Verify bio + torch + CUDA are usable.
pixi run -e cuda python -c "import bio, torch; print('bio ok | torch', torch.__version__, '| cuda', torch.cuda.is_available())"
echo "[bootstrap] done."

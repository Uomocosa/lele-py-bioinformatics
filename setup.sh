#!/bin/bash
# This script runs the full update and execution task

# Exit immediately if any command fails
set -e

# --- Setup Logging ---
LOG_DIR="$HOME/cli_logs"
echo "$LOG_DIR"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/session-$(date +%Y_%m_%d-%H_%M_%S).log"

# This magic line redirects all future stdout and stderr to 'tee'
# 'tee' will write output to *both* the log file and the VSCode terminal
exec > >(tee -a "$LOG_FILE") 2>&1

echo "--- Starinting task: INSTALL UV ---"
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing now..."
    # Download and install uv to the local user directory
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Temporarily add uv to the PATH for this running script
    # (The installer usually puts it in ~/.cargo/bin or ~/.local/bin)
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH" 
else
    echo "uv is already installed."
fi

echo "--- Starinting task: UV SYNC ---"
uv sync
uv pip install --upgrade torch --torch-backend=auto

echo "--- Starinting task: INSTALL bio scripts ---"
PYTHON_VERSION="3.11"
uv tool install --editable . --python "$PYTHON_VERSION"
echo "--- Task Finished Successfully ---"

if [[ -z "$GH_LELE_TOKEN" ]]; then
    echo "ERROR: GH_LELE_TOKEN is not set. You will not be able to push to github."
    exit 1
fi

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
# exec > >(tee -a "$LOG_FILE") 2>&1

echo "--- Starinting task: INSTALL PIXI ---"
if ! command -v pixi &> /dev/null; then
    echo "pixi not found. Installing now..."
    # Download and install pixi to the local user directory
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="/home/maggiori/.pixi/bin:$PATH" 
else
    echo "pixi is already installed."
fi

echo "--- Starinting task: PIXI INSTALL ---"
pixi install -e cuda
# uv pip install --upgrade torch --torch-backend=auto

# echo "--- Starinting task: INSTALL bio scripts ---"
# PYTHON_VERSION="3.11"
# uv tool install --editable . --python "$PYTHON_VERSION"
# echo "--- Task Finished Successfully ---"

if [ -f $HOME/.env ]; then
    export $(cat $HOME/.env | xargs)
fi

if [[ -z "$GH_LELE_TOKEN" ]]; then
    echo "ERROR: GH_LELE_TOKEN is not set. You will not be able to push to github."
    exit 1
fi

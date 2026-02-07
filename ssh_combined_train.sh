#!/bin/bash
# This script runs the full update and execution task

# Exit immediately if any command fails
set -e

if [ -f $HOME/.env ]; then
    export $(cat $HOME/.env | xargs)
fi

# --- Setup Logging ---
LOG_DIR="$HOME/cli_logs"
echo "$LOG_DIR"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/session-$(date +%Y_%m_%d-%H_%M_%S).log"

# This magic line redirects all future stdout and stderr to 'tee'
# 'tee' will write output to *both* the log file and the VSCode terminal
exec > >(tee -a "$LOG_FILE") 2>&1

echo "--- Starinting task ---"
echo "--- RUNNING: 'combined_train' ---"
CHECKPOINTS_DIR="COMBINED_checkpoints"
# TO TEST USE THIS: 
uv run --no-sync combined_train --checkpoint_dir="$CHECKPOINTS_DIR"_test --early-stop-patience=1 --epochs=2 --dataset.max-size=1000
# OTHERWISE USE THIS:
# uv run --no-sync combined_train --checkpoint_dir="$CHECKPOINTS_DIR"

echo "--- RUNNING: 'git push' ---"
if [[ -n $(git status --porcelain "$CHECKPOINTS_DIR") ]]; then
    if [[ -z "$GH_LELE_TOKEN" ]]; then
        echo "ERROR: GH_LELE_TOKEN is not set. Cannot push."
        exit 1
    fi
    echo "--- Changes detected in $CHECKPOINTS_DIR. Staging, committing, and pushing. ---"
    git config user.email "maggiori.samuele@gmail.com"
    git config user.name "lele-mecai"
    git add "$CHECKPOINTS_DIR"
    COMMIT_MSG="Automated: Update output_models on $(date +'%Y-%m-%d %H:%M:%S')"
    git commit -m "$COMMIT_MSG"
    REMOTE_RAW=$(git remote get-url origin | sed 's|.*://||')
    AUTH_USER="Uomocosa" 
    git push "https://${AUTH_USER}:${GH_LELE_TOKEN}@${REMOTE_RAW}" main
    # git push "https://lele-mecai:${GH_LELE_TOKEN}@${REMOTE_RAW}" main
else
    echo "--- No changes detected in $CHECKPOINTS_DIR. Skipping commit. ---"
fi

echo "--- Task Finished Successfully ---"

#!/bin/bash
# This script uploads the RESULTS dir to GitHub

# Exit immediately if any command fails
set -e

# 1. Go back to the root of your git repository
cd ~/lele-py-bioinformatics

# 2. Load your environment variables (so GH_LELE_TOKEN is available)
if [ -f $HOME/.env ]; then export $(cat $HOME/.env | xargs); fi

# 3. Set your git config (just like in your old script)
git config user.email "maggiori.samuele@gmail.com"
git config user.name "lele-mecai"

# 4. Stage ONLY the newly generated model folder
git add RESULTS

# 5. Commit the changes
git commit -m "Automated: Update output_models on $(date +'%Y-%m-%d %H:%M:%S')"

# 6. Push using your token to bypass the password prompt
REMOTE_RAW=$(git remote get-url origin | sed 's|.*://||')
AUTH_USER="Uomocosa" 
git push "https://${AUTH_USER}:${GH_LELE_TOKEN}@${REMOTE_RAW}" main

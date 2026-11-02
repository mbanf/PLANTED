#!/usr/bin/env bash
# Launch ConditionEXP training on Jarvislabs A100.
# Uploads only scripts (not data), runs on GPU, auto-pauses when done.
#
# Usage: bash scripts/launch_remote.sh
# Requires: jl CLI, HF_TOKEN in .env

set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ"

# Load HF token
if [ -f .env ]; then
    source .env
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set. Add it to .env or export it."
    exit 1
fi

# Create a temporary staging directory with only what we need
STAGING=$(mktemp -d)
trap "rm -rf $STAGING" EXIT

mkdir -p "$STAGING/scripts"
cp requirements.txt "$STAGING/"
cp scripts/00_download_data.sh "$STAGING/scripts/"
cp scripts/01_precompute_embeddings.py "$STAGING/scripts/"
cp scripts/01b_finetune_lora.py "$STAGING/scripts/"
cp scripts/02_train_conditionexp.py "$STAGING/scripts/"
cp scripts/run_jarvislabs.sh "$STAGING/scripts/"

echo "Staging directory: $STAGING ($(du -sh "$STAGING" | cut -f1))"
echo ""
echo "Launching on A100-80GB spot ($0.89/hr) with --pause..."
echo "The instance will auto-pause when training completes."
echo ""

# Launch: upload staging dir, run the pipeline, auto-pause on completion
/opt/homebrew/bin/jl run start \
    --gpu A100-80GB \
    --spot \
    --name conditionexp \
    --storage 100 \
    --pause \
    --setup "export HF_TOKEN=$HF_TOKEN" \
    --script scripts/run_jarvislabs.sh \
    --yes \
    "$STAGING"

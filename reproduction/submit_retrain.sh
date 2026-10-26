#!/usr/bin/env bash
# Submit the reproduction: prep (CPU), then the 15-model GPU array once prep succeeds.
#
#   RELEASE=/path/to/Dietnet_training DIETNET_VENV=/path/.venv EXP_DIR=/path/to/run_dir \
#   ACCOUNT=my-account bash reproduction/submit_retrain.sh
set -euo pipefail
: "${RELEASE:?}" "${DIETNET_VENV:?}" "${EXP_DIR:?}" "${ACCOUNT:?}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RELEASE DIETNET_VENV EXP_DIR REPO
mkdir -p "$EXP_DIR/logs"

prep=$(sbatch --parsable --account="$ACCOUNT" --output="$EXP_DIR/logs/%x.%j.out" \
    "$REPO/reproduction/prep.sbatch")
train=$(sbatch --parsable --account="$ACCOUNT" --dependency=afterok:"$prep" \
    --output="$EXP_DIR/logs/%x.%A_%a.out" "$REPO/reproduction/retrain.sbatch")
echo "prep $prep, train array $train (starts after prep succeeds)"

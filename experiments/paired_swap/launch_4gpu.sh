#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SCOPE=${1:-full}
CLUSTER_ROOT=${2:-$SCRIPT_DIR/results/${SCOPE}_4gpu}
GPU_IDS=${GPU_IDS:-0,1,2,3}
MODELS_FILE=${3:-${MODELS_FILE:-}}
[[ -n "$MODELS_FILE" ]] || {
  printf 'Usage: %s [scope] [result-root] models.local.json\n' "$0" >&2
  exit 1
}

python3 "$SCRIPT_DIR/prepare_4gpu.py" --scope "$SCOPE" --root "$CLUSTER_ROOT" \
  --gpus "$GPU_IDS" --models "$MODELS_FILE"
"$SCRIPT_DIR/preflight_4gpu.sh" "$SCOPE" "$CLUSTER_ROOT"
"$SCRIPT_DIR/start_workers_4gpu.sh" "$SCOPE" "$CLUSTER_ROOT"
"$SCRIPT_DIR/status_4gpu.sh" "$SCOPE" "$CLUSTER_ROOT"

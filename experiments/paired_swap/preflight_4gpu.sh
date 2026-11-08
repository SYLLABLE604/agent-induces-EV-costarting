#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SCOPE=${1:-full}
CLUSTER_ROOT=${2:-$SCRIPT_DIR/results/${SCOPE}_4gpu}
CLUSTER_ROOT=$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$CLUSTER_ROOT")
TABLE=$CLUSTER_ROOT/launch.tsv
[[ -f "$TABLE" ]] || { printf 'Missing %s. Run prepare_4gpu.py first.\n' "$TABLE" >&2; exit 1; }
mkdir -p "$CLUSTER_ROOT/logs"

tail -n +2 "$TABLE" | while IFS=$'\t' read -r worker gpu model model_path model_identity manifest output expected; do
  printf 'Preflight worker %s on GPU %s (%s)\n' "$worker" "$gpu" "$model"
  CUDA_VISIBLE_DEVICES="$gpu" python3 -u "$SCRIPT_DIR/run.py" preflight --scope "$SCOPE" \
    --worker-manifest "$manifest" --output "$output" \
    > "$CLUSTER_ROOT/logs/worker_${worker}_preflight.log" 2>&1
done

printf 'All four worker preflights completed; their outcomes will be reused.\n'

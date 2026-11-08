#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SCOPE=${1:-full}
CLUSTER_ROOT=${2:-$SCRIPT_DIR/results/${SCOPE}_4gpu}
CLUSTER_ROOT=$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$CLUSTER_ROOT")
TABLE=$CLUSTER_ROOT/launch.tsv
[[ -f "$TABLE" ]] || { printf 'Missing %s. Run prepare_4gpu.py first.\n' "$TABLE" >&2; exit 1; }
mkdir -p "$CLUSTER_ROOT/logs" "$CLUSTER_ROOT/pids"
MARKER=$CLUSTER_ROOT/keep_workers_running
touch "$MARKER"

pid_is_ours() {
  local pid=$1 token=$2
  [[ -r "/proc/$pid/cmdline" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq -- "$token"
}

tail -n +2 "$TABLE" | while IFS=$'\t' read -r worker gpu model model_path model_identity manifest output expected; do
  CUDA_VISIBLE_DEVICES="$gpu" python3 "$SCRIPT_DIR/run.py" check --scope "$SCOPE" \
    --worker-manifest "$manifest" --output "$output" \
    > "$CLUSTER_ROOT/logs/worker_${worker}_check.json"
done

tail -n +2 "$TABLE" | while IFS=$'\t' read -r worker gpu model model_path model_identity manifest output expected; do
  supervisor_pid_file=$CLUSTER_ROOT/pids/worker_supervisor_${worker}.pid
  child_pid_file=$CLUSTER_ROOT/pids/worker_child_${worker}.pid
  failed_file=$CLUSTER_ROOT/worker_${worker}.failed
  if python3 - "$output" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1]) / "status.json"
raise SystemExit(0 if path.exists() and json.loads(path.read_text()).get("status") == "complete" else 1)
PY
  then
    printf 'Worker %s is already complete.\n' "$worker"
    continue
  fi
  if [[ -f "$supervisor_pid_file" ]]; then
    pid=$(<"$supervisor_pid_file")
    if kill -0 "$pid" 2>/dev/null && pid_is_ours "$pid" "$SCRIPT_DIR/supervise_worker.sh"; then
      printf 'Worker supervisor %s already running.\n' "$worker"
      continue
    fi
    rm -f "$supervisor_pid_file"
  fi
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$SCRIPT_DIR/supervise_worker.sh" "$SCRIPT_DIR" \
    "$SCOPE" "$manifest" "$output" "$MARKER" "$child_pid_file" \
    "$CLUSTER_ROOT/logs/worker_${worker}.log" "$failed_file" \
    > "$CLUSTER_ROOT/logs/worker_supervisor_${worker}.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$supervisor_pid_file"
  printf 'Started worker %s: GPU %s, %s, expected outcomes %s\n' "$worker" "$gpu" "$model" "$expected"
done

printf 'All incomplete workers have been started in the background.\n'

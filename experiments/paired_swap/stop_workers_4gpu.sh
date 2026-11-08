#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SCOPE=${1:-full}
CLUSTER_ROOT=${2:-$SCRIPT_DIR/results/${SCOPE}_4gpu}
CLUSTER_ROOT=$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$CLUSTER_ROOT")
rm -f "$CLUSTER_ROOT/keep_workers_running"

for worker in 0 1 2 3; do
  pid_file=$CLUSTER_ROOT/pids/worker_supervisor_${worker}.pid
  if [[ ! -f "$pid_file" ]]; then continue; fi
  pid=$(<"$pid_file")
  if [[ -r "/proc/$pid/cmdline" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq -- "$SCRIPT_DIR/supervise_worker.sh"; then
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
  else
    printf 'Worker supervisor PID %s is not owned by this experiment; not stopping it.\n' "$pid" >&2
  fi
  rm -f "$pid_file"
done
printf 'Experiment worker supervisors stopped. Completed outcomes remain resumable.\n'

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SCOPE=${1:-full}
CLUSTER_ROOT=${2:-$SCRIPT_DIR/results/${SCOPE}_4gpu}
CLUSTER_ROOT=$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$CLUSTER_ROOT")
TABLE=$CLUSTER_ROOT/launch.tsv
[[ -f "$TABLE" ]] || { printf 'Missing %s.\n' "$TABLE" >&2; exit 1; }

printf 'worker gpu model             process completed/expected status\n'
printf '%s\n' '----------------------------------------------------------------'
total=0
complete=0
tail -n +2 "$TABLE" | while IFS=$'\t' read -r worker gpu model model_path model_identity manifest output expected; do
  child_pid_file=$CLUSTER_ROOT/pids/worker_child_${worker}.pid
  worker_state=down
  if [[ -f "$child_pid_file" ]] && kill -0 "$(<"$child_pid_file")" 2>/dev/null; then worker_state=up; fi
  read -r done state < <(python3 - "$output" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
status = root / "status.json"
if not status.exists():
    print(0, "prepared")
else:
    value = json.loads(status.read_text())
    print(value.get("completed", 0), value.get("status", "unknown"))
PY
  )
  if [[ -f "$CLUSTER_ROOT/worker_${worker}.failed" ]]; then state=failed; fi
  printf '%-6s %-3s %-17s %-7s %7s/%-7s %s\n' \
    "$worker" "$gpu" "$model" "$worker_state" "$done" "$expected" "$state"
done

python3 - "$TABLE" <<'PY'
import csv, json, pathlib, sys
rows = list(csv.DictReader(pathlib.Path(sys.argv[1]).open(), delimiter="\t"))
done = expected = 0
for row in rows:
    path = pathlib.Path(row["output"]) / "status.json"
    expected += int(row["expected"])
    if path.exists():
        done += int(json.loads(path.read_text()).get("completed", 0))
print(f"Total progress: {done}/{expected} ({done/expected:.2%})")
PY

if command -v nvidia-smi >/dev/null; then
  printf '\nGPU processes and memory:\n'
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
fi

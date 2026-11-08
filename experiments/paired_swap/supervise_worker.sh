#!/usr/bin/env bash
set -u

SCRIPT_DIR=$1
SCOPE=$2
MANIFEST=$3
OUTPUT=$4
MARKER=$5
CHILD_PID_FILE=$6
WORKER_LOG=$7
FAILED_FILE=$8

child_pid=""
shutdown() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  rm -f "$CHILD_PID_FILE"
  exit 0
}
trap shutdown TERM INT

failures=0
rm -f "$FAILED_FILE"
while [[ -f "$MARKER" ]]; do
  started=$(date +%s)
  python3 -u "$SCRIPT_DIR/run.py" run --scope "$SCOPE" \
    --worker-manifest "$MANIFEST" --output "$OUTPUT" \
    >> "$WORKER_LOG" 2>&1 &
  child_pid=$!
  printf '%s\n' "$child_pid" > "$CHILD_PID_FILE"
  wait "$child_pid"
  exit_code=$?
  elapsed=$(( $(date +%s) - started ))
  rm -f "$CHILD_PID_FILE"
  child_pid=""
  if python3 - "$OUTPUT" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1]) / "status.json"
raise SystemExit(0 if path.exists() and json.loads(path.read_text()).get("status") == "complete" else 1)
PY
  then
    exit 0
  fi
  if [[ ! -f "$MARKER" ]]; then
    exit 0
  fi
  if (( elapsed < 30 )); then
    failures=$((failures + 1))
  else
    failures=1
  fi
  if (( failures >= 20 )); then
    printf '%s worker stopped after %s rapid failures; inspect %s\n' \
      "$(date -u +%FT%TZ)" "$failures" "$WORKER_LOG" > "$FAILED_FILE"
    exit "$exit_code"
  fi
  delay=$((failures * 5))
  if (( delay > 60 )); then delay=60; fi
  printf '%s worker exited code=%s; restarting in %s s\n' \
    "$(date -u +%FT%TZ)" "$exit_code" "$delay" >> "$WORKER_LOG"
  sleep "$delay"
done

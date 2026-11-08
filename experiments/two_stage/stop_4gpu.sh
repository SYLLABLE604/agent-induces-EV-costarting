#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
OUT="${OUT:-$PWD/results}"
for f in "$OUT"/pids/*.pid; do
  [[ -f "$f" ]] || continue
  pid="$(cat "$f")"
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  if [[ -r "/proc/$pid/cmdline" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q 'run.py run'; then
    kill -TERM "$pid"
    echo "Requested graceful stop of $pid"
  fi
done

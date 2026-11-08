#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
"${PYTHON_BIN:-python3}" run.py status --out "${OUT:-$PWD/results}"
if command -v nvidia-smi >/dev/null; then nvidia-smi; fi

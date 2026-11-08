#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODELS="${MODELS:-$PWD/models.local.json}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS:-0,1,2,3}"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON_BIN" run.py preflight --model qwen --models "$MODELS"
CUDA_VISIBLE_DEVICES="${GPUS[2]}" "$PYTHON_BIN" run.py preflight --model gemma --models "$MODELS"

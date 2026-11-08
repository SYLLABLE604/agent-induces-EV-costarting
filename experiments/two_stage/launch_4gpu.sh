#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT="${OUT:-$PWD/results}"
MODELS="${MODELS:-$PWD/models.local.json}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS:-0,1,2,3}"
if [[ ${#GPUS[@]} -ne 4 ]]; then echo 'GPU_IDS must contain four GPU IDs' >&2; exit 1; fi
"$PYTHON_BIN" run.py check
mkdir -p "$OUT/logs" "$OUT/pids"
for slot in 0 1 2 3; do
  model=qwen; shard=$slot
  if (( slot >= 2 )); then model=gemma; shard=$((slot - 2)); fi
  pidfile="$OUT/pids/${model}_${shard}.pid"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "Worker already running: $pidfile" >&2; exit 1
  fi
done
for slot in 0 1 2 3; do
  model=qwen; shard=$slot
  if (( slot >= 2 )); then model=gemma; shard=$((slot - 2)); fi
  env CUDA_VISIBLE_DEVICES="${GPUS[$slot]}" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      nohup "$PYTHON_BIN" -u run.py run --model "$model" --shard "$shard" --shards 2 \
      --models "$MODELS" --out "$OUT" \
      >> "$OUT/logs/${model}_${shard}.log" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$OUT/pids/${model}_${shard}.pid"
  echo "GPU ${GPUS[$slot]}: $model shard $shard/2 pid=$pid"
done
echo "Progress: $PYTHON_BIN run.py status --out '$OUT'"
echo "GPU usage: watch -n 2 nvidia-smi"
echo "Resume: rerun this script after all previous workers have stopped."

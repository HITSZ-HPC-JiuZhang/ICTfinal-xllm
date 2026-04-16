#!/bin/bash
set -o pipefail

rm -rf core.*

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
else
  echo "[WARN] ATB env script not found: /usr/local/Ascend/nnal/atb/set_env.sh"
fi

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-43432}"

MODEL_PATH="${MODEL_PATH:-/mnt/nvme0/models/Qwen3.5/Qwen3.5-9B}"
MASTER_NODE_ADDR="${MASTER_NODE_ADDR:-127.0.0.1:9748}"
START_PORT="${START_PORT:-18000}"
START_DEVICE="${START_DEVICE:-0}"
LOG_DIR="${LOG_DIR:-log}"
NNODES="${NNODES:-4}"
TAIL_LINES="${TAIL_LINES:-40}"

# Docker default /dev/shm may be too small for shm transport.
ENABLE_SHM="${ENABLE_SHM:-false}"
PREWARM_NPU="${PREWARM_NPU:-false}"

mkdir -p "$LOG_DIR"
declare -a PIDS

if [[ -f "$MODEL_PATH/model.safetensors.index.json" ]]; then
  echo "[INFO] Checking model shard completeness via model.safetensors.index.json"
  export XLLM_MODEL_PATH="$MODEL_PATH"
  missing_files="$(python3 - <<'PY'
import json
import os

model_path = os.environ["XLLM_MODEL_PATH"]
index_path = os.path.join(model_path, "model.safetensors.index.json")
with open(index_path, "r", encoding="utf-8") as f:
    data = json.load(f)

weight_map = data.get("weight_map", {})
required = sorted(set(weight_map.values()))
missing = [name for name in required if not os.path.isfile(os.path.join(model_path, name))]
for name in missing:
    print(name)
PY
)"

  if [[ -n "$missing_files" ]]; then
    echo "[ERROR] Missing model shard file(s):"
    while IFS= read -r line; do
      [[ -n "$line" ]] && echo "  - $line"
    done <<< "$missing_files"
    echo "[ERROR] Model path is incomplete: $MODEL_PATH"
    exit 1
  fi
fi

if [[ "$PREWARM_NPU" == "true" ]]; then
  echo "[INFO] Prewarming NPU devices before launching xllm..."
  export XLLM_ASCEND_DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
  if ! python3 - <<'PY'
import os
import sys

devices_env = os.environ.get("XLLM_ASCEND_DEVICES", "0")
devices = []
for part in devices_env.split(","):
    part = part.strip()
    if part:
        devices.append(int(part))
if not devices:
    devices = [0]

try:
    import torch_npu
except Exception as e:  # pragma: no cover
    print(f"[ERROR] import torch_npu failed: {e}")
    sys.exit(1)

for dev in devices:
    torch_npu.npu.set_device(dev)
    print(f"[PREWARM-OK] npu:{dev}")
PY
  then
    echo "[ERROR] NPU prewarm failed. You can bypass with PREWARM_NPU=false."
    exit 1
  fi
fi

echo "[INFO] Launching xllm nodes..."
echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] NNODES=$NNODES ENABLE_SHM=$ENABLE_SHM PREWARM_NPU=$PREWARM_NPU ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"

if [[ -f "/.dockerenv" && "$ENABLE_SHM" == "true" ]]; then
  echo "[WARN] Docker detected with ENABLE_SHM=true. Consider --ipc=host or --shm-size=2g."
fi

for ((i = 0; i < NNODES; i++)); do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"

  /home/kuma/project/ict-final/xllm/build/xllm/core/server/xllm \
    --model="$MODEL_PATH" \
    --devices="npu:$DEVICE" \
    --port "$PORT" \
    --master_node_addr="$MASTER_NODE_ADDR" \
    --nnodes="$NNODES" \
    --max_memory_utilization=0.25 \
    --block_size=128 \
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=false \
    --enable_shm="$ENABLE_SHM" \
    --max_tokens_per_batch= \
    --enable_graph=true \
    --node_rank="$i" > "$LOG_FILE" 2>&1 &

  pid=$!
  PIDS[$i]="$pid"
  echo "[LAUNCHED] rank=$i pid=$pid device=npu:$DEVICE port=$PORT log=$LOG_FILE"
done

dead_count=0
for ((i = 0; i < NNODES; i++)); do
  pid="${PIDS[$i]}"
  log_file="$LOG_DIR/node_$i.log"

  if kill -0 "$pid" 2>/dev/null; then
    echo "[ALIVE] rank=$i pid=$pid"
  else
    rc=0
    if wait "$pid"; then
      rc=0
    else
      rc=$?
    fi

    dead_count=$((dead_count + 1))
    echo "[EARLY-EXIT] rank=$i pid=$pid exit_code=$rc log=$log_file"
    if [[ -f "$log_file" ]]; then
      echo "[LOG-TAIL] rank=$i (last ${TAIL_LINES} lines)"
      tail -n "$TAIL_LINES" "$log_file" | sed 's/^/  /'
    fi
  fi
done

if ((dead_count > 0)); then
  echo "[SUMMARY] ${dead_count}/${NNODES} ranks exited early."
  exit 1
fi

echo "[SUMMARY] All ${NNODES} ranks are alive after ${STARTUP_CHECK_SEC}s."
echo "[INFO] Check logs with: tail -f ${LOG_DIR}/node_0.log"
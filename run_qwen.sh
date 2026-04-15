#!/bin/bash
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
XLLM_BIN="${XLLM_BIN:-$SCRIPT_DIR/build/xllm/core/server/xllm}"
MASTER_NODE_ADDR="${MASTER_NODE_ADDR:-127.0.0.1:9748}"
START_PORT="${START_PORT:-18000}"
START_DEVICE="${START_DEVICE:-0}"
LOG_DIR="${LOG_DIR:-log}"
NNODES="${NNODES:-4}"
TAIL_LINES="${TAIL_LINES:-40}"
STARTUP_CHECK_SEC="${STARTUP_CHECK_SEC:-0}"
# Default fits the current 16K input + 1K output single-concurrency PLD run.
MAX_TOKENS_PER_BATCH="${MAX_TOKENS_PER_BATCH:-34000}"
MAX_SEQS_PER_BATCH="${MAX_SEQS_PER_BATCH:-1}"
SPECULATIVE_ALGORITHM="${SPECULATIVE_ALGORITHM:-PLD}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-8}"
SPECULATIVE_SUFFIX_CACHE_MAX_DEPTH="${SPECULATIVE_SUFFIX_CACHE_MAX_DEPTH:-64}"

# Docker default /dev/shm may be too small for shm transport.
ENABLE_SHM="${ENABLE_SHM:-false}"
PREWARM_NPU="${PREWARM_NPU:-false}"

mkdir -p "$LOG_DIR"
declare -a PIDS
declare -a LAUNCH_DEVICES

IFS=',' read -r -a VISIBLE_DEVICE_LIST <<< "$ASCEND_RT_VISIBLE_DEVICES"
VISIBLE_DEVICE_COUNT=0
for raw_device in "${VISIBLE_DEVICE_LIST[@]}"; do
  device_id="${raw_device//[[:space:]]/}"
  if [[ -n "$device_id" ]]; then
    VISIBLE_DEVICE_COUNT=$((VISIBLE_DEVICE_COUNT + 1))
  fi
done

if ((NNODES < 1)); then
  echo "[ERROR] NNODES must be at least 1, got $NNODES"
  exit 1
fi

if ((START_DEVICE < 0)); then
  echo "[ERROR] START_DEVICE must be non-negative, got $START_DEVICE"
  exit 1
fi

if ((STARTUP_CHECK_SEC < 0)); then
  echo "[ERROR] STARTUP_CHECK_SEC must be non-negative, got $STARTUP_CHECK_SEC"
  exit 1
fi

for ((i = 0; i < NNODES; i++)); do
  device_id=$((START_DEVICE + i))
  if ((device_id >= VISIBLE_DEVICE_COUNT)); then
    echo "[ERROR] Requested logical device npu:$device_id exceeds visible device count $VISIBLE_DEVICE_COUNT from ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
    echo "[ERROR] Please align START_DEVICE/NNODES with the visible device range."
    exit 1
  fi
  LAUNCH_DEVICES[$i]="$device_id"
done

if [[ ! -f "$XLLM_BIN" || ! -x "$XLLM_BIN" ]]; then
  echo "[ERROR] xLLM binary is not executable: $XLLM_BIN"
  echo "[ERROR] Set XLLM_BIN to the binary built from this source tree."
  exit 1
fi
XLLM_BIN_RESOLVED="$(realpath "$XLLM_BIN")"

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
  XLLM_ASCEND_DEVICES="$(IFS=,; echo "${LAUNCH_DEVICES[*]}")"
  export XLLM_ASCEND_DEVICES
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
echo "[INFO] XLLM_BIN=$XLLM_BIN"
echo "[INFO] XLLM_BIN_RESOLVED=$XLLM_BIN_RESOLVED"
echo "[INFO] NNODES=$NNODES ENABLE_SHM=$ENABLE_SHM PREWARM_NPU=$PREWARM_NPU ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
echo "[INFO] LAUNCH_DEVICES=$(IFS=,; echo "${LAUNCH_DEVICES[*]}") START_DEVICE=$START_DEVICE"
echo "[INFO] MAX_TOKENS_PER_BATCH=$MAX_TOKENS_PER_BATCH MAX_SEQS_PER_BATCH=$MAX_SEQS_PER_BATCH"
echo "[INFO] SPECULATIVE_ALGORITHM=$SPECULATIVE_ALGORITHM NUM_SPECULATIVE_TOKENS=$NUM_SPECULATIVE_TOKENS SPECULATIVE_SUFFIX_CACHE_MAX_DEPTH=$SPECULATIVE_SUFFIX_CACHE_MAX_DEPTH"

if [[ -f "/.dockerenv" && "$ENABLE_SHM" == "true" ]]; then
  echo "[WARN] Docker detected with ENABLE_SHM=true. Consider --ipc=host or --shm-size=2g."
fi

for ((i = 0; i < NNODES; i++)); do
  PORT=$((START_PORT + i))
  DEVICE="${LAUNCH_DEVICES[$i]}"
  LOG_FILE="$LOG_DIR/node_$i.log"

  "$XLLM_BIN" \
    --model="$MODEL_PATH" \
    --devices="npu:$DEVICE" \
    --port "$PORT" \
    --master_node_addr="$MASTER_NODE_ADDR" \
    --nnodes="$NNODES" \
    --max_memory_utilization=0.88 \
    --block_size=128 \
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=false \
    --enable_shm="$ENABLE_SHM" \
    --max_tokens_per_batch="$MAX_TOKENS_PER_BATCH" \
    --max_seqs_per_batch="$MAX_SEQS_PER_BATCH" \
    --enable_graph=false \
    --enable_graph_collective_scope_log=false \
    --speculative_algorithm "$SPECULATIVE_ALGORITHM" \
    --num_speculative_tokens "$NUM_SPECULATIVE_TOKENS" \
    --speculative_suffix_cache_max_depth "$SPECULATIVE_SUFFIX_CACHE_MAX_DEPTH" \
    --node_rank="$i" > "$LOG_FILE" 2>&1 &

  pid=$!
  PIDS[$i]="$pid"
  echo "[LAUNCHED] rank=$i pid=$pid device=npu:$DEVICE port=$PORT log=$LOG_FILE"
done

if ((STARTUP_CHECK_SEC > 0)); then
  sleep "$STARTUP_CHECK_SEC"
fi

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

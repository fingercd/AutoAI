#!/usr/bin/env bash
set -Eeuo pipefail

# One serving recipe for all three local quantized backbones.  GPU assignment is
# explicit at launch time; this script never selects or kills another process.
MODEL_KEY="${1:?usage: serve_model.sh qwen35_9b|qwen35_27b|qwen38_27b}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES after an nvidia-smi check}"
ROOT="${AUTOAI_SHARED_ROOT:-/users/fotile/AutoAI/shared}"
MODEL_ROOT="$ROOT/models/quantized/w8a8-int8"
VLLM_COMMAND="${AUTOAI_VLLM_COMMAND:-vllm}"

case "$MODEL_KEY" in
  qwen35_9b)
    MODEL_PATH="$MODEL_ROOT/qwen3.5-9b"
    PORT=8101
    SERVED_NAME=qwen35_9b
    TENSOR_PARALLEL=1
    ;;
  qwen35_27b)
    MODEL_PATH="$MODEL_ROOT/qwen3.5-27b"
    PORT=8102
    SERVED_NAME=qwen35_27b
    # TP=1 is the default because it fits on one A100; TP=2 is an explicit
    # safe fallback for long-context smoke on two preflight-approved GPUs.
    TENSOR_PARALLEL="${AUTOAI_TENSOR_PARALLEL:-1}"
    ;;
  qwen38_27b)
    MODEL_PATH="$MODEL_ROOT/qwen3.8-27b"
    PORT=8103
    SERVED_NAME=qwen38_27b
    TENSOR_PARALLEL="${AUTOAI_TENSOR_PARALLEL:-1}"
    ;;
  *)
    echo "unknown model key: $MODEL_KEY" >&2
    exit 2
    ;;
esac

test -d "$MODEL_PATH"
exec "$VLLM_COMMAND" serve "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TENSOR_PARALLEL" \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.80 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --language-model-only

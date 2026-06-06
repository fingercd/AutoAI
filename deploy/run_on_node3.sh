#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/AutoAI}"
ENV_NAME="${ENV_NAME:-autoai}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export CUDA_VISIBLE_DEVICES

cd "$APP_DIR"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [ -x "$HOME/miniconda3/bin/conda" ]; then
  CONDA_BASE="$HOME/miniconda3"
elif [ -x "$HOME/anaconda3/bin/conda" ]; then
  CONDA_BASE="$HOME/anaconda3"
else
  echo "conda not found. Please load miniconda/anaconda first." >&2
  exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

mkdir -p "$APP_DIR/storage/logs"
exec python -m uvicorn backend.app.main:app --host "$HOST" --port "$PORT"

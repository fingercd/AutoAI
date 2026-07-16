#!/usr/bin/env bash
# Web-only cluster launcher. This script does not start the training worker; queued Runs
# require a separately managed `python -m backend.app.runs.worker` process.
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/AutoAI}"
ENV_NAME="${ENV_NAME:-autoai}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export CUDA_VISIBLE_DEVICES

# SSH tunnel deployments remain loopback-only and need no shared browser token. Any
# externally reachable bind is always forced into server mode. The token is supplied by
# the process manager/environment and is deliberately never echoed or written to disk.
case "$HOST" in
  127.0.0.1|localhost|::1)
    export AUTOAI_DEPLOYMENT_MODE="${AUTOAI_DEPLOYMENT_MODE:-local}"
    ;;
  *)
    export AUTOAI_DEPLOYMENT_MODE="server"
    ;;
esac

API_TOKEN_LENGTH=0
if [ -n "${AUTOAI_API_TOKEN:-}" ]; then
  API_TOKEN_LENGTH="${#AUTOAI_API_TOKEN}"
fi
if [ "$AUTOAI_DEPLOYMENT_MODE" = "server" ] && [ "$API_TOKEN_LENGTH" -lt 32 ]; then
  echo "server mode requires AUTOAI_API_TOKEN with at least 32 characters" >&2
  exit 1
fi

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
# exec keeps the shell PID equal to uvicorn's PID so persistent/SGE process managers can
# stop and observe the real Web process without an intermediate shell.
exec python -m uvicorn backend.app.main:app --host "$HOST" --port "$PORT"

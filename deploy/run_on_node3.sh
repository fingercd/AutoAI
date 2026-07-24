#!/usr/bin/env bash
# Web-only cluster launcher. This script does not start the training worker; queued Runs
# require a separately managed `python -m backend.app.runs.worker` process.
#
# 【中文说明】集群（node3）上的 Web 服务启动脚本，只启动 FastAPI/uvicorn Web 进程：
#   - 根据绑定地址自动推导部署模式：回环地址 → local（SSH 隧道场景），
#     其它地址 → 强制 server 模式（要求 AUTOAI_API_TOKEN ≥ 32 字符）；
#   - 激活 Conda 环境后用 exec 启动 uvicorn，使 shell 进程与 Web 进程同 PID，
#     方便上层进程管理器（nohup/SGE）直接观察和停止真正的 Web 进程。
# 注意：本脚本不启动训练 worker；queued Run 需要另行托管
# `python -m backend.app.runs.worker` 进程才会被真正执行。
set -euo pipefail

# 以下参数均可通过环境变量覆盖。CUDA_VISIBLE_DEVICES 默认指向 6 号卡并导出给
# 子进程，使进程内触发的任何 GPU 计算（如深度模型可解释性分析）落在指定显卡上。
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

# 与 install_on_node2.sh 同理：集群登录 shell 不一定初始化 Conda，需显式定位根目录。
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
# 中文：exec 用 uvicorn 进程镜像替换当前 shell（PID 不变），上层 nohup/SGE
# 进程管理器因此可直接停止/观察真正的 Web 进程，中间没有 shell 垫层。
exec python -m uvicorn backend.app.main:app --host "$HOST" --port "$PORT"

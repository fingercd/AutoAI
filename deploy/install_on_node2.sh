#!/usr/bin/env bash
# Legacy cluster bootstrap helper. Defaults target the historical AutoAI directory and
# Python 3.11 environment; review APP_DIR/ENV_NAME and prefer server_deploy.md for v2.
#
# 【中文说明】集群节点（node2）上的一次性环境安装脚本（历史遗留 bootstrap）：
#   - 在集群上定位并激活 Conda，创建/复用名为 ENV_NAME 的 Python 3.11 环境；
#   - 安装 backend/requirements.txt 中的全部后端依赖；
#   - 最后用一段内联 Python 做“冒烟导入”，确认 fastapi/uvicorn/torch 等关键包可用。
# 注意：默认参数面向历史 AutoAI 目录；v2 部署请优先参考 deploy/server_deploy.md。
set -euo pipefail
# set -euo pipefail：严格模式——任一命令失败即退出（-e）、引用未定义变量报错（-u）、
# 管道中任一环节失败则整段管道视为失败（-o pipefail），避免半成品安装被当成成功。

# 安装位置与环境名均可通过环境变量覆盖；默认 $HOME/AutoAI 与 autoai。
APP_DIR="${APP_DIR:-$HOME/AutoAI}"
ENV_NAME="${ENV_NAME:-autoai}"

cd "$APP_DIR"

# Cluster login shells do not always initialize Conda, so locate its base explicitly
# before sourcing conda.sh.
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

# 幂等创建：仅当目标环境尚不存在时才创建，重复执行本脚本是安全的。
# awk 取每行第一列（环境名），grep -qx 做整行精确匹配，避免子串误判
#（例如已有 autoai2 时误判 autoai 存在）。
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.11
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip
# 显式指定官方 PyPI 源，避免集群上残留的不可用镜像配置导致依赖安装失败。
python -m pip install -r backend/requirements.txt -i https://pypi.org/simple

# 依赖冒烟检查：逐个 import 关键包，任一缺失都会抛异常并以非零码退出；
# 配合 set -e 让安装问题在部署阶段尽早暴露，而不是拖到服务启动时才报错。
python - <<'PY'
import importlib
mods = ["fastapi", "uvicorn", "pandas", "numpy", "sklearn", "torch", "rampy", "multipart"]
for mod in mods:
    importlib.import_module(mod)
print("SpecAutoAI dependency check OK")
PY

#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/AutoAI}"
ENV_NAME="${ENV_NAME:-autoai}"

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

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.11
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt -i https://pypi.org/simple

python - <<'PY'
import importlib
mods = ["fastapi", "uvicorn", "pandas", "numpy", "sklearn", "torch", "rampy", "multipart"]
for mod in mods:
    importlib.import_module(mod)
print("AutoAI dependency check OK")
PY

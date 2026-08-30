#!/usr/bin/env bash
set -Eeuo pipefail

# Build isolated runtimes on node3.  The script is safe to re-run: an existing
# environment is reused, while every install/freeze/probe is logged separately.
SOURCE_ROOT="${AUTOAI_SOURCE_ROOT:-/users/fotile/AutoAI/current}"
AUTOAI_ENV="${AUTOAI_ENV_PATH:-/users/fotile/AutoAI/envs/autoai-app}"
QWEN_ENV="${QWEN_SERVING_ENV_PATH:-/users/fotile/AutoAI/shared/envs/qwen-serving}"
EVIDENCE_ROOT="${AUTOAI_ENV_EVIDENCE_ROOT:-/users/fotile/AutoAI/shared/work/agent_v1_poc/envs}"
CONDA_BIN="${CONDA_BIN:-/users/fotile/miniconda3/bin/conda}"
CONDA_SH="$(dirname "$CONDA_BIN")/../etc/profile.d/conda.sh"
mkdir -p "$EVIDENCE_ROOT"
source "$CONDA_SH"
NODE3_CONSTRAINTS="${AUTOAI_NODE3_CONSTRAINTS:-$SOURCE_ROOT/deploy/agent_poc/constraints-node3.txt}"
PYTORCH_INDEX_URL="${AUTOAI_PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
PIP_RETRIES="${AUTOAI_PIP_RETRIES:-20}"
PIP_RESUME_RETRIES="${AUTOAI_PIP_RESUME_RETRIES:-20}"
PIP_TIMEOUT="${AUTOAI_PIP_TIMEOUT:-120}"
PIP_NETWORK_ARGS=(--retries "$PIP_RETRIES" --resume-retries "$PIP_RESUME_RETRIES" --timeout "$PIP_TIMEOUT")

on_exit() {
  local rc="$?"
  if [[ "$rc" -ne 0 ]]; then
    printf '{"phase":"failed","exit_code":%s,"updated_at_utc":"%s"}\n' \
      "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$EVIDENCE_ROOT/bootstrap.failed.json"
  fi
  exit "$rc"
}
trap on_exit EXIT

write_status() {
  local name="$1" phase="$2" rc="${3:-0}"
  printf '{"name":"%s","phase":"%s","exit_code":%s,"updated_at_utc":"%s"}\n' \
    "$name" "$phase" "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >"$EVIDENCE_ROOT/$name.status.json"
}

ensure_conda_env() {
  local prefix="$1" name="$2"
  if [[ ! -x "$prefix/bin/python" ]]; then
    write_status "$name" "creating" ""
    "$CONDA_BIN" create -y -p "$prefix" python=3.12 \
      >"$EVIDENCE_ROOT/$name.conda-create.log" 2>&1
  fi
  "$prefix/bin/python" --version | tee "$EVIDENCE_ROOT/$name.python-version.log"
}

ensure_conda_env "$AUTOAI_ENV" autoai-app
write_status autoai-app installing
"$AUTOAI_ENV/bin/python" -m pip install --upgrade pip \
  >"$EVIDENCE_ROOT/autoai-app.install.log" 2>&1
"$AUTOAI_ENV/bin/python" -m pip install \
  -r "$SOURCE_ROOT/backend/requirements.txt" \
  -c "$NODE3_CONSTRAINTS" \
  --prefer-binary \
  >>"$EVIDENCE_ROOT/autoai-app.install.log" 2>&1
"$AUTOAI_ENV/bin/python" -m pip install -r "$SOURCE_ROOT/agent_poc/requirements.txt" \
  >>"$EVIDENCE_ROOT/autoai-app.install.log" 2>&1
"$AUTOAI_ENV/bin/python" -m pip freeze >"$EVIDENCE_ROOT/autoai-app.pip-freeze.txt"
PYTHONPATH="$SOURCE_ROOT" "$AUTOAI_ENV/bin/python" -c \
  "from backend.app.main import app; print(app.title); import agent_poc; print(agent_poc.__version__)" \
  >"$EVIDENCE_ROOT/autoai-app.probe.log" 2>&1
write_status autoai-app ready

ensure_conda_env "$QWEN_ENV" qwen-serving
write_status qwen-serving installing
"$QWEN_ENV/bin/python" -m pip install --upgrade pip \
  >"$EVIDENCE_ROOT/qwen-serving.install.log" 2>&1
"$QWEN_ENV/bin/python" -m pip install \
  --index-url "$PYTORCH_INDEX_URL" \
  "torch==2.6.0" \
  "${PIP_NETWORK_ARGS[@]}" \
  >>"$EVIDENCE_ROOT/qwen-serving.install.log" 2>&1
"$QWEN_ENV/bin/python" -m pip install \
  "vllm>=0.17.0,<0.18" llmcompressor modelscope \
  "${PIP_NETWORK_ARGS[@]}" \
  >>"$EVIDENCE_ROOT/qwen-serving.install.log" 2>&1
"$QWEN_ENV/bin/python" -m pip freeze >"$EVIDENCE_ROOT/qwen-serving.pip-freeze.txt"
"$QWEN_ENV/bin/python" -c \
  "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); import vllm; print('vllm', getattr(vllm, '__version__', 'unknown')); import llmcompressor; print('llmcompressor', getattr(llmcompressor, '__version__', 'unknown'))" \
  >"$EVIDENCE_ROOT/qwen-serving.probe.log" 2>&1
write_status qwen-serving ready
echo "environment_bootstrap_complete_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

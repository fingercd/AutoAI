#!/usr/bin/env bash
set -Eeuo pipefail

# The node3 host uses glibc 2.17, while the official vLLM 0.17 wheel uses
# newer symbols.  Keep the compatibility loader user-scoped and leave the
# host libc, CUDA driver, and other processes untouched.
VLLM_ENV="${AUTOAI_VLLM_ENV:-/users/fotile/AutoAI/shared/envs/qwen-serving-vllm017}"
VLLM_PYTHON="${AUTOAI_VLLM_PYTHON:-$VLLM_ENV/bin/python-glibc}"
QWEN_BASE_ENV="${AUTOAI_QWEN_BASE_ENV:-/users/fotile/AutoAI/shared/envs/qwen-serving}"
GLIBC_ROOT="${AUTOAI_GLIBC_ROOT:-/users/fotile/AutoAI/shared/work/agent_v1_poc/envs/glibc/root}"
LOADER="${AUTOAI_GLIBC_LOADER:-$GLIBC_ROOT/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2}"
LIBRARY_PATH="${AUTOAI_GLIBC_LIBRARY_PATH:-$GLIBC_ROOT/lib/x86_64-linux-gnu:$VLLM_ENV/lib:$QWEN_BASE_ENV/lib:/users/fotile/AutoAI/envs/autoai-app/lib:/usr/lib64:/lib64}"
RUNTIME_LIBRARY_PATH="${AUTOAI_VLLM_RUNTIME_LIBRARY_PATH:-$VLLM_ENV/lib:$QWEN_BASE_ENV/lib:/users/fotile/AutoAI/envs/autoai-app/lib:/usr/lib64:/lib64}"
# Triton compiles a small CUDA helper at runtime.  The host GCC 4.8 defaults
# to GNU89 and rejects the C99 loop declarations used by Triton 0.17; use the
# already-installed user-scoped conda GCC 12 when available.
TRITON_CC="${AUTOAI_TRITON_CC:-/users/fotile/miniconda3/envs/gcc12/bin/x86_64-conda-linux-gnu-gcc}"
if [ -z "${CC:-}" ] && [ -x "$TRITON_CC" ]; then
  export CC="$TRITON_CC"
fi
# Keep the vLLM/torch site-packages first, then reuse compatible runtime-only
# packages already installed in the isolated AutoAI and qwen-serving envs.
# This avoids copying large NumPy/Pydantic trees and never lets their torch
# installations shadow the dedicated torch 2.10 tree above.
PYTHONPATH="${AUTOAI_VLLM_PYTHONPATH:-$VLLM_ENV/lib/python3.12/site-packages:/users/fotile/AutoAI/envs/autoai-app/lib/python3.12/site-packages:/users/fotile/AutoAI/shared/envs/qwen-serving/lib/python3.12/site-packages}"
export PYTHONPATH
# vLLM launches an architecture-inspection subprocess using sys.executable.
# Export the same user-scoped library path so that child process does not fall
# back to the host glibc 2.17.  The current process still uses the explicit
# loader invocation below.
export LD_LIBRARY_PATH="$RUNTIME_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

test -x "$LOADER"
test -x "$VLLM_PYTHON"
exec "$LOADER" --library-path "$LIBRARY_PATH" "$VLLM_PYTHON" \
  -m vllm.entrypoints.cli.main "$@"

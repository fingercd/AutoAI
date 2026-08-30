#!/usr/bin/env bash
set -Eeuo pipefail

# Resumable, serial ModelScope downloader. It intentionally leaves ModelScope
# temporary files in place after interruption so the same command can resume.
ROOT="${AUTOAI_SHARED_ROOT:-/users/fotile/AutoAI/shared}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="$ROOT/models/original"
MANIFEST_ROOT="$ROOT/manifests/models"
EVIDENCE_ROOT="$ROOT/work/agent_v1_poc/model_download"
LOG_ROOT="$EVIDENCE_ROOT/logs"
STATUS_ROOT="$EVIDENCE_ROOT/status"
MODELSCOPE_MAX_WORKERS="${AUTOAI_MODELSCOPE_MAX_WORKERS:-1}"
MODELSCOPE_RETRY_ARGS=(--max-workers "$MODELSCOPE_MAX_WORKERS")
mkdir -p "$MODEL_ROOT" "$MANIFEST_ROOT" "$LOG_ROOT" "$STATUS_ROOT"

SCRIPT_PID_FILE="$EVIDENCE_ROOT/script.pid"
FINAL_MARKER="$EVIDENCE_ROOT/complete.ok"
FAIL_MARKER="$EVIDENCE_ROOT/failed"
printf '%s\n' "$$" > "$SCRIPT_PID_FILE"
rm -f "$FINAL_MARKER" "$FAIL_MARKER"

write_status() {
  local key="$1" phase="$2" pid="${3:-}" exit_code="${4:-}" message="${5:-}"
  local model_dir
  case "$key" in
    qwen35_9b) model_dir="$MODEL_ROOT/qwen3.5-9b" ;;
    qwen35_27b) model_dir="$MODEL_ROOT/qwen3.5-27b" ;;
    qwen38_27b) model_dir="$MODEL_ROOT/qwen3.8-27b" ;;
    *) model_dir="$MODEL_ROOT/$key" ;;
  esac
  local manifest="$MANIFEST_ROOT/$key.json" bytes=0 expected=0
  if [[ -d "$model_dir" ]]; then
    if [[ -f "$manifest" ]]; then
      bytes="$(python3 - "$model_dir" "$manifest" <<'PY'
import json
import os
import sys

root, manifest_path = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
print(sum(os.path.getsize(os.path.join(root, item["path"])) for item in manifest["files"] if os.path.isfile(os.path.join(root, item["path"]))))
PY
)"
    else
      bytes="$(du -sb "$model_dir" 2>/dev/null | awk '{print $1}' || printf '0')"
    fi
  fi
  if [[ -f "$manifest" ]]; then
    expected="$(python3 - "$manifest" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(sum(int(item["size"]) for item in json.load(handle)["files"]))
PY
)"
  fi
  local tmp="$STATUS_ROOT/.${key}.json.tmp"
  python3 - "$tmp" "$key" "$phase" "$pid" "$exit_code" "$bytes" "$expected" "$message" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, key, phase, pid, exit_code, bytes_done, expected, message = sys.argv[1:]
payload = {
    "schema_version": "autoai-model-download-status-v1",
    "model_key": key,
    "phase": phase,
    "pid": int(pid) if pid.isdigit() else None,
    "exit_code": int(exit_code) if exit_code.lstrip("-").isdigit() else None,
    "bytes_downloaded_on_disk": int(bytes_done or 0),
    "expected_bytes": int(expected or 0),
    "progress_ratio": (int(bytes_done) / int(expected)) if expected and int(expected) else None,
    "message": message,
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY
  mv -f "$tmp" "$STATUS_ROOT/$key.json"
}

cleanup() {
  local rc="$?"
  if [[ "$rc" -eq 0 ]]; then
    touch "$FINAL_MARKER"
  else
    printf '%s\n' "$rc" > "$FAIL_MARKER"
  fi
  rm -f "$SCRIPT_PID_FILE"
  exit "$rc"
}
trap cleanup EXIT

MODELS=(qwen35_9b qwen35_27b qwen38_27b)
MODEL_IDS=(Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B Qwen/Qwen3.8-27B)
MODEL_DIRS=(qwen3.5-9b qwen3.5-27b qwen3.8-27b)

echo "download_script_pid=$$"
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

for i in "${!MODELS[@]}"; do
  key="${MODELS[$i]}"
  model_id="${MODEL_IDS[$i]}"
  manifest="$MANIFEST_ROOT/$key.json"
  if [[ -f "$manifest" ]]; then
    echo "manifest_already_frozen key=$key path=$manifest"
  else
    echo "freezing_manifest key=$key model=$model_id"
    python3 "$SCRIPT_DIR/fetch_modelscope_manifest.py" --model-key "$key" --output "$manifest" \
      2>&1 | tee "$LOG_ROOT/${key}.manifest.log"
  fi
  write_status "$key" "manifest_frozen" "$$" "0" "manifest_frozen"
done

ensure_hidden_metadata() {
  local model_id="$1" model_dir="$2" manifest="$3"
  if python3 - "$manifest" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if any(item["path"] == ".gitattributes" for item in manifest["files"]) else 1)
PY
  then
    if [[ ! -f "$model_dir/.gitattributes" ]]; then
      echo "fetch_hidden_metadata model=$model_id file=.gitattributes"
      curl --fail --location --retry 8 --retry-all-errors --connect-timeout 30 --max-time 600 \
        -o "$model_dir/.gitattributes" \
        "https://modelscope.cn/models/$model_id/resolve/master/.gitattributes"
    fi
  fi
}

for i in "${!MODELS[@]}"; do
  key="${MODELS[$i]}"
  model_id="${MODEL_IDS[$i]}"
  model_dir="$MODEL_ROOT/${MODEL_DIRS[$i]}"
  manifest="$MANIFEST_ROOT/$key.json"
  log="$LOG_ROOT/${key}.download.log"
  report="$STATUS_ROOT/${key}.verification.json"
  mkdir -p "$model_dir"

  # A previous persistent attempt may already have produced a complete,
  # hash-verified snapshot.  Re-verify that exact directory first so a retry
  # does not needlessly re-enter the ModelScope CLI or touch valid blobs.
  if python3 "$SCRIPT_DIR/verify_model_snapshot.py" --root "$model_dir" --manifest "$manifest" --report "$report" \
    >"$LOG_ROOT/${key}.preverify.log" 2>&1; then
    echo "snapshot_already_verified key=$key directory=$model_dir"
    write_status "$key" "verified" "$$" "0" "reused existing streaming SHA-256 verification"
    touch "$STATUS_ROOT/${key}.complete.ok"
    continue
  fi

  write_status "$key" "starting" "$$" "" "starting ModelScope download"
  echo "download_start key=$key model=$model_id directory=$model_dir"

  modelscope download --model "$model_id" --revision master --local_dir "$model_dir" \
    "${MODELSCOPE_RETRY_ARGS[@]}" \
    >"$log" 2>&1 &
  download_pid="$!"
  write_status "$key" "downloading" "$download_pid" "" "resumable ModelScope CLI download"
  while kill -0 "$download_pid" 2>/dev/null; do
    write_status "$key" "downloading" "$download_pid" "" "byte progress sampled from snapshot"
    sleep 30
  done
  set +e
  wait "$download_pid"
  download_rc="$?"
  set -e
  write_status "$key" "download_finished" "$download_pid" "$download_rc" "ModelScope CLI exited"
  if [[ "$download_rc" -ne 0 ]]; then
    echo "download_failed key=$key exit_code=$download_rc log=$log" >&2
    exit "$download_rc"
  fi

  ensure_hidden_metadata "$model_id" "$model_dir" "$manifest"

  set +e
  python3 "$SCRIPT_DIR/verify_model_snapshot.py" --root "$model_dir" --manifest "$manifest" --report "$report" \
    2>&1 | tee "$LOG_ROOT/${key}.verify.log"
  verify_rc="${PIPESTATUS[0]}"
  set -e
  write_status "$key" "verified" "$$" "$verify_rc" "streaming SHA-256 verification completed"
  if [[ "$verify_rc" -ne 0 ]]; then
    echo "verification_failed key=$key report=$report" >&2
    exit "$verify_rc"
  fi
  touch "$STATUS_ROOT/${key}.complete.ok"
done

echo "all_model_downloads_verified_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

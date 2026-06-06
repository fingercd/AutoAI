#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/AutoAI}"
PORT="${PORT:-8000}"
LOG_DIR="$APP_DIR/storage/logs"
PID_FILE="$APP_DIR/storage/autoai.pid"

mkdir -p "$LOG_DIR"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "AutoAI already running with PID $(cat "$PID_FILE")"
  exit 0
fi

cd "$APP_DIR"
nohup bash "$APP_DIR/deploy/run_on_node3.sh" > "$LOG_DIR/autoai.out.log" 2> "$LOG_DIR/autoai.err.log" &
echo $! > "$PID_FILE"
echo "AutoAI started with PID $(cat "$PID_FILE") on port $PORT"

#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/AutoAI}"
PID_FILE="$APP_DIR/storage/autoai.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "No PID file: $PID_FILE"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "Stopped AutoAI PID $PID"
else
  echo "PID $PID is not running"
fi
rm -f "$PID_FILE"

#!/usr/bin/env bash
# Stop only the Web process recorded by start_persistent_node3.sh. A separately managed
# training worker must be stopped through its own process manager.
#
# 【中文说明】与 start_persistent_node3.sh 配对的停止脚本：只停止 PID 文件中
# 记录的 Web 进程；独立托管的训练 worker 需通过其自身的进程管理方式停止。
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/AutoAI}"
PID_FILE="$APP_DIR/storage/autoai.pid"

# 没有 PID 文件视为“未在运行”，正常退出（幂等，重复执行安全）。
if [ ! -f "$PID_FILE" ]; then
  echo "No PID file: $PID_FILE"
  exit 0
fi

PID="$(cat "$PID_FILE")"
# kill -0 只探测进程是否存在，并不真正发送信号；确认存在后才发 SIGTERM 优雅停止。
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "Stopped SpecAutoAI PID $PID"
else
  echo "PID $PID is not running"
fi
# 无论进程是否存在都清理 PID 文件，避免陈旧状态影响下次启动时的“已在运行”判断。
rm -f "$PID_FILE"

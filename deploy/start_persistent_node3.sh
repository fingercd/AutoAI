#!/usr/bin/env bash
# Minimal Web PID-file wrapper for environments without systemd. Training worker
# lifecycle remains independent; see server_deploy.md.
#
# 【中文说明】无 systemd 环境下的简易“常驻”启动包装：通过 nohup 后台拉起
# deploy/run_on_node3.sh（Web 服务），并把其 PID 写入 PID 文件，供
# stop_persistent_node3.sh 精确停止。训练 worker 的生命周期独立管理，
# 不在本脚本职责范围内（见 deploy/server_deploy.md）。
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/AutoAI}"
PORT="${PORT:-8000}"
LOG_DIR="$APP_DIR/storage/logs"
PID_FILE="$APP_DIR/storage/autoai.pid"

mkdir -p "$LOG_DIR"

# A PID file is trusted only when the process still exists; stale files are overwritten.
# 中文：PID 文件只有在对应进程仍然存在（kill -0 探测）时才可信；
# 进程已死的陈旧 PID 文件直接忽略并覆盖，避免误判“已在运行”。
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "SpecAutoAI already running with PID $(cat "$PID_FILE")"
  exit 0
fi

cd "$APP_DIR"
# nohup + &：脱离终端后台运行，stdout/stderr 分别落盘到日志，便于事后排查。
nohup bash "$APP_DIR/deploy/run_on_node3.sh" > "$LOG_DIR/autoai.out.log" 2> "$LOG_DIR/autoai.err.log" &
# $! 是刚后台化进程的 PID；run_on_node3.sh 内部用 exec 把自身替换为 uvicorn，
# 因此该 PID 最终就是 Web 进程本身的 PID，停止脚本可直接对其发信号。
echo $! > "$PID_FILE"
echo "SpecAutoAI started with PID $(cat "$PID_FILE") on port $PORT"

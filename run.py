"""
SpecAutoAI 一键启动脚本

用法:
    python run.py                # 默认 127.0.0.1:8000，自动打开浏览器
    python run.py --port 9000    # 自定义端口
    python run.py --no-browser   # 不自动打开浏览器
    python run.py --server --host 0.0.0.0 # 受令牌保护的服务器模式

本地前端快捷入口:
    python run_classic.py        # 启动服务并打开经典前端
    python run_v2.py             # 启动服务并打开 v2 工作台

PyCharm 使用:
    右键 run.py → Run / Debug 即可启动，浏览器会自动打开。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

CLASSIC_FRONTEND_PATH = "/"
V2_FRONTEND_PATH = "/v2"


def build_frontend_url(base_url: str, frontend_path: str) -> str:
    """将受支持的前端入口拼到同一个 FastAPI 服务地址上。"""
    if frontend_path not in {CLASSIC_FRONTEND_PATH, V2_FRONTEND_PATH}:
        raise ValueError(f"不支持的前端入口: {frontend_path}")
    return f"{base_url.rstrip('/')}{frontend_path}"


class WorkerSupervisor:
    """有限退避重启由本启动器托管的独立 worker 子进程。"""

    def __init__(self, command: list[str], *, max_fast_restarts: int = 5) -> None:
        self.command = list(command)
        self.max_fast_restarts = max_fast_restarts
        self._process: subprocess.Popen[bytes] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._fast_restarts = 0

    def _spawn(self) -> None:
        self._process = subprocess.Popen(self.command)
        self._started_at = time.monotonic()

    def start(self) -> None:
        self._spawn()

        def monitor() -> None:
            while not self._stop.wait(1.0):
                process = self._process
                if process is None or process.poll() is None:
                    continue
                lived_seconds = time.monotonic() - self._started_at
                if lived_seconds >= 60:
                    self._fast_restarts = 0
                else:
                    self._fast_restarts += 1
                if self._fast_restarts > self.max_fast_restarts:
                    print('\n  ⚠️ 训练 worker 连续退出，已停止自动重启；请检查日志和 /health。\n')
                    self._process = None
                    return
                delay = min(2 ** max(self._fast_restarts - 1, 0), 10)
                print(f'\n  ⚠️ 训练 worker 已退出，{delay} 秒后尝试重新启动。\n')
                if self._stop.wait(delay):
                    return
                self._spawn()

        self._thread = threading.Thread(target=monitor, name='autoai-worker-supervisor', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=2)


def main(*, frontend_path: str = CLASSIC_FRONTEND_PATH) -> None:
    """解析本地启动参数，并在同一生命周期管理 Web 服务与可选 worker。"""
    parser = argparse.ArgumentParser(description="SpecAutoAI 谱学建模平台启动器")
    parser.add_argument("--host", default="127.0.0.1", help="绑定地址 (默认 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="绑定端口 (默认 8000)")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--reload", action="store_true", help="开启热重载（开发用）")
    parser.add_argument("--no-worker", action="store_true", help="不启动本地训练 worker")
    parser.add_argument(
        "--server",
        action="store_true",
        help="启用服务器安全模式（必须通过环境变量设置 AUTOAI_API_TOKEN）",
    )
    args = parser.parse_args()

    if args.server:
        os.environ['AUTOAI_DEPLOYMENT_MODE'] = 'server'
    deployment_mode = os.environ.get('AUTOAI_DEPLOYMENT_MODE', 'local').strip().lower()
    from backend.app.http.security import is_loopback_host, load_security_settings
    from backend.app.version import WORKER_CONTRACT_VERSION

    if not is_loopback_host(args.host) and deployment_mode != 'server':
        parser.error('对外绑定必须同时启用 --server，并通过环境变量配置访问令牌')
    # 在创建 worker 或 Web 服务前验证设置；异常不会包含 token 内容。
    try:
        security_settings = load_security_settings()
    except RuntimeError as exc:
        parser.error(str(exc))

    import uvicorn

    browser_host = '127.0.0.1' if args.host in {'0.0.0.0', '::'} else args.host
    base_url = f"http://{browser_host}:{args.port}"
    frontend_url = build_frontend_url(base_url, frontend_path)

    if not args.no_browser:
        # 在另一个线程打开浏览器，避免阻塞 uvicorn 启动
        import threading

        def _open_browser() -> None:
            import time
            time.sleep(1.5)  # 等 uvicorn 启动
            webbrowser.open(frontend_url)
            print(f"\n  🌐 浏览器已打开: {frontend_url}\n")

        threading.Thread(target=_open_browser, daemon=True).start()

    print(f"""
  ╔══════════════════════════════════════════════╗
  ║       🧪 SpecAutoAI 谱学建模平台              ║
  ║                                              ║
  ║  前端入口: {frontend_url:<34}║
  ║  API 文档: {base_url + '/docs':<34}║
  ║  健康检查: {base_url + '/health':<34}║
  ║                                              ║
  ║  部署模式: {security_settings.mode:<34}║
  ║  结果契约: {WORKER_CONTRACT_VERSION:<34}║
  ║  按 Ctrl+C 停止服务                          ║
  ╚══════════════════════════════════════════════╝
""")

    worker: WorkerSupervisor | None = None
    if not args.no_worker:
        # Web 只负责创建 queued Run；训练能力依赖这个独立子进程。使用当前
        # 解释器可确保 worker 与 uvicorn 共享同一虚拟环境和依赖集。
        worker = WorkerSupervisor(
            [
                sys.executable,
                "-m",
                "backend.app.runs.worker",
                "--contract-version",
                WORKER_CONTRACT_VERSION,
            ]
        )
        worker.start()
    try:
        uvicorn.run(
            "backend.app.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    finally:
        # 无论正常退出还是 Ctrl+C，都回收由本启动器创建的 worker，避免残留
        # 进程继续领取新 Run。外部独立托管的 worker 不受这里影响。
        if worker is not None:
            worker.stop()


if __name__ == "__main__":
    main()

"""
SpecAutoAI 一键启动脚本

用法:
    python run.py                # 默认 127.0.0.1:8000，自动打开浏览器
    python run.py --port 9000    # 自定义端口
    python run.py --no-browser   # 不自动打开浏览器
    python run.py --host 0.0.0.0 # 允许局域网访问

PyCharm 使用:
    右键 run.py → Run / Debug 即可启动，浏览器会自动打开。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """解析本地启动参数，并在同一生命周期管理 Web 服务与可选 worker。"""
    parser = argparse.ArgumentParser(description="SpecAutoAI 谱学建模平台启动器")
    parser.add_argument("--host", default="127.0.0.1", help="绑定地址 (默认 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="绑定端口 (默认 8000)")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--reload", action="store_true", help="开启热重载（开发用）")
    parser.add_argument("--no-worker", action="store_true", help="不启动本地训练 worker")
    args = parser.parse_args()

    import uvicorn

    url = f"http://{args.host}:{args.port}"

    if not args.no_browser:
        # 在另一个线程打开浏览器，避免阻塞 uvicorn 启动
        import threading

        def _open_browser() -> None:
            import time
            time.sleep(1.5)  # 等 uvicorn 启动
            webbrowser.open(url)
            print(f"\n  🌐 浏览器已打开: {url}\n")

        threading.Thread(target=_open_browser, daemon=True).start()

    print(f"""
  ╔══════════════════════════════════════════════╗
  ║       🧪 SpecAutoAI 谱学建模平台              ║
  ║                                              ║
  ║  本地访问: {url}                     ║
  ║  API 文档: {url}/docs                      ║
  ║  健康检查: {url}/health                    ║
  ║                                              ║
  ║  按 Ctrl+C 停止服务                          ║
  ╚══════════════════════════════════════════════╝
""")

    worker = None
    if not args.no_worker:
        # Web 只负责创建 queued Run；训练能力依赖这个独立子进程。使用当前
        # 解释器可确保 worker 与 uvicorn 共享同一虚拟环境和依赖集。
        worker = subprocess.Popen([sys.executable, "-m", "backend.app.runs.worker"])
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
        if worker is not None and worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=10)


if __name__ == "__main__":
    main()

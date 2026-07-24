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

# ============================================================================
# 模块级说明（教学注释）
# ----------------------------------------------------------------------------
# 本文件是 SpecAutoAI 的“一键启动器”，处于整个系统的最外层入口，负责把
# FastAPI 后端（backend.app.main:app）以 uvicorn 形式拉起，并可选地：
#   1. 自动打开浏览器（经典前端 "/" 或 v2 工作台 "/v2"）；
#   2. 启动并托管一个独立的训练 worker 子进程（backend.app.runs.worker）。
#
# 与系统其它部分的关系：
#   - Web 进程只负责创建 queued Run，真正的模型训练由 worker 进程领取执行，
#     二者通过 storage/ 下的运行队列解耦（BackgroundTasks 不承担训练执行）；
#   - run_classic.py / run_v2.py 是本模块 main() 的薄封装，仅切换默认前端入口；
#   - 安全设置由 backend.app.http.security.load_security_settings() 统一校验。
#
# 关键设计约束：
#   - 绑定非回环地址（如 0.0.0.0）属于对外暴露，必须启用 server 模式
#     （--server 或 AUTOAI_DEPLOYMENT_MODE=server），并配置足够长度的
#     AUTOAI_API_TOKEN；token 只存在于环境变量，报错信息不含其内容；
#   - worker 使用当前解释器（sys.executable）启动，保证与 Web 进程共享
#     同一虚拟环境和依赖集；
#   - 无论 uvicorn 正常退出还是被 Ctrl+C 中断，finally 块都会回收本启动器
#     创建的 worker，避免残留进程继续领取新 Run。
# ============================================================================

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
WORKER_STOP_EXIT_CODE = 75


def build_frontend_url(base_url: str, frontend_path: str) -> str:
    """将受支持的前端入口拼到同一个 FastAPI 服务地址上。

    参数:
        base_url: 形如 ``http://127.0.0.1:8000`` 的服务根地址。
        frontend_path: 前端入口路径，只允许 ``/``（经典前端）或 ``/v2``（v2 工作台）。

    返回:
        拼接后的完整前端 URL。

    异常:
        ValueError: 传入白名单之外的入口路径时抛出，防止把任意路径当成前端打开。
    """
    # 白名单校验：前端入口是固定的两个常量，防御未来调用方传入未审查的路径。
    if frontend_path not in {CLASSIC_FRONTEND_PATH, V2_FRONTEND_PATH}:
        raise ValueError(f"不支持的前端入口: {frontend_path}")
    # rstrip('/') 兜底 base_url 末尾多写的斜杠，避免出现 "http://host:8000//v2"。
    return f"{base_url.rstrip('/')}{frontend_path}"


class WorkerSupervisor:
    """有限退避重启由本启动器托管的独立 worker 子进程。

    设计意图：
        训练 worker（``backend.app.runs.worker``）是真正执行模型训练的进程，
        可能因数据或环境问题崩溃。本类在后台守护线程中监视它：

        - 进程存活超过 60 秒后才退出，视为“正常跑过一段”，快速重启计数清零；
        - 60 秒内就退出视为“快速失败”，累计超过 ``max_fast_restarts`` 次后
          放弃重启（说明存在持续性错误，无限重启只会刷爆日志）；
        - 重启间隔按 1, 2, 4, ... 秒指数退避，封顶 10 秒，给系统留出恢复时间。

    线程模型:
        ``start()`` 启动一个 daemon 守护线程执行 monitor 循环；``_stop`` 事件
        用于让 ``stop()`` 能即时唤醒并终止该循环（Event.wait 兼作可中断 sleep）。
    """

    def __init__(self, command: list[str], *, max_fast_restarts: int = 5) -> None:
        # list(command) 拷贝一份，避免外部后续修改列表影响子进程命令。
        self.command = list(command)
        self.max_fast_restarts = max_fast_restarts
        self._process: subprocess.Popen[bytes] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # _started_at：最近一次 spawn 的单调时钟起点；_fast_restarts：连续“快速失败”次数。
        self._started_at = 0.0
        self._fast_restarts = 0

    def _spawn(self) -> None:
        """拉起一次 worker 子进程，并记录启动时间（供快速失败判定使用）。"""
        self._process = subprocess.Popen(self.command)
        # 用单调时钟（monotonic）而非 wall clock，避免系统时间被校时修改后
        # 导致“存活时长”计算出现负值或跳变。
        self._started_at = time.monotonic()

    def start(self) -> None:
        """启动 worker 并开启后台监视线程（立即返回，不阻塞调用方）。"""
        self._spawn()

        def monitor() -> None:
            # 每秒轮询一次；Event.wait 兼作可中断的 sleep，stop() 调用能立刻退出循环。
            while not self._stop.wait(1.0):
                process = self._process
                # poll() 返回 None 表示子进程仍在运行，无需处理。
                return_code = None if process is None else process.poll()
                if process is None or return_code is None:
                    continue
                if return_code == WORKER_STOP_EXIT_CODE:
                    # 用户 STOP 会让 worker 主动退出，以打断无法协作取消的原生计算。
                    # 这是预期控制流，不计入“快速失败”，立即拉起干净 worker。
                    self._fast_restarts = 0
                    print('\n  ⏹️ 当前训练已停止，正在重新启动训练 worker。\n')
                    self._spawn()
                    continue
                lived_seconds = time.monotonic() - self._started_at
                if lived_seconds >= 60:
                    # 存活超过 60 秒：视为曾经的正常运行，重置快速失败计数。
                    self._fast_restarts = 0
                else:
                    # 60 秒内退出：累计一次“快速失败”。
                    self._fast_restarts += 1
                if self._fast_restarts > self.max_fast_restarts:
                    # 连续快速失败说明存在持续性错误，放弃重启并提示人工排查。
                    print('\n  ⚠️ 训练 worker 连续退出，已停止自动重启；请检查日志和 /health。\n')
                    self._process = None
                    return
                # 指数退避：1, 2, 4, 8, 10, 10 ... 秒，封顶 10 秒。
                delay = min(2 ** max(self._fast_restarts - 1, 0), 10)
                print(f'\n  ⚠️ 训练 worker 已退出，{delay} 秒后尝试重新启动。\n')
                # 退避等待同样可中断：若期间 stop() 被调用则直接退出，不再重启。
                if self._stop.wait(delay):
                    return
                self._spawn()

        # daemon=True：主进程退出时不必等监视线程，配合 stop() 的 join 做优雅收尾。
        self._thread = threading.Thread(target=monitor, name='autoai-worker-supervisor', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止监视并回收 worker 子进程（幂等，可在退出路径中安全调用）。"""
        self._stop.set()
        process = self._process
        if process is not None and process.poll() is None:
            # 先 SIGTERM（terminate）给 worker 优雅退出的机会……
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # ……10 秒仍未退出则强杀（kill），避免关闭流程被卡死。
                process.kill()
                process.wait(timeout=5)
        # 等监视线程收尾，最多 2 秒；它是 daemon 线程，超时也不阻塞进程退出。
        if self._thread is not None:
            self._thread.join(timeout=2)


def main(*, frontend_path: str = CLASSIC_FRONTEND_PATH) -> None:
    """解析本地启动参数，并在同一生命周期管理 Web 服务与可选 worker。

    参数:
        frontend_path: 启动后自动打开的前端入口（``/`` 或 ``/v2``），仅允许
            关键字传参；run_classic.py / run_v2.py 通过它复用本函数。

    流程:
        解析参数 → 推导/校验部署模式与安全设置 → 打印启动信息并打开浏览器
        → 启动训练 worker（WorkerSupervisor 托管）→ uvicorn 阻塞运行
        → finally 回收 worker。
    """
    parser = argparse.ArgumentParser(description="SpecAutoAI 谱学建模平台启动器")
    # 以下均为可选开关：默认绑定回环地址 127.0.0.1:8000，面向单机使用；
    # 需要对外提供服务的场景必须显式使用 --server 并配合环境变量令牌。
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

    # --server 只是写入环境变量的快捷方式；真正的判定统一读取
    # AUTOAI_DEPLOYMENT_MODE，这样直接设置环境变量启动也走同一条路径。
    if args.server:
        os.environ['AUTOAI_DEPLOYMENT_MODE'] = 'server'
    deployment_mode = os.environ.get('AUTOAI_DEPLOYMENT_MODE', 'local').strip().lower()
    # 延迟导入：只有真正启动服务时才加载 backend 包，让 --help 等轻量路径保持快速。
    from backend.app.http.security import is_loopback_host, load_security_settings
    from backend.app.version import WORKER_CONTRACT_VERSION

    # 安全红线：绑定非回环地址（对外可达）却不启用 server 模式，直接拒绝启动——
    # local 模式没有令牌保护，绝不能暴露到网络。
    if not is_loopback_host(args.host) and deployment_mode != 'server':
        parser.error('对外绑定必须同时启用 --server，并通过环境变量配置访问令牌')
    # 在创建 worker 或 Web 服务前验证设置；异常不会包含 token 内容。
    # 中文补充：load_security_settings() 在 server 模式下会强制校验
    # AUTOAI_API_TOKEN 长度（≥32）与 CORS 来源等；校验失败在此就终止启动。
    try:
        security_settings = load_security_settings()
    except RuntimeError as exc:
        parser.error(str(exc))

    import uvicorn

    # 绑定 0.0.0.0 / :: 表示“监听所有网卡”，并不是可访问的地址；
    # 浏览器需要具体地址，因此打开浏览器时统一回退到本机回环 127.0.0.1。
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
        # --contract-version 把后端的结果契约版本（WORKER_CONTRACT_VERSION）
        # 显式传给 worker，避免 worker 与 Web 对 run-result 契约理解不一致。
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

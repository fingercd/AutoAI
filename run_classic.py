"""启动 SpecAutoAI 服务并打开经典前端。

模块说明：
    本模块是 ``run.py`` 的薄封装（thin wrapper），本身不含任何服务逻辑：
    参数解析、安全校验、uvicorn 启动与训练 worker 托管全部复用 ``run.main``，
    仅把默认前端入口固定为经典前端 ``/``（对应后端静态托管的
    ``static/index.html``，该入口受既有字符串契约测试保护）。

用法：
    python run_classic.py       # 等价于 run.py 的默认行为
"""

from run import CLASSIC_FRONTEND_PATH, main


# 直接转发到 run.main，仅传入经典前端路径常量；这也是 run.main 的默认参数，
# 复用公共启动器的服务与 Worker 生命周期。
if __name__ == "__main__":
    main(frontend_path=CLASSIC_FRONTEND_PATH)

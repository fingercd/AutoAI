"""启动 SpecAutoAI 服务并打开 v2 独立工作台。

模块说明：
    本模块是 ``run.py`` 的薄封装（thin wrapper），本身不含任何服务逻辑：
    参数解析、安全校验、uvicorn 启动与训练 worker 托管全部复用 ``run.main``，
    仅把默认前端入口切换为 v2 工作台路径 ``/v2``（对应后端静态托管的
    ``static/v2/index.html`` 独立工作台，复用 ``static/js/api-client.js``
    与 run-result-v1 结果契约）。

用法：
    python run_v2.py            # 等价于 run.py，但浏览器自动打开 /v2
"""

from run import V2_FRONTEND_PATH, main


# 直接转发到 run.main，仅传入 v2 前端路径常量；其余行为与 `python run.py` 完全一致。
if __name__ == "__main__":
    main(frontend_path=V2_FRONTEND_PATH)

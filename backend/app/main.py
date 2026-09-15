"""SpecAutoAI FastAPI 应用装配入口。

该模块只完成目录初始化、中间件、静态资源和路由注册。训练不会在请求线程或
FastAPI BackgroundTasks 中执行；`POST /api/training/runs` 创建的任务由独立
`backend.app.runs.worker` 进程领取。

【模块级说明】
- 职责：把存储目录、安全/CORS 中间件、静态文件挂载和各业务 router 装配成
  一个可运行的 FastAPI 应用实例 `app`；本文件不含任何业务处理逻辑。
- 系统位置：进程入口 `run.py`（或 uvicorn 命令行）以 `backend.app.main:app`
  导入本模块；导入瞬间即完成整个服务的初始化，因此模块顶层每一行都会在
  服务启动时执行一次。
- 协作模块：
  * `.paths`：存储目录常量与 `ensure_storage()`；
  * `.http.security`：`ServerAuthMiddleware`（校验 token、注入 Principal 身份）
    与 `load_security_settings()`；
  * `.routers`：auth（健康检查/登录）、catalog（能力目录/首页）、datasets
    （数据集管理）、preprocess（拉曼/HPLC 预处理）、runs（训练 Run 管理）。
- 关键设计约束：
  * 训练请求只创建 queued 状态的 Run 记录，绝不在请求线程或
    BackgroundTasks 里跑训练，避免长任务阻塞事件循环；
  * local/server 两种部署模式的行为差异由 security_settings 统一驱动，
    本文件除 CORS 正则外不做模式分支。
"""

from __future__ import annotations

# FastAPI 核心类与官方中间件/静态文件组件。
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# 项目内部依赖：路径/目录初始化、安全中间件，以及按业务域拆分的五组 router。
from .paths import RUNS_DIR, STATIC_DIR, ensure_storage
from .http.security import ServerAuthMiddleware, load_security_settings
from .routers import agent, auth, catalog, datasets, preprocess, runs


# 导入应用时先保证静态文件、上传和 Run 路径存在，路由随后即可安全写入。
ensure_storage()
# 从环境变量读取部署模式（local/server）、API token、允许的 CORS 来源等配置；
# 启动时加载一次后冻结，运行期不再变化，保证鉴权行为可预期。
security_settings = load_security_settings()
# 创建应用实例；title/version 会出现在 /docs 与 OpenAPI schema 中。
app = FastAPI(title='SpecAutoAI', version='0.1.0')
# 把安全配置挂到 app.state，供中间件和路由在请求处理期间读取同一份配置。
app.state.security_settings = security_settings
# 鉴权中间件：解析并校验浏览器 token、注入 Principal，业务路由拿到的身份
# 只来自这里，请求体不允许传入 owner_id/tenant_id，实现按 Principal 的数据隔离。
app.add_middleware(ServerAuthMiddleware, settings=security_settings)
# 注意 Starlette 中后注册的中间件位于更外层、先执行：CORS 在鉴权外层，
# 使不带 token 的 OPTIONS 预检请求能被正常响应，鉴权失败也不会缺 CORS 头。
app.add_middleware(
    CORSMiddleware,
    # 显式允许的来源列表：server 模式必须配置明确来源；local 模式通常为空。
    allow_origins=list(security_settings.allowed_origins),
    # 仅 local 模式额外放行任意端口的 127.0.0.1/localhost，方便本机开发调试；
    # server 模式置 None，只信任显式配置的来源，避免宽松正则带来跨域风险。
    allow_origin_regex=(
        r'^https?://(127\.0\.0\.1|localhost)(:\d+)?$'
        if security_settings.mode == 'local'
        else None
    ),
    # 浏览器 token 放在 Authorization 头而非 Cookie，故必须关闭 credentials，
    # 浏览器才会允许前端读取跨域响应。
    allow_credentials=False,
    allow_methods=['*'],
    # 只放行实际用到的请求头，收紧跨域攻击面。
    allow_headers=['Authorization', 'Content-Type', 'Accept'],
)

# 静态资源与 API 同源托管；根路由本身由 catalog router 返回 index.html。
# 目录存在才挂载：缺少前端静态目录的环境（如仅跑 API 的节点）不会因导入报错。
if STATIC_DIR.exists():
    app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')

# 注册五组业务 router；catalog 在最前，包含根路径 '/' 与能力目录接口，
# 其余分别负责登录、数据集、预处理和训练 Run 的 HTTP 端点。
app.include_router(catalog.router)
app.include_router(auth.router)
app.include_router(datasets.router)
app.include_router(preprocess.router)
app.include_router(runs.router)
app.include_router(agent.router)
app.include_router(agent.current_router)

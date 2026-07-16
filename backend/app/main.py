"""SpecAutoAI FastAPI 应用装配入口。

该模块只完成目录初始化、中间件、静态资源和路由注册。训练不会在请求线程或
FastAPI BackgroundTasks 中执行；`POST /api/training/runs` 创建的任务由独立
`backend.app.runs.worker` 进程领取。
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .paths import RUNS_DIR, STATIC_DIR, ensure_storage
from .http.security import ServerAuthMiddleware, load_security_settings
from .routers import auth, catalog, datasets, preprocess, runs


# 导入应用时先保证静态文件、上传和 Run 路径存在，路由随后即可安全写入。
ensure_storage()
security_settings = load_security_settings()
app = FastAPI(title='SpecAutoAI', version='0.1.0')
app.state.security_settings = security_settings
app.add_middleware(ServerAuthMiddleware, settings=security_settings)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(security_settings.allowed_origins),
    allow_origin_regex=(
        r'^https?://(127\.0\.0\.1|localhost)(:\d+)?$'
        if security_settings.mode == 'local'
        else None
    ),
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['Authorization', 'Content-Type', 'Accept'],
)

# 静态资源与 API 同源托管；根路由本身由 catalog router 返回 index.html。
if STATIC_DIR.exists():
    app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')

app.include_router(catalog.router)
app.include_router(auth.router)
app.include_router(datasets.router)
app.include_router(preprocess.router)
app.include_router(runs.router)

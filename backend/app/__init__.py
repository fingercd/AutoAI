"""FastAPI application and ML services for SpecAutoAI.

【模块级说明】
- 职责：这是 `backend.app` 包的标识文件，使 `backend.app.main`、
  `backend.app.routers` 等子模块可以被 Python 正常导入；本文件本身不对外
  暴露任何符号、不产生任何副作用。
- 系统位置：后端全部业务代码都位于本包之下——`main.py` 负责装配 FastAPI
  应用实例；`paths.py` / `version.py` 提供路径常量与契约版本；`http/` 提供
  安全中间件；`routers/` 承载 HTTP 接口；`runs/` 包含训练 Worker 与
  Repository。包外通过 `backend.app.main:app` 进入整个服务。
- 关键设计约束：保持本文件“空实现”，不在包导入时初始化目录、数据库或
  应用实例——这些副作用全部收敛到 `main.py` 的显式装配过程中，避免
  Worker、测试等只 import 子模块的场合被意外触发。
"""

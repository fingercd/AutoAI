"""FastAPI 请求上下文与服务端身份注入边界。

【在系统中的位置】
本包是 HTTP 层的"横切关注点"集合，位于路由层（backend/app/routers/）之下、
具体业务处理之上，负责两件与业务无关但所有请求都依赖的事：

- principal.py：把认证中间件注入到 request.state 的服务端身份（Principal）以
  FastAPI 依赖（Depends）的形式提供给路由函数，保证身份只来自服务端，
  客户端无法在 JSON 请求体里伪造 owner_id / tenant_id。
- security.py：加载并校验进程级安全配置（local / server 模式、Bearer token、
  CORS 来源白名单），并实现 ServerAuthMiddleware 在 ASGI 层做 Bearer 认证。

【关键设计约束】
local 模式面向单机开发，不强制认证；server 模式必须显式配置
AUTOAI_DEPLOYMENT_MODE=server 和足够长的 AUTOAI_API_TOKEN，密钥永远只来自
进程环境变量，不接受请求体、URL、命令行参数或仓库配置文件传入。
"""

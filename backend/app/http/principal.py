"""请求身份注入边界。

身份只由认证中间件和服务端环境配置生成。客户端不能在 JSON 中声明
``owner_id`` 或 ``tenant_id``。

【在系统中的位置】
本模块是 FastAPI 依赖注入（Depends）与服务端认证之间的接缝：
security.ServerAuthMiddleware 在 ASGI 层完成 Bearer 校验后，把服务端
Principal 写入 request.state；本模块的 get_principal 再把它以依赖的形式交给
路由函数。路由因此只认服务端身份，客户端 JSON 中的 owner_id / tenant_id
字段一律不被信任（契约层也不会接收这些字段）。

【协作模块】
- ..runs.contracts.Principal：身份值对象（owner_id / tenant_id）。
- security.SecuritySettings：提供 local 模式的"无身份"回退判断依据。
"""

from fastapi import HTTPException, Request

from ..runs.contracts import Principal


def get_principal(request: Request) -> Principal:
    """返回认证中间件注入的服务端 Principal。

    决策顺序：
    1. 若 request.state 上已有中间件注入的 Principal，直接信任并返回——这是
       server 模式（以及中间件已注入过的 local 模式）的正常路径。
    2. 否则若是 local 模式（无中间件或中间件未注入），返回匿名 Principal()，
       保持本机开发零摩擦。
    3. 其余情况（server 模式但中间件缺失/未注入）返回 401，并带
       WWW-Authenticate: Bearer 响应头，提示客户端用 Bearer token 重试。

    参数：request — 当前 FastAPI 请求对象。
    返回：Principal；local 模式为 owner_id/tenant_id 全 None 的匿名身份。
    异常：HTTPException 401（需要有效的服务器访问令牌）。
    """

    # 优先信任中间件在 ASGI 层注入的服务端身份，这是唯一可信的身份来源
    principal = getattr(request.state, 'principal', None)
    if isinstance(principal, Principal):
        return principal
    # 中间件未注入时，仅 local 模式允许回退为匿名身份（单机开发免认证）
    settings = getattr(request.app.state, 'security_settings', None)
    if settings is not None and settings.mode == 'local':
        return Principal()
    # server 模式下没有注入身份属于异常状态：拒绝访问并按 Bearer 协议提示
    raise HTTPException(
        status_code=401,
        detail={'code': 'authentication_required', 'message': '需要有效的服务器访问令牌'},
        headers={'WWW-Authenticate': 'Bearer'},
    )

"""请求身份注入边界。

身份只由认证中间件和服务端环境配置生成。客户端不能在 JSON 中声明
``owner_id`` 或 ``tenant_id``。
"""

from fastapi import HTTPException, Request

from ..runs.contracts import Principal


def get_principal(request: Request) -> Principal:
    """返回认证中间件注入的服务端 Principal。"""

    principal = getattr(request.state, 'principal', None)
    if isinstance(principal, Principal):
        return principal
    settings = getattr(request.app.state, 'security_settings', None)
    if settings is not None and settings.mode == 'local':
        return Principal()
    raise HTTPException(
        status_code=401,
        detail={'code': 'authentication_required', 'message': '需要有效的服务器访问令牌'},
        headers={'WWW-Authenticate': 'Bearer'},
    )

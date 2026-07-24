"""Authentication discovery and session verification endpoints.

认证发现与会话校验接口。

【在系统中的位置】
本路由提供两个只读端点，属于"前端引导信息"层：浏览器打开页面后先调
/api/auth/config 了解部署模式与 token 存放约定，再用 /api/auth/session
探测当前 token 是否有效。真正的 Bearer 校验发生在
http/security.py 的 ServerAuthMiddleware（ASGI 层），本模块不做任何认证决策。

【关键设计约束】
- 只暴露非机密信息：mode、是否需要认证、token 存放约定；绝不回显 token 本身。
- 浏览器 token 只允许放在当前标签页的 sessionStorage（'token_storage': 'session'
  是与前端的契约字段），不得写入 URL、localStorage、日志或仓库。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ..http.principal import get_principal
from ..runs.contracts import Principal


router = APIRouter()


@router.get('/api/auth/config')
def auth_config(request: Request) -> dict[str, object]:
    """Expose non-secret browser bootstrap information.

    向前端暴露非机密的引导配置：部署模式（local / server）、是否需要认证、
    token 的存放约定（sessionStorage）。security_settings 由应用启动阶段写入
    app.state，这里只读取其中可公开的字段，token 等机密绝不出现在响应里。

    参数：request — 当前请求对象，仅用于访问 app.state.security_settings。
    返回：{'mode', 'auth_required', 'token_storage'} 三个非机密字段。
    """

    # security_settings 在 create_app 启动时由 load_security_settings 生成并挂载
    settings = request.app.state.security_settings
    return {
        'mode': settings.mode,
        'auth_required': settings.auth_required,
        'token_storage': 'session',
    }


@router.get('/api/auth/session')
def auth_session(principal: Principal = Depends(get_principal)) -> dict[str, object]:
    """Validate the current Bearer token without exposing it.

    校验当前 Bearer token 是否有效：能执行到这里，说明 ServerAuthMiddleware
    已在 ASGI 层放行（否则请求早被 401 拦截），且 get_principal 已返回服务端
    身份，因此直接回答 authenticated=True 和 Principal 的 owner_id / tenant_id。
    前端用它做"静默登录检查"，响应不回显 token。

    参数：principal — 由 get_principal 依赖注入的服务端身份。
    返回：{'authenticated': True, 'principal': {'owner_id', 'tenant_id'}}。
    """

    # 依赖注入保证了这里的 principal 只可能来自服务端中间件，不可能是客户端伪造
    return {
        'authenticated': True,
        'principal': {
            'owner_id': principal.owner_id,
            'tenant_id': principal.tenant_id,
        },
    }

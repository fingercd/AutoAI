"""Server deployment authentication and CORS settings.

Local development remains deliberately frictionless. A deployment reachable
from another machine must opt into ``server`` mode and provide a Bearer token
through the process environment. Secrets are never accepted in request bodies,
URLs, command-line flags, or repository configuration files.

【在系统中的位置】
本模块是部署安全的"配置 + 门禁"两层：
- 配置层：SecuritySettings / load_security_settings 从进程环境变量读取并校验
  AUTOAI_DEPLOYMENT_MODE、AUTOAI_API_TOKEN、AUTOAI_ALLOWED_ORIGINS 等；
  任何非法配置都在启动阶段直接 RuntimeError——宁可服务起不来，也不带病运行。
- 门禁层：ServerAuthMiddleware 是纯 ASGI 中间件，对受保护路径强制
  Authorization: Bearer <token>，通过后把服务端 Principal 写入 scope['state']。

【协作模块】
- http/principal.get_principal：读取本中间件注入的 Principal，供路由使用。
- 应用工厂（main.py）：启动时调用 load_security_settings，挂载本中间件，
  并用 cors_origins 的返回值配置 CORSMiddleware。
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from typing import Iterable, Literal

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from ..runs.contracts import Principal


# 部署模式只允许两种取值：local（单机免认证）/ server（强制 Bearer 认证）
DeploymentMode = Literal['local', 'server']


def _split_origins(value: str | None) -> tuple[str, ...]:
    """把逗号分隔的 CORS 来源环境变量解析成干净的来源元组。

    逐段 strip 并去掉尾部 '/'，因为 CORS 的 Origin 头不带路径和尾斜杠，
    归一化后才能与浏览器实际发送的值精确比较；空段被丢弃，未配置时返回
    空元组（由上层决定拒绝跨域或走默认行为）。
    """
    if not value:
        return ()
    return tuple(origin.strip().rstrip('/') for origin in value.split(',') if origin.strip())


@dataclass(frozen=True)
class SecuritySettings:
    """Immutable process-level security settings.

    进程级不可变安全配置（frozen dataclass）：启动时加载一次，运行期不修改，
    避免配置被请求处理路径意外改动。

    字段：
        mode: 'local'（单机免认证）或 'server'（强制 Bearer 认证）。
        api_token: server 模式的 Bearer token；local 模式恒为 None。
        owner_id / tenant_id: server 模式注入 Principal 的身份；local 模式为 None。
        allowed_origins: 显式 CORS 来源白名单（加载阶段已拒绝 '*'）。
    """

    mode: DeploymentMode
    api_token: str | None
    owner_id: str | None
    tenant_id: str | None
    allowed_origins: tuple[str, ...]

    @property
    def auth_required(self) -> bool:
        """是否要求请求认证：只有 server 模式强制认证，local 保持零摩擦。"""
        return self.mode == 'server'

    @property
    def principal(self) -> Principal:
        """生成本进程要注入每个请求的服务端 Principal。

        local 模式返回匿名 Principal（owner/tenant 全 None，不触发作用域隔离）；
        server 模式返回带 owner_id/tenant_id 的身份，仓储层据此做数据隔离。
        """
        if self.mode == 'local':
            return Principal()
        return Principal(owner_id=self.owner_id, tenant_id=self.tenant_id)


def load_security_settings(environ: dict[str, str] | None = None) -> SecuritySettings:
    """Load and validate security settings from the process environment.

    从进程环境变量加载并校验安全配置。设计意图：把"配置不合法"变成启动期
    硬失败（RuntimeError），而不是运行到某次请求才暴露；密钥只来自环境变量，
    不接受请求体 / URL / 命令行参数 / 仓库配置文件。

    参数：environ — 可注入的环境字典（测试用），None 时使用 os.environ。
    返回：校验通过的 SecuritySettings。
    异常：RuntimeError —— 模式取值非法、CORS 使用 '*'、server 模式 token 缺失
        或少于 32 字符、Principal/Tenant ID 去空白后为空。
    """

    values = os.environ if environ is None else environ
    # 读取部署模式并归一化（去空白、转小写），缺省按 local 处理
    raw_mode = values.get('AUTOAI_DEPLOYMENT_MODE', 'local').strip().lower()
    if raw_mode not in {'local', 'server'}:
        # 只接受 local/server 两种取值，其余一律视为配置错误
        raise RuntimeError('AUTOAI_DEPLOYMENT_MODE 必须是 local 或 server')
    mode: DeploymentMode = raw_mode  # type: ignore[assignment]
    # token 与 CORS 来源同样只从环境变量读取，绝不接受其他渠道
    token = values.get('AUTOAI_API_TOKEN')
    origins = _split_origins(values.get('AUTOAI_ALLOWED_ORIGINS'))
    if '*' in origins:
        # 显式拒绝通配来源：server 部署必须列出具体 Origin，防止任意站点跨域调用
        raise RuntimeError('AUTOAI_ALLOWED_ORIGINS 不允许使用 *，请填写明确来源')
    if mode == 'server':
        if token is None or len(token) < 32:
            # server 模式强制长 token（>=32 字符），抬高爆破与误用弱口令的成本
            raise RuntimeError('server 模式必须设置至少 32 个字符的 AUTOAI_API_TOKEN')
        owner_id = values.get('AUTOAI_PRINCIPAL_ID', 'server-admin').strip()
        tenant_id = values.get('AUTOAI_TENANT_ID', 'default').strip()
        if not owner_id or not tenant_id:
            # 身份字段有默认值，但显式配成空白同样拒绝，否则作用域隔离失去意义
            raise RuntimeError('server 模式的 Principal ID 和 Tenant ID 不能为空')
    else:
        # local 模式彻底清空 token/身份字段：认证关闭，残留配置一律不生效
        token = None
        owner_id = None
        tenant_id = None
    return SecuritySettings(
        mode=mode,
        api_token=token,
        owner_id=owner_id,
        tenant_id=tenant_id,
        allowed_origins=origins,
    )


def is_loopback_host(host: str) -> bool:
    """Return whether a bind host is restricted to this machine.

    判断绑定地址是否仅监听本机回环（127.0.0.1 / localhost / ::1）。启动期用它
    提示：绑定回环地址时即使配置不当也不会把服务暴露到局域网。strip('[]')
    是为了兼容 '[::1]' 这种 IPv6 字面量写法。
    """

    normalized = host.strip().lower().strip('[]')
    return normalized in {'127.0.0.1', 'localhost', '::1'}


class ServerAuthMiddleware:
    """Require the configured Bearer token for protected server paths.

    纯 ASGI 中间件（不依赖 FastAPI 路由，因此在路由匹配之前就生效），对受保护
    路径强制 Authorization: Bearer <token>。设计要点：
    - 公开白名单（首页、健康检查、认证配置发现、favicon）与 /static/ 静态资源
      免认证，保证浏览器能先拿到前端入口，再在前端配置 token。
    - local 模式完全旁路认证，但仍注入匿名 Principal，让下游代码路径统一。
    - token 比较用 secrets.compare_digest 做常数时间比较，避免时序侧信道。
    - 认证通过后把服务端 Principal 写进 scope['state']，供 principal.get_principal
      读取——身份只来自这里，客户端无法在请求体中伪造。
    """

    # 免认证的精确路径：入口页、健康检查、前端引导配置、favicon
    _PUBLIC_PATHS = frozenset({'/', '/health', '/api/auth/config', '/favicon.ico'})
    # 受保护前缀：所有 /api/ 业务接口与 OpenAPI 文档（文档会暴露接口结构，
    # server 模式下同样需要保护）
    _PROTECTED_PREFIXES = ('/api/', '/docs', '/redoc', '/openapi.json')

    def __init__(self, app: ASGIApp, *, settings: SecuritySettings) -> None:
        """包裹下游 ASGI 应用并持有进程级安全配置。"""
        self.app = app
        self.settings = settings

    @classmethod
    def _requires_auth(cls, path: str) -> bool:
        """判断路径是否需要认证：白名单与 /static/ 直接放行，命中受保护前缀则要求 token。"""
        if path in cls._PUBLIC_PATHS or path.startswith('/static/'):
            return False
        return any(path == prefix or path.startswith(prefix) for prefix in cls._PROTECTED_PREFIXES)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI 入口：按模式与路径决定放行、校验或直接 401。

        参数：scope/receive/send — 标准 ASGI 三元组；principal 注入在
            scope['state'] 上完成，随请求生命周期传递。
        """
        # 非 HTTP/WebSocket 的 ASGI 事件（如 lifespan）不承载请求，直接透传
        if scope['type'] not in {'http', 'websocket'}:
            await self.app(scope, receive, send)
            return

        # local 模式：不校验 token，但注入匿名 Principal，保证下游行为一致
        if self.settings.mode == 'local':
            scope.setdefault('state', {})['principal'] = self.settings.principal
            await self.app(scope, receive, send)
            return

        path = str(scope.get('path') or '')
        # 仅对受保护路径做 Bearer 校验；公开路径与静态资源跳过
        if self._requires_auth(path):
            headers = Headers(scope=scope)
            # 解析 "Authorization: Bearer <token>"；partition 兼容缺失/畸形头
            scheme, _, credential = headers.get('authorization', '').partition(' ')
            expected = self.settings.api_token or ''
            # 常数时间比较防时序攻击；scheme 大小写不敏感，credential 必须非空
            authenticated = (
                scheme.lower() == 'bearer'
                and bool(credential)
                and secrets.compare_digest(credential, expected)
            )
            if not authenticated:
                # 校验失败：统一 401 + WWW-Authenticate，不区分"没带"和"带错"，
                # 避免向探测者泄露任何关于 token 的信息
                response = JSONResponse(
                    status_code=401,
                    content={
                        'error': {
                            'code': 'authentication_required',
                            'message': '需要有效的服务器访问令牌',
                        }
                    },
                    headers={'WWW-Authenticate': 'Bearer'},
                )
                await response(scope, receive, send)
                return

        # 通过（或无需认证）的请求统一注入服务端 Principal，身份由此进入请求生命周期
        scope.setdefault('state', {})['principal'] = self.settings.principal
        await self.app(scope, receive, send)


def cors_origins(settings: SecuritySettings) -> Iterable[str]:
    """Return explicit origins accepted by CORSMiddleware.

    把配置中的显式来源白名单交给 CORSMiddleware。加载阶段已拒绝 '*'，
    所以这里返回的一定是具体 Origin 列表（可能为空，表示不允许任何跨域来源）。
    """

    return settings.allowed_origins

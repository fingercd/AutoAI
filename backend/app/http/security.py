"""Server deployment authentication and CORS settings.

Local development remains deliberately frictionless. A deployment reachable
from another machine must opt into ``server`` mode and provide a Bearer token
through the process environment. Secrets are never accepted in request bodies,
URLs, command-line flags, or repository configuration files.
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


DeploymentMode = Literal['local', 'server']


def _split_origins(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(origin.strip().rstrip('/') for origin in value.split(',') if origin.strip())


@dataclass(frozen=True)
class SecuritySettings:
    """Immutable process-level security settings."""

    mode: DeploymentMode
    api_token: str | None
    owner_id: str | None
    tenant_id: str | None
    allowed_origins: tuple[str, ...]

    @property
    def auth_required(self) -> bool:
        return self.mode == 'server'

    @property
    def principal(self) -> Principal:
        if self.mode == 'local':
            return Principal()
        return Principal(owner_id=self.owner_id, tenant_id=self.tenant_id)


def load_security_settings(environ: dict[str, str] | None = None) -> SecuritySettings:
    """Load and validate security settings from the process environment."""

    values = os.environ if environ is None else environ
    raw_mode = values.get('AUTOAI_DEPLOYMENT_MODE', 'local').strip().lower()
    if raw_mode not in {'local', 'server'}:
        raise RuntimeError('AUTOAI_DEPLOYMENT_MODE 必须是 local 或 server')
    mode: DeploymentMode = raw_mode  # type: ignore[assignment]
    token = values.get('AUTOAI_API_TOKEN')
    origins = _split_origins(values.get('AUTOAI_ALLOWED_ORIGINS'))
    if '*' in origins:
        raise RuntimeError('AUTOAI_ALLOWED_ORIGINS 不允许使用 *，请填写明确来源')
    if mode == 'server':
        if token is None or len(token) < 32:
            raise RuntimeError('server 模式必须设置至少 32 个字符的 AUTOAI_API_TOKEN')
        owner_id = values.get('AUTOAI_PRINCIPAL_ID', 'server-admin').strip()
        tenant_id = values.get('AUTOAI_TENANT_ID', 'default').strip()
        if not owner_id or not tenant_id:
            raise RuntimeError('server 模式的 Principal ID 和 Tenant ID 不能为空')
    else:
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
    """Return whether a bind host is restricted to this machine."""

    normalized = host.strip().lower().strip('[]')
    return normalized in {'127.0.0.1', 'localhost', '::1'}


class ServerAuthMiddleware:
    """Require the configured Bearer token for protected server paths."""

    _PUBLIC_PATHS = frozenset({'/', '/health', '/api/auth/config', '/favicon.ico'})
    _PROTECTED_PREFIXES = ('/api/', '/docs', '/redoc', '/openapi.json')

    def __init__(self, app: ASGIApp, *, settings: SecuritySettings) -> None:
        self.app = app
        self.settings = settings

    @classmethod
    def _requires_auth(cls, path: str) -> bool:
        if path in cls._PUBLIC_PATHS or path.startswith('/static/'):
            return False
        return any(path == prefix or path.startswith(prefix) for prefix in cls._PROTECTED_PREFIXES)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] not in {'http', 'websocket'}:
            await self.app(scope, receive, send)
            return

        if self.settings.mode == 'local':
            scope.setdefault('state', {})['principal'] = self.settings.principal
            await self.app(scope, receive, send)
            return

        path = str(scope.get('path') or '')
        if self._requires_auth(path):
            headers = Headers(scope=scope)
            scheme, _, credential = headers.get('authorization', '').partition(' ')
            expected = self.settings.api_token or ''
            authenticated = (
                scheme.lower() == 'bearer'
                and bool(credential)
                and secrets.compare_digest(credential, expected)
            )
            if not authenticated:
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

        scope.setdefault('state', {})['principal'] = self.settings.principal
        await self.app(scope, receive, send)


def cors_origins(settings: SecuritySettings) -> Iterable[str]:
    """Return explicit origins accepted by CORSMiddleware."""

    return settings.allowed_origins

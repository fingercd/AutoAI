"""薄 HTTP Agent Tool Client；不复制服务端权限、预算或选模策略。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


CONTRACT_VERSION = 'agent-session-v1'


class AgentClientError(RuntimeError):
    pass


class AgentConnectionError(AgentClientError):
    pass


class AgentTimeoutError(AgentClientError):
    pass


class AgentContractError(AgentClientError):
    pass


class AgentHTTPError(AgentClientError):
    def __init__(self, message: str, *, status_code: int, code: str | None,
                 retryable: bool, allowed_actions: list[str]) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.retryable = retryable
        self.allowed_actions = list(allowed_actions)


class Transport(Protocol):
    def request(self, method: str, url: str, *, headers: dict[str, str],
                json: dict[str, Any] | None, timeout: float) -> Any: ...


class _HttpxTransport:
    def request(self, method: str, url: str, *, headers: dict[str, str],
                json: dict[str, Any] | None, timeout: float) -> Any:
        import httpx
        return httpx.request(method, url, headers=headers, json=json, timeout=timeout)


@dataclass(repr=False)
class AutoAIClient:
    base_url: str
    token: str | None = None
    timeout: float = 10.0
    max_retries: int = 2
    transport: Transport | None = None

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip('/')
        if not self.base_url.startswith(('http://', 'https://')):
            raise ValueError('base_url 必须是 HTTP(S) URL')
        if self.transport is None:
            self.transport = _HttpxTransport()

    def __repr__(self) -> str:
        return f'AutoAIClient(base_url={self.base_url!r}, token=<redacted>)'

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None,
                 *, idempotent: bool = False) -> dict[str, Any]:
        headers = {'Accept': 'application/json'}
        if body is not None:
            headers['Content-Type'] = 'application/json'
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        attempts = 1 + (self.max_retries if method == 'GET' or idempotent else 0)
        for index in range(attempts):
            try:
                assert self.transport is not None
                response = self.transport.request(
                    method, self.base_url + path, headers=headers, json=body, timeout=self.timeout
                )
                break
            except TimeoutError as exc:
                if index + 1 == attempts:
                    raise AgentTimeoutError('SpecAutoAI 请求超时') from exc
            except (ConnectionError, OSError) as exc:
                if index + 1 == attempts:
                    raise AgentConnectionError('无法连接 SpecAutoAI 服务') from exc
            except Exception as exc:
                # httpx 可选依赖的异常不进入公共 repr，也不回显请求头。
                name = type(exc).__name__.lower()
                if 'timeout' in name and index + 1 == attempts:
                    raise AgentTimeoutError('SpecAutoAI 请求超时') from exc
                if index + 1 == attempts:
                    raise AgentConnectionError('无法连接 SpecAutoAI 服务') from exc
        try:
            payload = response.json()
        except Exception as exc:
            raise AgentContractError('服务端返回了无效 JSON') from exc
        if not 200 <= int(response.status_code) < 300:
            detail = payload.get('detail') if isinstance(payload, dict) else None
            detail = detail if isinstance(detail, dict) else {}
            raise AgentHTTPError(
                str(detail.get('message') or f'HTTP {response.status_code}'),
                status_code=int(response.status_code),
                code=str(detail['code']) if detail.get('code') else None,
                retryable=bool(detail.get('retryable')),
                allowed_actions=[str(item) for item in detail.get('allowed_actions', [])],
            )
        if not isinstance(payload, dict) or payload.get('contract_version') != CONTRACT_VERSION:
            raise AgentContractError('Agent API contract version 不匹配')
        return payload

    def inspect_ml_capabilities(self) -> dict[str, Any]:
        return self._request('GET', '/api/agent/health')

    def start_ml_session(self, *, dataset_id: str, selection_metric: str,
                         allowed_models: list[str], max_runs: int, seed: int = 42,
                         evaluation: dict[str, Any] | None = None,
                         modules: list[str] | None = None,
                         context_policy: dict[str, Any] | None = None,
                         client_request_id: str | None = None) -> dict[str, Any]:
        body = {
            'dataset_id': dataset_id, 'selection_metric': selection_metric,
            'allowed_models': allowed_models, 'max_runs': max_runs, 'seed': seed,
            'evaluation': evaluation or {'split_mode': 'stratified_holdout', 'split_train': 8,
                                         'split_valid': 1, 'split_test': 1},
            'modules': modules or [],
            'context_policy': context_policy or {'source_role': 'development', 'case_write': False},
        }
        if client_request_id: body['client_request_id'] = client_request_id
        return self._request('POST', '/api/agent/sessions', body, idempotent=bool(client_request_id))

    def inspect_ml_session(self, session_id: str) -> dict[str, Any]:
        return self._request('GET', f'/api/agent/sessions/{session_id}')

    def submit_ml_experiment(self, session_id: str, *, model_type: str,
                             normalization: str = 'zscore', class_balance: str = 'none',
                             parent_run_id: str | None = None, rationale: str | None = None,
                             client_request_id: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            'model_type': model_type, 'normalization': normalization,
            'class_balance': class_balance,
        }
        for key, value in (('parent_run_id', parent_run_id), ('rationale', rationale),
                           ('client_request_id', client_request_id)):
            if value is not None: body[key] = value
        return self._request(
            'POST', f'/api/agent/sessions/{session_id}/experiments', body,
            idempotent=bool(client_request_id),
        )

    def observe_ml_experiment(self, session_id: str, run_id: str) -> dict[str, Any]:
        return self._request('GET', f'/api/agent/sessions/{session_id}/experiments/{run_id}/feedback')

    def finalize_ml_session(self, session_id: str, selected_run_id: str) -> dict[str, Any]:
        # 服务端 finalize 对相同 selected_run_id 幂等，因此允许有界重试。
        return self._request(
            'POST', f'/api/agent/sessions/{session_id}/finalize',
            {'selected_run_id': selected_run_id}, idempotent=True,
        )

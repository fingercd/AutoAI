"""Registry-driven, path-safe probes for local OpenAI-compatible runtimes."""

from __future__ import annotations

import json
import re
import socket
import time
from pathlib import Path
from typing import Any, Callable, Literal
from urllib import error, request
from urllib.parse import urlparse

from agent_poc.state import ModelConfig, load_model_registry
from ..paths import PROJECT_ROOT


RUNTIME_HEALTH_CONTRACT = 'agent-runtime-health-v1'
DEFAULT_MODELS_CONFIG = PROJECT_ROOT / 'agent_poc' / 'config' / 'models.toml'
_PUBLIC_IDENTIFIER = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$')


class AgentRegistryUnavailable(RuntimeError):
    pass


def _latency_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _load_registry(path: Path) -> dict[str, ModelConfig]:
    try:
        loaded = load_model_registry(path)
    except Exception as exc:
        raise AgentRegistryUnavailable('agent registry unavailable') from exc
    if not loaded:
        raise AgentRegistryUnavailable('agent registry unavailable')
    registry: dict[str, ModelConfig] = {}
    for key_text, model in loaded.items():
        parsed = urlparse(model.base_url)
        if (
            not _PUBLIC_IDENTIFIER.fullmatch(key_text)
            or not _PUBLIC_IDENTIFIER.fullmatch(model.served_model_name)
            or parsed.scheme not in {'http', 'https'}
            or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise AgentRegistryUnavailable('agent registry unavailable')
        registry[key_text] = model
    return registry


def _request_json(
    url: str,
    *,
    method: str = 'GET',
    payload: dict[str, Any] | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode('utf-8')
    req = request.Request(
        url,
        data=body,
        method=method,
        headers={'Content-Type': 'application/json'},
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:
        decoded = json.loads(response.read().decode('utf-8'))
    if not isinstance(decoded, dict):
        raise ValueError('non_object_response')
    return decoded


def _error_code(exc: Exception) -> str:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return 'timeout'
    if isinstance(exc, error.HTTPError):
        if exc.code in {401, 403}:
            return 'authentication_failed'
        if exc.code == 429:
            return 'rate_limited'
        return 'upstream_http_error'
    if isinstance(exc, (error.URLError, ConnectionError)):
        return 'unreachable'
    if isinstance(exc, (ValueError, KeyError, IndexError, TypeError)):
        return 'invalid_probe_response'
    return 'probe_failed'


def _advertised_ids(payload: dict[str, Any]) -> list[str]:
    items = payload.get('data')
    if not isinstance(items, list):
        raise ValueError('invalid_models_response')
    result = []
    for item in items:
        if isinstance(item, dict) and isinstance(item.get('id'), str):
            result.append(item['id'])
    return result


def _probe_one(
    model_key: str,
    config: ModelConfig,
    *,
    probe: Literal['models', 'inference'],
    requester: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    served_name = config.served_model_name
    result: dict[str, Any] = {
        'model_key': model_key,
        'served_model_name': served_name,
        'models_api': {
            'ok': False,
            'advertised': False,
            'advertised_count': 0,
            'latency_ms': None,
            'error_code': None,
        },
        'inference': {
            'requested': probe == 'inference',
            'attempted': False,
            'ok': None if probe == 'models' else False,
            'latency_ms': None,
            'error_code': None,
        },
    }
    started = time.perf_counter()
    try:
        payload = requester(f"{config.base_url.rstrip('/')}/models")
        advertised = _advertised_ids(payload)
    except Exception as exc:
        result['models_api']['latency_ms'] = _latency_ms(started)
        result['models_api']['error_code'] = _error_code(exc)
        if probe == 'inference':
            result['inference']['error_code'] = 'models_probe_failed'
        return result

    result['models_api'].update({
        'ok': True,
        'advertised': served_name in advertised,
        'advertised_count': len(advertised),
        'latency_ms': _latency_ms(started),
    })
    if served_name not in advertised:
        result['models_api']['error_code'] = 'served_model_not_advertised'
        if probe == 'inference':
            result['inference']['error_code'] = 'models_probe_failed'
        return result

    if probe == 'models':
        return result

    started = time.perf_counter()
    result['inference']['attempted'] = True
    try:
        payload = requester(
            f"{config.base_url.rstrip('/')}/chat/completions",
            method='POST',
            payload={
                'model': served_name,
                'messages': [
                    {
                        'role': 'user',
                        'content': 'Return only JSON with ok equal to true.',
                    }
                ],
                'temperature': 0.0,
                'max_tokens': 32,
                'structured_outputs': {
                    'json': {
                        'type': 'object',
                        'properties': {'ok': {'const': True}},
                        'required': ['ok'],
                        'additionalProperties': False,
                    }
                },
                'chat_template_kwargs': {'enable_thinking': False},
            },
        )
        content = payload['choices'][0]['message']['content']
        parsed = json.loads(content)
        if parsed != {'ok': True}:
            raise ValueError('invalid_probe_response')
    except Exception as exc:
        result['inference']['latency_ms'] = _latency_ms(started)
        result['inference']['error_code'] = _error_code(exc)
        return result

    result['inference'].update({
        'ok': True,
        'latency_ms': _latency_ms(started),
    })
    return result


def probe_runtime_health(
    *,
    probe: Literal['models', 'inference'] = 'models',
    model_key: str | None = None,
    config_path: Path = DEFAULT_MODELS_CONFIG,
    requester: Callable[..., dict[str, Any]] = _request_json,
) -> dict[str, Any]:
    registry = _load_registry(config_path)
    if probe == 'inference' and not model_key:
        raise ValueError('inference probe requires model_key')
    if model_key and model_key not in registry:
        raise KeyError(model_key)
    selected = {model_key: registry[model_key]} if model_key else registry
    models = [
        _probe_one(key, selected[key], probe=probe, requester=requester)
        for key in sorted(selected)
    ]
    ready = all(
        item['models_api']['ok']
        and item['models_api']['advertised']
        and (probe == 'models' or item['inference']['ok'])
        for item in models
    )
    return {
        'contract_version': RUNTIME_HEALTH_CONTRACT,
        'status': 'ready' if ready else 'degraded',
        'probe': probe,
        'agent_api': {'ready': True},
        'models': models,
    }

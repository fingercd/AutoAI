"""Aggregate the protected backend/Agent runtime health contract safely."""

from __future__ import annotations

from typing import Any, Iterable

from .clients.autoai_client import AutoAIClient
from .state import ModelConfig


HEALTH_SCHEMA_VERSION = 'agent-health-v1'
EXPECTED_AGENT_CONTRACT = 'agent-session-v1'
EXPECTED_RUNTIME_CONTRACT = 'agent-runtime-health-v1'


def _error_code(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    if 'timeout' in name:
        return 'timeout'
    if 'authentication' in name or 'permission' in name:
        return 'authentication_failed'
    if 'connect' in name or 'connection' in name:
        return 'unreachable'
    if 'http' in name or 'status' in name:
        return 'upstream_http_error'
    return 'probe_failed'


def probe_agent_stack(
    autoai: AutoAIClient,
    model_configs: Iterable[ModelConfig],
    *,
    run_inference: bool = False,
) -> dict[str, Any]:
    """Read global worker health plus the protected runtime probe.

    The expected registry keys come from the same config used to launch this
    client. A backend response cannot claim readiness if its contracts or
    model set drift from those expectations.
    """
    configs = list(model_configs)
    expected_keys = [item.key for item in configs]
    expected_unique = len(expected_keys) == len(set(expected_keys))
    requested_key = expected_keys[0] if len(expected_keys) == 1 else None

    try:
        backend = autoai.health()
    except Exception as exc:
        backend = {'status': 'unreachable', 'error_code': _error_code(exc)}
    try:
        agent_api = autoai.agent_health(
            probe='inference' if run_inference else 'models',
            model_key=requested_key,
        )
    except Exception as exc:
        agent_api = {
            'status': 'unreachable',
            'error_code': _error_code(exc),
            'models': [],
        }

    model_reports = agent_api.get('models')
    if not isinstance(model_reports, list):
        model_reports = []
    returned_keys = [
        item.get('model_key')
        for item in model_reports
        if isinstance(item, dict) and isinstance(item.get('model_key'), str)
    ]
    returned_unique = len(returned_keys) == len(set(returned_keys))
    model_keys_match = (
        expected_unique
        and returned_unique
        and set(expected_keys) == set(returned_keys)
        and len(expected_keys) == len(returned_keys)
    )
    runtime_contract_ok = (
        agent_api.get('contract_version') == EXPECTED_RUNTIME_CONTRACT
    )
    agent_contract_ok = (
        agent_api.get('agent_contract') == EXPECTED_AGENT_CONTRACT
    )

    worker = backend.get('worker') if isinstance(backend, dict) else None
    worker_ready = isinstance(worker, dict) and (
        worker.get('available') is True and worker.get('compatible') is True
    )
    models_ready = bool(model_reports) and all(
        isinstance(item, dict)
        and item.get('models_api', {}).get('ok') is True
        and item.get('models_api', {}).get('advertised') is True
        and (
            not run_inference
            or item.get('inference', {}).get('ok') is True
        )
        for item in model_reports
    )
    ready = (
        backend.get('status') == 'ok'
        and agent_api.get('status') == 'ready'
        and worker_ready
        and models_ready
        and model_keys_match
        and runtime_contract_ok
        and agent_contract_ok
    )
    return {
        'schema_version': HEALTH_SCHEMA_VERSION,
        'status': 'ready' if ready else 'degraded',
        'backend': {
            'status': backend.get('status'),
            'deployment_mode': backend.get('deployment_mode'),
            'worker_ready': worker_ready,
        },
        'contract_checks': {
            'runtime_contract': runtime_contract_ok,
            'agent_contract': agent_contract_ok,
            'model_keys_match': model_keys_match,
            'model_keys_unique': expected_unique and returned_unique,
        },
        'agent_api': agent_api,
        'models': model_reports,
    }

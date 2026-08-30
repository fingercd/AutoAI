from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.agent import runtime_health
from backend.app.http.security import ServerAuthMiddleware, load_security_settings
from backend.app.main import app
from backend.app.routers import agent as agent_router


@pytest.fixture
def client():
    return TestClient(app)


def _requester(url, *, method='GET', payload=None, timeout_seconds=10.0):
    if url.endswith('/models'):
        if ':8101/' in url:
            return {'data': [{'id': 'qwen35_9b'}]}
        if ':8102/' in url:
            return {'data': [{'id': '/private/model/path'}]}
        raise ConnectionError('Bearer secret /private/path https://internal')
    assert method == 'POST'
    assert payload['temperature'] == 0.0
    assert payload['max_tokens'] == 32
    return {'choices': [{'message': {'content': '{"ok": true}'}}]}


def test_models_probe_is_degraded_and_never_leaks_upstream_ids(monkeypatch, client):
    original_probe = runtime_health.probe_runtime_health
    monkeypatch.setattr(
        agent_router,
        'probe_runtime_health',
        lambda **kwargs: original_probe(
            **kwargs, requester=_requester
        ),
    )
    response = client.get('/api/agent/health')
    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'degraded'
    assert payload['frontend_required'] is False
    assert payload['database_ready'] is True
    flat = json.dumps(payload, ensure_ascii=False).lower()
    for forbidden in ('/private/', 'bearer ', 'https://', 'base_url', 'model_path'):
        assert forbidden not in flat


def test_single_model_inference_probe(monkeypatch, client):
    original_probe = runtime_health.probe_runtime_health
    monkeypatch.setattr(
        agent_router,
        'probe_runtime_health',
        lambda **kwargs: original_probe(
            **kwargs, requester=_requester
        ),
    )
    response = client.get(
        '/api/agent/health',
        params={'probe': 'inference', 'model_key': 'qwen35_9b'},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'ready'
    assert payload['models'][0]['inference']['ok'] is True


def test_inference_requires_known_single_model(monkeypatch, client):
    original_probe = runtime_health.probe_runtime_health
    monkeypatch.setattr(
        agent_router,
        'probe_runtime_health',
        lambda **kwargs: original_probe(
            **kwargs, requester=_requester
        ),
    )
    missing = client.get('/api/agent/health', params={'probe': 'inference'})
    assert missing.status_code == 422
    unknown = client.get(
        '/api/agent/health',
        params={'probe': 'models', 'model_key': 'not-configured'},
    )
    assert unknown.status_code == 404


def test_openapi_contains_agent_health(client):
    schema = client.get('/openapi.json').json()
    assert '/api/agent/health' in schema['paths']


def test_registry_rejects_non_loopback_or_path_like_identifiers(tmp_path):
    config = tmp_path / 'models.toml'
    config.write_text(
        '[models.bad]\n'
        'base_url = "https://example.invalid/v1"\n'
        'served_model_name = "/private/model"\n',
        encoding='utf-8',
    )
    with pytest.raises(runtime_health.AgentRegistryUnavailable):
        runtime_health.probe_runtime_health(config_path=config)


def test_backend_failure_uses_fixed_path_safe_error(monkeypatch, client):
    def fail_backend():
        raise RuntimeError('Bearer secret /private/database C:\\secret\\agent.db')

    monkeypatch.setattr(agent_router, '_agent_service', fail_backend)
    response = client.get('/api/agent/health')
    assert response.status_code == 503
    assert response.json() == {'detail': {'code': 'agent_backend_unavailable'}}
    flat = response.text.lower()
    for forbidden in ('bearer ', '/private/', 'c:\\secret'):
        assert forbidden not in flat


def test_malformed_models_response_is_not_reported_as_model_mismatch():
    report = runtime_health.probe_runtime_health(
        requester=lambda *args, **kwargs: {'data': 'not-a-list'}
    )
    assert report['status'] == 'degraded'
    assert all(
        item['models_api']['error_code'] == 'invalid_probe_response'
        and item['models_api']['ok'] is False
        for item in report['models']
    )


def test_inference_transport_error_keeps_stable_code_and_attempt_state():
    def requester(url, **kwargs):
        if url.endswith('/models'):
            return {'data': [{'id': 'qwen35_9b'}]}
        raise TimeoutError('secret upstream details')

    report = runtime_health.probe_runtime_health(
        probe='inference',
        model_key='qwen35_9b',
        requester=requester,
    )
    inference = report['models'][0]['inference']
    assert inference['requested'] is True
    assert inference['attempted'] is True
    assert inference['error_code'] == 'timeout'


def test_server_mode_requires_bearer_for_agent_health(monkeypatch):
    settings = load_security_settings({
        'AUTOAI_DEPLOYMENT_MODE': 'server',
        'AUTOAI_API_TOKEN': 'a' * 32,
        'AUTOAI_PRINCIPAL_ID': 'health-test',
        'AUTOAI_TENANT_ID': 'default',
    })
    secured = FastAPI()
    secured.state.security_settings = settings
    secured.add_middleware(ServerAuthMiddleware, settings=settings)
    secured.include_router(agent_router.router)
    monkeypatch.setattr(agent_router, '_agent_service', lambda: object())
    monkeypatch.setattr(
        agent_router,
        'probe_runtime_health',
        lambda **_: {
            'contract_version': 'agent-runtime-health-v1',
            'status': 'ready',
            'probe': 'models',
            'agent_api': {'ready': True},
            'models': [],
        },
    )
    secured_client = TestClient(secured)
    assert secured_client.get('/api/agent/health').status_code == 401
    response = secured_client.get(
        '/api/agent/health',
        headers={'Authorization': f"Bearer {'a' * 32}"},
    )
    assert response.status_code == 200

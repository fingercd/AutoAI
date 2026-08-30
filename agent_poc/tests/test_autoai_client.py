from __future__ import annotations

import httpx
import pytest

from agent_poc.clients.autoai_client import AutoAIClient


def test_get_health_retries_but_experiment_post_does_not():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == '/health':
            count = sum(path == '/health' for _, path in calls)
            if count < 3:
                return httpx.Response(503, request=request)
            return httpx.Response(200, json={'status': 'ok'}, request=request)
        if request.url.path == '/api/agent/health':
            assert request.url.params['probe'] == 'models'
            return httpx.Response(200, json={'status': 'ok'}, request=request)
        return httpx.Response(503, json={'detail': 'busy'}, request=request)

    http = httpx.Client(
        base_url='http://autoai',
        transport=httpx.MockTransport(handler),
    )
    client = AutoAIClient('http://autoai', http_client=http)
    assert client.health()['status'] == 'ok'
    assert calls.count(('GET', '/health')) == 3
    assert client.agent_health()['status'] == 'ok'
    assert calls.count(('GET', '/api/agent/health')) == 1
    with pytest.raises(httpx.HTTPStatusError):
        client.create_experiment('session-1', {'model_type': 'svm'})
    assert calls.count(('POST', '/api/agent/sessions/session-1/experiments')) == 1
    client.close()

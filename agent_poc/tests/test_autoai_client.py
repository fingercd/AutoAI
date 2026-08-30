from __future__ import annotations

import httpx
import pytest

from agent_poc.clients.autoai_client import AutoAIClient
from agent_poc.budget import BudgetController, BudgetExceeded
from agent_poc.state import AgentBudgetConfig


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


def test_real_http_attempts_and_retries_are_hard_debited_before_call():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(503, request=request)

    limits = AgentBudgetConfig(
        max_model_fits=10,
        max_llm_calls=2,
        max_api_calls=10,
        max_wall_clock_seconds=30,
        max_retry_attempts=0,
    )
    controller = BudgetController(limits)
    http = httpx.Client(
        base_url='http://autoai',
        transport=httpx.MockTransport(handler),
    )
    client = AutoAIClient(
        'http://autoai',
        http_client=http,
        budget_controller=controller,
    )
    with pytest.raises(BudgetExceeded) as result:
        client.health()
    assert result.value.dimension == 'max_retry_attempts'
    assert calls == ['/health']
    assert controller.safe_usage()['api_call_count'] == 1
    assert controller.safe_usage()['retry_attempt_count'] == 0
    client.close()

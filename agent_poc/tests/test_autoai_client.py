import pytest

from agent_poc.clients.autoai_client import (
    AgentConnectionError, AgentContractError, AgentHTTPError, AutoAIClient,
)


class Response:
    def __init__(self, status, body): self.status_code, self.body = status, body
    def json(self): return self.body


class FakeTransport:
    def __init__(self, responses): self.responses, self.calls = list(responses), []
    def request(self, method, url, *, headers, json, timeout):
        self.calls.append((method, url, headers, json, timeout))
        item = self.responses.pop(0)
        if isinstance(item, Exception): raise item
        return item


def ok(**extra): return Response(200, {'contract_version': 'agent-session-v1', **extra})


def test_all_six_tools_use_finite_endpoints():
    transport = FakeTransport([ok()] * 6)
    client = AutoAIClient('http://localhost:8000', transport=transport)
    client.inspect_ml_capabilities()
    client.start_ml_session(dataset_id='ds', selection_metric='macro_f1',
                            allowed_models=['logistic_regression'], max_runs=1)
    client.inspect_ml_session('s')
    client.submit_ml_experiment('s', model_type='logistic_regression')
    client.observe_ml_experiment('s', 'r')
    client.finalize_ml_session('s', 'r')
    assert len(transport.calls) == 6


def test_token_is_redacted_from_repr_and_connection_error():
    token = 'highly-secret-token'
    transport = FakeTransport([ConnectionError(token)])
    client = AutoAIClient('http://localhost', token=token, transport=transport, max_retries=0)
    with pytest.raises(AgentConnectionError) as info: client.inspect_ml_capabilities()
    assert token not in repr(client)
    assert token not in str(info.value)


def test_post_without_idempotency_key_is_not_retried():
    transport = FakeTransport([ConnectionError('down'), ok()])
    client = AutoAIClient('http://localhost', transport=transport)
    with pytest.raises(AgentConnectionError):
        client.submit_ml_experiment('s', model_type='logistic_regression')
    assert len(transport.calls) == 1


def test_post_with_idempotency_key_retries_to_fixed_limit():
    transport = FakeTransport([ConnectionError('down'), ok(run_id='r')])
    client = AutoAIClient('http://localhost', transport=transport, max_retries=2)
    result = client.submit_ml_experiment(
        's', model_type='logistic_regression', client_request_id='stable-1'
    )
    assert result['run_id'] == 'r'
    assert len(transport.calls) == 2


def test_contract_mismatch_fails_closed():
    client = AutoAIClient('http://localhost', transport=FakeTransport([
        Response(200, {'contract_version': 'future'})
    ]))
    with pytest.raises(AgentContractError): client.inspect_ml_capabilities()


@pytest.mark.parametrize('status', [409, 422, 503])
def test_domain_error_metadata_is_preserved(status):
    client = AutoAIClient('http://localhost', transport=FakeTransport([Response(status, {
        'detail': {'code': 'stable', 'message': 'safe', 'retryable': status == 503,
                   'allowed_actions': ['inspect_ml_session']}
    })]))
    with pytest.raises(AgentHTTPError) as info: client.inspect_ml_capabilities()
    assert info.value.code == 'stable'
    assert info.value.allowed_actions == ['inspect_ml_session']

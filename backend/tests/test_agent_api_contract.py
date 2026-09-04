from fastapi.testclient import TestClient

from backend.app.agent.contracts import CreateAgentSessionRequest
from backend.app.main import app


def test_openapi_agent_request_is_closed_and_integer_fields_are_strict():
    schema = CreateAgentSessionRequest.model_json_schema()
    assert schema['additionalProperties'] is False
    assert schema['properties']['max_runs']['type'] == 'integer'
    assert schema['properties']['seed']['type'] == 'integer'


def test_agent_health_reports_planned_modules_as_unavailable():
    response = TestClient(app).get('/api/agent/health')
    assert response.status_code == 200
    body = response.json()
    assert body['contract_version'] == 'agent-session-v1'
    assert all(not module['available'] for module in body['modules'].values())
    assert all(module['status'] == 'unavailable' for module in body['modules'].values())


def test_agent_request_models_reject_identity_paths_and_test_fields():
    base = {
        'dataset_id': 'ds-x', 'selection_metric': 'macro_f1',
        'allowed_models': ['logistic_regression'], 'max_runs': 1,
    }
    for key in ('owner_id', 'tenant_id', 'data_path', 'test_dataset_id', 'test_data_path'):
        payload = dict(base); payload[key] = 'forbidden'
        response = TestClient(app).post('/api/agent/sessions', json=payload)
        assert response.status_code == 422


def test_unavailable_module_uses_stable_error_envelope():
    response = TestClient(app).post('/api/agent/sessions', json={
        'dataset_id': 'not-resolved-because-module-fails-first',
        'selection_metric': 'macro_f1', 'allowed_models': ['logistic_regression'],
        'max_runs': 1, 'modules': ['evidence_card'],
    })
    assert response.status_code == 422
    assert response.json()['detail'] == {
        'code': 'agent_module_unavailable',
        'message': '请求的模块当前不可用：evidence_card',
        'retryable': False,
        'allowed_actions': [],
    }

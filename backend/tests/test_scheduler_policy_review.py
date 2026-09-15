"""Strict scheduler bounds and immutable compatibility of pre-fix snapshots."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from backend.tests.test_agent_model_sessions import api
from backend.app.model_config import resolve_model_params


def old_snapshot():
    return json.loads((Path(__file__).parent/'fixtures/agent_capability_v1.json').read_text())


@pytest.mark.parametrize('factor', [0, 1, 1.01, True, None, '0.5'])
def test_invalid_scheduler_rejected_before_session(api, factor):
    client, storage, dataset = api
    response = client.post('/api/agent/v2/sessions', json=dict(dataset_id=dataset,
        selection_metric='macro_f1', allowed_models=['cnn1d'], max_runs=1,
        model_configs={'cnn1d':{'scheduler_factor':factor}}))
    assert response.status_code == 422
    from backend.app.routers.agent import _agent_service
    assert _agent_service('agent-session-v2').runs.list() == []


def test_real_scheduler_and_legacy_training_validation():
    import torch
    from backend.app.contracts import TrainingSpec, TrainingConfigValidationError
    for factor in (0, 1):
        with pytest.raises(ValueError): resolve_model_params('cnn1d', {'scheduler_factor':factor})
        with pytest.raises(TrainingConfigValidationError):
            TrainingSpec.from_legacy({'model_type':'cnn1d','scheduler_factor':factor}).validated(has_external_test=False)
    params = resolve_model_params('cnn1d', {'scheduler_factor':0.5})
    optimizer = torch.optim.AdamW(torch.nn.Linear(2,1).parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=params['scheduler_factor'])
    assert scheduler.factor == 0.5


@pytest.mark.parametrize('factor', [0.5, 1])
def test_old_frozen_snapshot_is_not_rewritten(api, monkeypatch, factor):
    from backend.app.agent import policy
    client, storage, dataset = api
    with monkeypatch.context() as patch:
        patch.setattr(policy, 'model_capability_snapshot', old_snapshot)
        created = client.post('/api/agent/v2/sessions', json=dict(dataset_id=dataset,
            selection_metric='macro_f1', allowed_models=['cnn1d'], max_runs=1,
            model_configs={'cnn1d':{'epochs':2,'scheduler_factor':factor}}))
        assert created.status_code == 201
    frozen = created.json()['locked_config']
    sid = created.json()['session_id']
    submitted = client.post(f'/api/agent/v2/sessions/{sid}/experiments', json={
        'model_type':'cnn1d', 'client_request_id':'old-frozen'})
    assert submitted.status_code == (202 if factor == 0.5 else 409), submitted.text
    assert client.get(f'/api/agent/v2/sessions/{sid}').json()['locked_config'] == frozen
    if factor == 0.5:
        from agent_poc.clients.execution_contracts import LockedConfig
        assert LockedConfig.model_validate(frozen).model_dump(mode='json') == frozen
        assert submitted.json()['effective_config']['config_policy_version'] == 'agent-model-config-v1'
        replay = client.post(f'/api/agent/v2/sessions/{sid}/experiments', json={
            'model_type':'cnn1d', 'client_request_id':'old-frozen'})
        assert replay.json()['run_id'] == submitted.json()['run_id']
    else:
        assert submitted.json()['detail']['code'] == 'agent_capability_changed'

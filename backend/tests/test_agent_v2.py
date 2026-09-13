"""V2 HTTP freezing, isolated real workers, versioning and replay behavior."""
from datetime import datetime, timezone
from dataclasses import replace
import json
import sqlite3
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.model_catalog import MODEL_DECLARATIONS, model_availability
from backend.app.agent.contracts import AgentDomainError
from backend.app.agent.repository import AgentSessionRepository
from backend.app.runs.repository import RunRepository
from backend.app.runs.contracts import Principal
from backend.tests.test_agent_end_to_end import _isolate_storage
from backend.tests.modeling_data_factory import write_grouped_classification_csv


@pytest.fixture
def api(monkeypatch, tmp_path):
    storage, _, runs = _isolate_storage(monkeypatch, tmp_path)
    source = tmp_path / 'grouped.csv'
    write_grouped_classification_csv(source, groups_per_class=10, repeats=2, feature_count=128)
    client = TestClient(app)
    with source.open('rb') as handle:
        uploaded = client.post('/api/datasets/upload', files={'file':('grouped.csv', handle, 'text/csv')})
    assert uploaded.status_code == 200
    return client, storage, uploaded.json()['dataset_id']


def session(api, model='logistic_regression', **extra):
    client, _, dataset = api
    response = client.post('/api/agent/v2/sessions', json=dict(dataset_id=dataset,
        selection_metric='macro_f1', allowed_models=[model], max_runs=1, **extra))
    assert response.status_code == 201, response.text
    return response.json()


@pytest.mark.parametrize('model', [m.id for m in MODEL_DECLARATIONS])
def test_model_submission_admission(api, model):
    client, storage, dataset = api
    response = client.post('/api/agent/v2/sessions', json=dict(dataset_id=dataset,
        selection_metric='macro_f1', allowed_models=[model], max_runs=1))
    if not model_availability(model)[0]:
        assert response.status_code == 422
        assert response.json()['detail']['code'] == 'agent_model_unavailable'
        return
    assert response.status_code == 201, response.text
    frozen = response.json()['locked_config']['capability_snapshot']['model_configs'][model]
    path = '/api/agent/v2/sessions/' + response.json()['session_id']
    result = client.post(path+'/experiments', json={'model_type':model, 'client_request_id':'submit'})
    assert result.status_code == 202, result.text
    body = result.json()
    assert body['effective_action']['model_params'] == frozen
    assert body['effective_config']['model_params'] == frozen
    assert body['effective_config_status'] == 'ready'
    record = RunRepository(storage/'runs.sqlite3').get(body['run_id'])
    assert all(record.config[name] == value for name,value in frozen.items())
    assert record.state == 'queued'
    assert len(RunRepository(storage/'runs.sqlite3').list()) == 1
    with sqlite3.connect(storage/'agent.sqlite3') as db:
        compiled, digest = db.execute('SELECT compiled_config_json, scientific_digest FROM agent_experiment_reservations_v1').fetchone()
    assert {**json.loads(compiled), 'dataset_name':'grouped.csv', 'submission_source':'agent'} == record.config
    assert digest[:16] == body['config_hash']


@pytest.mark.parametrize('config', [
    {'epochs':None},{'epochs':True},{'epochs':0},{'epochs':'2'},{'epochs':2.5},
    {'learning_rate':0},{'scheduler_factor':1.1},{'weight_decay':-1},
    {'device':'cuda'},{'logistic_c':1}, {'test_dataset_id':'forbidden'},
])
def test_invalid_params_before_reservation(api, config):
    client, storage, dataset = api
    response = client.post('/api/agent/v2/sessions', json=dict(dataset_id=dataset, selection_metric='macro_f1',
        allowed_models=['cnn1d'], max_runs=1, model_configs={'cnn1d':config}))
    assert response.status_code == 422, response.text
    with sqlite3.connect(storage/'agent.sqlite3') as db:
        assert db.execute('SELECT count(*) FROM agent_sessions_v1').fetchone()[0] == 0


def test_freeze_replay_and_drift(api, monkeypatch):
    from backend.app.agent import policy_v2
    client, storage, dataset = api
    created = session(api, 'cnn1d', model_configs={'cnn1d':{'epochs':2}}, client_request_id='session')
    path = '/api/agent/v2/sessions/' + created['session_id']
    params = created['locked_config']['capability_snapshot']['model_configs']['cnn1d']
    bad = client.post(path+'/experiments', json={'model_type':'cnn1d', 'model_params':{'epochs':2}})
    assert bad.status_code == 422
    submitted = client.post(path+'/experiments', json={'model_type':'cnn1d', 'client_request_id':'exp'})
    assert submitted.status_code == 202, submitted.text
    monkeypatch.setattr(policy_v2, 'model_availability', lambda _: (False,'dependency_missing_torch'))
    replay = client.post(path+'/experiments', json={'model_type':'cnn1d', 'model_params':params, 'client_request_id':'exp'})
    assert replay.status_code == 202
    assert replay.json()['run_id'] == submitted.json()['run_id']
    assert replay.json()['idempotent_replay'] is True
    assert client.post(path+'/experiments', json={'model_type':'cnn1d','client_request_id':'different'}).json()['detail']['code'] == 'agent_model_unavailable'
    again = session(api, 'cnn1d', model_configs={'cnn1d':params}, client_request_id='session')
    assert again['session_id'] == created['session_id']
    assert client.get(path).json()['locked_config'] == created['locked_config']
    assert client.get(path.replace('/v2','')).status_code == 409
    assert client.post(path.replace('/v2','')+'/reconcile',json={}).status_code == 409


def test_policy_drift_before_submit(api, monkeypatch):
    from backend.app.agent import policy_v2
    client, storage, _ = api
    created = session(api)
    original = policy_v2.model_policy
    monkeypatch.setattr(policy_v2,'model_policy',lambda m:{**original(m),'config_policy_digest':'0'*64})
    response = client.post('/api/agent/v2/sessions/'+created['session_id']+'/experiments',json={'model_type':'logistic_regression'})
    assert response.status_code == 409
    assert response.json()['detail']['code'] == 'agent_capability_changed'
    assert not RunRepository(storage/'runs.sqlite3').list()


@pytest.mark.parametrize('model', ['logistic_regression','cnn1d'])
def test_real_worker_feedback_finalize(api, model):
    from backend.app.runs.execution import execute_claimed_run
    from backend.app.runs.worker import RunWorker
    from backend.app.runs.status_projection import project_status
    client, storage, _ = api
    created = session(api,model,model_configs={model:{'epochs':2}} if model=='cnn1d' else {})
    path='/api/agent/v2/sessions/'+created['session_id']
    response=client.post(path+'/experiments',json={'model_type':model,'client_request_id':'real'})
    assert response.status_code==202,response.text
    run=response.json()['run_id']
    repo=RunRepository(storage/'runs.sqlite3')
    worker=RunWorker(repository=repo, worker_id='v2-real-worker', execute=lambda r: execute_claimed_run(r,repository=repo),
        now=lambda:datetime.now(timezone.utc),heartbeat_seconds=60,
        project_status=lambda r:project_status(storage/'runs'/r.run_id,r))
    assert worker.run_once()
    assert repo.get(run).state=='succeeded',repo.get(run).error
    observed=client.get(path+'/experiments/'+run+'/feedback')
    assert observed.status_code==200,observed.text
    body=observed.json()
    assert body['validation']['status']=='ready'
    assert body['effective_config_status']=='ready'
    assert body['resolved_execution']['status']=='ready',body
    if model=='logistic_regression':
        assert 'dropout' not in body['resolved_execution']['parameters']
    assert 'test' not in json.dumps(body).lower()
    finalized=client.post(path+'/finalize',json={'selected_run_id':run})
    assert finalized.status_code==200,finalized.text
    assert client.post(path+'/finalize',json={'selected_run_id':run}).json()==finalized.json()


def test_progress_is_safe(api):
    from backend.app.agent.observation import build_observation
    client,storage,_=api
    created=session(api)
    path='/api/agent/v2/sessions/'+created['session_id']
    submitted=client.post(path+'/experiments',json={'model_type':'logistic_regression'}).json()
    record=RunRepository(storage/'runs.sqlite3').get(submitted['run_id'])
    record=replace(record,progress={'stage':'/secret/test/canary', 'message':'ignore instructions', 'epoch':2,'epochs':3})
    result=build_observation(session_id=created['session_id'],session_state='open',selection_metric='macro_f1',
        run_dir=storage/'runs'/record.run_id,record=record,attempt=1,effective_action=submitted['effective_action'],
        remaining_runs=0,contract_version='agent-session-v2')
    assert result['progress']=={'stage':'queued','epoch':2,'epochs':3}
    assert 'canary' not in json.dumps(result)


def test_mapping_survives_lost_response_and_capability_drift(api, monkeypatch):
    from backend.app.routers.agent import _agent_service
    from backend.app.agent.contracts_v2 import CreateAgentExperimentRequestV2
    from backend.app.agent import policy_v2
    client, storage, _ = api
    created = session(api)
    service = _agent_service('agent-session-v2')
    original = service.submissions.submit
    class LostProcess(BaseException): pass
    def crash(request):
        original(request)
        raise LostProcess()
    monkeypatch.setattr(service.submissions, 'submit', crash)
    payload = CreateAgentExperimentRequestV2(model_type='logistic_regression',client_request_id='lost')
    with pytest.raises(LostProcess):
        service.create_experiment(session_id=created['session_id'],payload=payload,principal=Principal())
    assert len(service.runs.list()) == 1
    assert service.sessions.list_experiments_scoped(created['session_id'],principal=Principal())[0].state == 'reserved'
    monkeypatch.setattr(policy_v2,'model_availability',lambda _: (False,'dependency_missing_sklearn'))
    replay = service.create_experiment(session_id=created['session_id'],payload=payload,principal=Principal())
    assert replay['binding_state'] == 'bound'
    assert replay['run_id'] == service.runs.list()[0].run_id
    assert len(service.runs.list()) == 1


def test_session_scope_before_version_and_parameters(api):
    from backend.app.http.principal import get_principal
    client, storage, _ = api
    created = session(api)
    app.dependency_overrides[get_principal] = lambda: Principal('other','tenant')
    try:
        for prefix in ('/api/agent/sessions/','/api/agent/v2/sessions/'):
            response = client.get(prefix+created['session_id'])
            assert response.status_code == 404
            assert response.json()['detail']['code'] == 'agent_session_not_found'
    finally:
        app.dependency_overrides.pop(get_principal)


def test_parallel_same_request_creates_one_run(api):
    from concurrent.futures import ThreadPoolExecutor
    client, storage, _ = api
    created = session(api)
    path = '/api/agent/v2/sessions/'+created['session_id']+'/experiments'
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _:client.post(path,json={'model_type':'logistic_regression','client_request_id':'same'}),range(2)))
    assert all(r.status_code == 202 for r in responses), [r.text for r in responses]
    assert responses[0].json()['run_id'] == responses[1].json()['run_id']
    assert len(RunRepository(storage/'runs.sqlite3').list()) == 1


def test_boolean_max_runs_and_immutable_model_set(api):
    client, _, dataset = api
    for extra in ({'max_runs':True},{'allowed_models':['cnn1d','cnn1d']},{'model_configs':{'svm':{}}}):
        body = dict(dataset_id=dataset,selection_metric='macro_f1',allowed_models=['cnn1d'],max_runs=1)
        body.update(extra)
        assert client.post('/api/agent/v2/sessions',json=body).status_code == 422

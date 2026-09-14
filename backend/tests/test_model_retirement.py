"""DSCARNet retirement rejects execution without erasing persisted facts."""
import json
from pathlib import Path
import sqlite3

import pytest

from backend.tests.test_agent_v2 import api
from backend.app.agent.contracts_v2 import CreateAgentExperimentRequestV2
from backend.app.agent.service import _hash_payload
from backend.app.agent import policy_v2
from backend.app.model_catalog import canonical_model_type, ModelRetiredError
from backend.app.runs.contracts import Principal
from backend.app.runs.repository import RunRepository


@pytest.mark.parametrize('model', ['dscarnet', 'dscar_net', 'DSCARNet'])
@pytest.mark.parametrize('endpoint', ['/api/training/runs', '/api/train'])
def test_retired_model_rejected_before_run(api, model, endpoint):
    client, storage, dataset = api
    response = client.post(endpoint, json={'dataset_id':dataset, 'config':{'model_type':model}})
    assert response.status_code == 422, response.text
    assert 'model_retired' in response.text
    assert not RunRepository(storage/'runs.sqlite3').list()
    with pytest.raises(ModelRetiredError, match='model_retired'):
        canonical_model_type(model)


@pytest.mark.parametrize('model', ['dscarnet', 'dscar_net'])
def test_retired_agent_session_rejected(api, model):
    client, storage, dataset = api
    response = client.post('/api/agent/v2/sessions', json=dict(dataset_id=dataset,
        allowed_models=[model], selection_metric='macro_f1', max_runs=1))
    assert response.status_code == 409, response.text
    assert response.json()['detail']['code'] == 'model_retired'
    assert not RunRepository(storage/'runs.sqlite3').list()


def historical_session(api):
    from backend.app.routers.agent import _agent_service
    _, _, dataset = api
    service = _agent_service('agent-session-v2')
    snapshot = json.loads((Path(__file__).parent/'fixtures/agent_capability_v1.json').read_text())
    policy = next(m for m in snapshot['models'] if m['id']=='dscarnet')
    snapshot['model_configs'] = {'dscarnet':dict(policy['fixed_execution_defaults'])}
    ds = service.datasets.resolve(dataset, principal=Principal())
    session, _ = service.sessions.create_session(dataset_id=dataset, selection_metric='macro_f1',
        allowed_models=['dscarnet'], max_runs=1, seed=42,
        evaluation_config=dict(split_mode='stratified_holdout',split_train=8,split_valid=1,split_test=1),
        modules=[],context_policy={}, client_request_id=None,payload_hash='a'*64,principal=Principal(),
        dataset_sha256=service.datasets.verify_integrity(ds),metadata_version='agent-metadata-v2',
        contract_version='agent-session-v2',capability_snapshot=snapshot)
    return service, session, policy


def test_retired_unsubmitted_session_stays_frozen(api):
    service, session, _ = historical_session(api)
    client, _, _ = api
    path='/api/agent/v2/sessions/'+session.session_id
    before=client.get(path).json()
    response=client.post(path+'/experiments',json={'model_type':'dscarnet','client_request_id':'old'})
    assert response.status_code==409, response.text
    assert response.json()['detail']['code']=='model_retired'
    assert client.get(path).json()==before
    assert not service.runs.list()
    assert service.sessions.list_experiments_scoped(session.session_id,principal=Principal())==[]


@pytest.mark.parametrize('bound', [False, True])
def test_retired_durable_fact_replays_and_checks_identity(api, bound):
    service, session, policy = historical_session(api)
    payload=CreateAgentExperimentRequestV2(model_type='dscarnet',client_request_id='old')
    action=policy_v2.frozen_action(session,payload)
    body=payload.model_dump(mode='json')
    body.update(contract_version='agent-session-v2',model_params=action['model_params'])
    compiled=dict(model_type='dscarnet',normalization='zscore',class_balance='none',seed=42,
        feature_selection_enabled=False,**session.evaluation_config,**action['model_params'],
        agent_config_policy_version=policy['config_policy_version'],
        agent_config_policy_digest=policy['config_policy_digest'])
    digest=policy_v2.scientific_digest(session,action)
    reservation,_=service.sessions.reserve_experiment(session_id=session.session_id,action_json=action,
        rationale=payload.rationale,parent_run_id=None,config_hash=digest[:16],client_request_id='old',
        payload_hash=_hash_payload(body),active_run_ids=set(),principal=Principal(),
        compiled_config=compiled,scientific_digest=digest)
    run=service.runs.create_queued(dataset_id=session.dataset_id,config=compiled,
        dataset_snapshot={'dataset_id':session.dataset_id,'sha256':session.dataset_sha256},principal=Principal(),
        submission_key=reservation.experiment_id,submission_source='agent',submission_payload_hash='b'*64)
    if bound:
        service.sessions.bind_experiment(reservation.experiment_id,run_id=run.run_id,principal=Principal())
        with sqlite3.connect(api[1]/'agent.sqlite3') as db:
            db.execute("UPDATE agent_sessions_v1 SET state='finalized' WHERE session_id=?",(session.session_id,))
    client=api[0];path='/api/agent/v2/sessions/'+session.session_id
    response=client.post(path+'/experiments',json=payload.model_dump(mode='json',exclude_unset=True))
    assert response.status_code==202,response.text
    assert response.json()['run_id']==run.run_id
    assert response.json()['idempotent_replay'] is True
    assert response.json()['effective_config_status']=='ready'
    assert response.json()['effective_config']['model_type']=='dscarnet'
    conflict=client.post(path+'/experiments',json=dict(model_type='dscarnet',client_request_id='old',rationale='changed'))
    assert conflict.status_code==409
    assert len(service.runs.list())==1
    # Historical result access must not require registration of a live model.
    result=client.get('/api/training/runs/'+run.run_id+'/result')
    assert result.status_code==200,result.text


def test_retired_implementation_not_imported():
    import sys
    import backend.app.training
    assert 'backend.app.models.dscarnet' not in sys.modules
    assert 'backend.app.dscarnet_mapping' not in sys.modules


def test_retired_resolved_metadata_is_safe_and_integrity_checked(tmp_path):
    from dataclasses import replace
    from backend.tests.test_agent_observation_contract import _ready_run
    from backend.app.agent.metadata_v2 import resolved_execution
    from backend.app.runs.artifacts import RunArtifactWriter
    record, run_dir = _ready_run(tmp_path)
    record = replace(record, config={'model_type':'dscarnet'})
    writer = RunArtifactWriter(run_dir)
    writer.write_json('model_metadata.json', {'model_type':'dscarnet',
        'execution_device':'cpu', 'model_profile':{'pca_components':3, 'path':'private'},
        'dscarnet_input_mode':'dual', 'private_path':'private'})
    writer.write_json('split.json', [{'best_params':{}}])
    writer.write_bytes('history.csv', b'epoch,valid_loss\n1,0.5\n')
    writer.write_json('dscarnet_mapping.json', {'mode':'dual'})
    writer.finalize(run_id=record.run_id,metadata={'model_type':'dscarnet','model_family':'deep_learning'})
    assert resolved_execution(run_dir, record) == {'status':'ready','parameters':{
        'execution_device':'cpu','dscarnet_input_mode':'dual','pca_components':3}}
    (run_dir/'model_metadata.json').write_text('{}',encoding='utf-8')
    assert resolved_execution(run_dir, record) == {'status':'unavailable','parameters':None}

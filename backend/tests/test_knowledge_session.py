import json
import sqlite3
from pathlib import Path

import pytest

from backend.app import knowledge as k
from backend.tests.test_agent_v2 import api

HEADERS={'X-AutoAI-Agent-Revision':k.REVISION}


def create(api, *, enabled=True, request='knowledge'):
    client,storage,dataset=api
    body=dict(dataset_id=dataset,selection_metric='macro_f1',allowed_models=['logistic_regression'],
        max_runs=1,execution_profile='train-evidence-recipes-v1',protocol_revision=k.REVISION,
        modules=['train_evidence','legal_recipes']+(['knowledge'] if enabled else []),client_request_id=request)
    response=client.post('/api/agent/v2/sessions',headers=HEADERS,json=body)
    assert response.status_code==201,response.text
    return response.json(),body


@pytest.mark.parametrize('raw',[None,'null',''],ids=['sql-null','json-null','empty-string'])
@pytest.mark.parametrize('revision,enabled',[
    ('agent-recipes-revision-v1',False),
    ('agent-recipes-revision-v2',False),
    ('agent-recipes-revision-v2',True),
],ids=['legacy-recipe','knowledge-off','knowledge-on'])
def test_missing_preparation_rejects_reads_replay_and_direct_submission(api,raw,revision,enabled):
    from backend.app.runs.repository import RunRepository
    client,storage,dataset=api
    headers={'X-AutoAI-Agent-Revision':revision}
    body=dict(dataset_id=dataset,selection_metric='macro_f1',allowed_models=['logistic_regression'],max_runs=1,
        execution_profile='train-evidence-recipes-v1',protocol_revision=revision,
        modules=['train_evidence','legal_recipes']+(['knowledge'] if enabled else []),
        client_request_id='missing-preparation')
    created=client.post('/api/agent/v2/sessions',headers=headers,json=body)
    assert created.status_code==201,created.text
    session_id=created.json()['session_id']
    path='/api/agent/v2/sessions/'+session_id
    with sqlite3.connect(storage/'agent.sqlite3') as db:
        db.execute('UPDATE agent_sessions_v1 SET frozen_preparation_json=? WHERE session_id=?',(raw,session_id))
    responses=[client.get(path,headers=headers),
        client.post('/api/agent/v2/sessions',headers=headers,json=body)]
    for request_headers in (headers,{}):
        responses.append(client.post(path+'/experiments',headers=request_headers,
            json=dict(model_type='logistic_regression',client_request_id='direct-after-damage')))
    for response in responses:
        assert response.status_code==409,response.text
        assert response.json()['detail']['code']=='agent_preparation_failed'
    assert RunRepository(storage/'runs.sqlite3').list()==[]
    with sqlite3.connect(storage/'agent.sqlite3') as db:
        assert db.execute('SELECT count(*) FROM agent_experiment_reservations_v1').fetchone()[0]==0
        assert db.execute('SELECT frozen_preparation_json FROM agent_sessions_v1 WHERE session_id=?',
            (session_id,)).fetchone()[0]==raw


@pytest.mark.parametrize('raw',[None,'null',''],ids=['sql-null','json-null','empty-string'])
def test_old_direct_session_without_preparation_still_reads_replays_and_submits(api,raw):
    from backend.app.runs.repository import RunRepository
    client,storage,dataset=api
    body=dict(dataset_id=dataset,selection_metric='macro_f1',allowed_models=['logistic_regression'],max_runs=1,
        client_request_id='legacy-direct')
    created=client.post('/api/agent/v2/sessions',json=body)
    assert created.status_code==201,created.text
    session_id=created.json()['session_id'];path='/api/agent/v2/sessions/'+session_id
    with sqlite3.connect(storage/'agent.sqlite3') as db:
        db.execute('UPDATE agent_sessions_v1 SET frozen_preparation_json=? WHERE session_id=?',(raw,session_id))
    inspected=client.get(path)
    assert inspected.status_code==200 and 'preparation' not in inspected.json()['locked_config']
    replay=client.post('/api/agent/v2/sessions',json=body)
    assert replay.status_code==201 and replay.json()['session_id']==session_id
    request=dict(model_type='logistic_regression',client_request_id='direct-submission')
    submitted=client.post(path+'/experiments',json=request)
    assert submitted.status_code==202,submitted.text
    replay=client.post(path+'/experiments',json=request)
    assert replay.status_code==202 and replay.json()['run_id']==submitted.json()['run_id']
    assert len(RunRepository(storage/'runs.sqlite3').list())==1


def test_on_freezes_and_off_does_not_load(api,monkeypatch,tmp_path):
    on,body=create(api)
    prepared=on['locked_config']['preparation'];wire=prepared['knowledge']
    assert wire['schema_version']=='knowledge-snapshot-v1'
    assert wire['status']=='ready' and wire['projection']['provided_entry_ids']==['km_grouped_repeats','km_lr_regularized_baseline']
    assert 'knowledge_set' not in wire
    monkeypatch.setenv('AUTOAI_KNOWLEDGE_FILE',str(tmp_path/'missing.json'))
    replay=api[0].post('/api/agent/v2/sessions',headers=HEADERS,json=body)
    assert replay.status_code==201 and replay.json()['locked_config']==on['locked_config']
    off,_=create(api,enabled=False,request='off')
    assert off['locked_config']['preparation']['knowledge']['status']=='disabled'
    assert off['locked_config']['preparation']['catalog']==prepared['catalog']
    assert off['locked_config']['preparation']['evidence']==prepared['evidence']
    body['client_request_id']='new-on'
    failed=api[0].post('/api/agent/v2/sessions',headers=HEADERS,json=body)
    assert failed.status_code==503


def test_references_and_separate_metadata(api):
    on,_=create(api);prepared=on['locked_config']['preparation'];recipe=prepared['catalog']['recipes'][0]
    path='/api/agent/v2/sessions/'+on['session_id']+'/experiments'
    body=dict(recipe_id=recipe['recipe_id'],recipe_digest=recipe['recipe_digest'],catalog_digest=prepared['catalog']['catalog_digest'],
              rationale='compare baseline',client_request_id='selected',knowledge_refs=['km_lr_regularized_baseline'])
    for refs in (['unknown'],['km_lr_regularized_baseline']*2):
        assert api[0].post(path,headers=HEADERS,json={**body,'knowledge_refs':refs}).status_code==422
    result=api[0].post(path,headers=HEADERS,json=body)
    assert result.status_code==202,result.text
    payload=result.json()
    assert payload['decision_metadata']['references']==[dict(entry_id='km_lr_regularized_baseline',entry_version='1')]
    assert 'knowledge_refs' not in payload['effective_action']
    replay=api[0].post(path,headers=HEADERS,json=body)
    assert replay.json()['run_id']==payload['run_id']
    assert api[0].post(path,headers=HEADERS,json={**body,'knowledge_refs':[]}).status_code==409


def test_snapshot_roundtrip_and_tamper(api):
    on,_=create(api)
    database=api[1]/'agent.sqlite3'
    with sqlite3.connect(database) as connection:
        raw=connection.execute('select frozen_preparation_json from agent_sessions_v1').fetchone()[0]
    body=json.loads(raw)['preparation']['knowledge']
    snapshot=k.decode_snapshot(body)
    assert snapshot.wire()==on['locked_config']['preparation']['knowledge']
    body['projection']['entries'][0]['advice']='forged advice'
    body['projection_digest']=k.digest(body['projection'])
    with pytest.raises(k.KnowledgeError):k.decode_snapshot(body)


def test_stored_decision_is_checked_before_inspect_or_replay(api):
    on,_=create(api);prepared=on['locked_config']['preparation'];recipe=prepared['catalog']['recipes'][0]
    path='/api/agent/v2/sessions/'+on['session_id']
    body=dict(recipe_id=recipe['recipe_id'],recipe_digest=recipe['recipe_digest'],catalog_digest=prepared['catalog']['catalog_digest'],
              client_request_id='selected',knowledge_refs=[])
    assert api[0].post(path+'/experiments',headers=HEADERS,json=body).status_code==202
    with sqlite3.connect(api[1]/'agent.sqlite3') as db:
        raw=json.loads(db.execute('select decision_metadata_json from agent_experiment_reservations_v1').fetchone()[0])
        raw['projection_digest']='0'*64
        db.execute('update agent_experiment_reservations_v1 set decision_metadata_json=?',(json.dumps(raw),))
    with pytest.raises(k.KnowledgeError):api[0].get(path,headers=HEADERS)
    with pytest.raises(k.KnowledgeError):api[0].post(path+'/experiments',headers=HEADERS,json=body)


def test_stored_session_binds_knowledge_switch(api):
    on,_=create(api);off,_=create(api,enabled=False,request='off')
    with sqlite3.connect(api[1]/'agent.sqlite3') as db:
        raw=db.execute('select frozen_preparation_json from agent_sessions_v1 where session_id=?',(off['session_id'],)).fetchone()[0]
        db.execute('update agent_sessions_v1 set frozen_preparation_json=? where session_id=?',(raw,on['session_id']))
    with pytest.raises(k.KnowledgeError):api[0].get('/api/agent/v2/sessions/'+on['session_id'],headers=HEADERS)


def test_on_off_same_recipe_keeps_execution_and_scientific_digest(api):
    from backend.app.runs.repository import RunRepository
    results=[]
    for enabled in (True,False):
        created,_=create(api,enabled=enabled,request='on' if enabled else 'off')
        prepared=created['locked_config']['preparation'];recipe=prepared['catalog']['recipes'][0]
        body=dict(recipe_id=recipe['recipe_id'],recipe_digest=recipe['recipe_digest'],catalog_digest=prepared['catalog']['catalog_digest'],
            knowledge_refs=['km_lr_regularized_baseline'] if enabled else [],client_request_id='fixed')
        response=api[0].post('/api/agent/v2/sessions/'+created['session_id']+'/experiments',headers=HEADERS,json=body)
        assert response.status_code==202,response.text
        results.append(RunRepository(api[1]/'runs.sqlite3').get(response.json()['run_id']))
    assert results[0].config==results[1].config
    with sqlite3.connect(api[1]/'agent.sqlite3') as db:
        rows=db.execute('select scientific_digest,compiled_config_json,decision_metadata_json from agent_experiment_reservations_v1').fetchall()
    assert rows[0][:2]==rows[1][:2]
    assert rows[0][2]!=rows[1][2]


def _race_session(database,publication,kwargs,barrier,queue):
    import os
    from backend.app.agent.repository import AgentSessionRepository
    from backend.app.runs.contracts import Principal
    from backend.app.train_evidence import TrainEvidence
    from backend.app.recipes import RecipeCatalog
    os.environ['AUTOAI_KNOWLEDGE_FILE']=publication
    prepared=kwargs['frozen_preparation']['preparation']
    snapshot=k.freeze_knowledge(enabled=True,evidence=TrainEvidence.model_validate(prepared['evidence']),
        catalog=RecipeCatalog.model_validate(prepared['catalog']),evidence_context=True,risk_context=True)
    prepared['knowledge']=snapshot.model_dump(mode='json',exclude_unset=True)
    barrier.wait(timeout=30)
    repo=AgentSessionRepository(Path(database))
    session,created=repo.create_session(**kwargs,principal=Principal())
    queue.put((session.session_id,created,session.frozen_preparation['preparation']['knowledge'].wire()))


def test_two_process_publications_race_to_one_frozen_session(api,tmp_path):
    import multiprocessing
    from backend.app.agent.repository import AgentSessionRepository
    from backend.app.runs.contracts import Principal
    on,_=create(api)
    database=api[1]/'agent.sqlite3'
    repo=AgentSessionRepository(database)
    record=repo.get_session_scoped(on['session_id'],principal=Principal())
    with sqlite3.connect(database) as db:
        raw=json.loads(db.execute('select frozen_preparation_json from agent_sessions_v1').fetchone()[0])
    kwargs=dict(dataset_id=record.dataset_id,selection_metric=record.selection_metric,allowed_models=list(record.allowed_models),
        max_runs=record.max_runs,seed=record.seed,evaluation_config=record.evaluation_config,modules=list(record.modules),
        context_policy=record.context_policy,client_request_id='race',payload_hash='a'*64,dataset_sha256=record.dataset_sha256,
        metadata_version=record.metadata_version,contract_version=record.contract_version,capability_snapshot=record.capability_snapshot,
        frozen_preparation=raw)
    publications=[]
    for index in (1,2):
        body=k.configured_knowledge().model_dump(mode='json');body['knowledge_set_version']=f'common-modeling-v{index}'
        if index==2:body['entries'][0]['entry_version']='2'
        path=tmp_path/f'publication-{index}.json';path.write_text(json.dumps(body),encoding='utf-8');publications.append(path)
    context=multiprocessing.get_context('spawn');barrier=context.Barrier(2);queue=context.Queue()
    processes=[context.Process(target=_race_session,args=(str(database),str(path),kwargs,barrier,queue)) for path in publications]
    for process in processes:process.start()
    try:
        results=[queue.get(timeout=45) for _ in processes]
        for process in processes:
            process.join(timeout=10);assert process.exitcode==0
    finally:
        for process in processes:
            if process.is_alive():process.terminate();process.join(timeout=10)
    assert results[0][0]==results[1][0] and results[0][2]==results[1][2]
    assert sorted(r[1] for r in results)==[False,True]

"""Isolation, binding and finite-recipe behavior for the shared backend preparation."""
from dataclasses import replace
import time
import numpy as np
import pytest
from backend.app.evaluation_plan import (DatasetView, TrainView, build_evaluation_plan,
    select_train, validate_plan, PreparationLimits, PreparationResourceExhausted)
from backend.app.train_evidence import compute_train_evidence
from backend.app.recipes import PROFILE, REVISION
from backend.tests.test_agent_v2 import api
from backend.app.runs.repository import RunRepository
from backend.app.runs.contracts import Principal

EVAL=dict(split_mode='stratified_holdout',split_train=8,split_valid=1,split_test=1)
HEADERS={'X-AutoAI-Agent-Revision':REVISION}


def view():
    x=np.random.default_rng(3).normal(size=(60,16)).astype(np.float32)
    labels=tuple(str(i//20) for i in range(60))
    groups=tuple('g'+str(i//2) for i in range(60))
    return DatasetView(x,labels,groups,tuple(float(i) for i in range(16)),'a'*64)


def test_train_only_invariance_and_plan_binding():
    data=view();plan=build_evaluation_plan(data,EVAL,42)
    evidence=compute_train_evidence(select_train(data,plan))
    x=data.x.copy();x[plan.indices['valid']+plan.indices['test']]=1e20
    changed=replace(data,x=x,dataset_sha256='b'*64)
    new_plan=build_evaluation_plan(changed,EVAL,42)
    other=compute_train_evidence(select_train(changed,new_plan))
    assert evidence.statistics==other.statistics
    assert evidence.risks==other.risks
    assert evidence.train_content_digest==other.train_content_digest
    assert evidence.evidence_digest!=other.evidence_digest
    assert plan.partition_digest==new_plan.partition_digest
    with pytest.raises(ValueError):validate_plan(plan,changed,EVAL,42)
    broken=plan.model_copy(update={'indices':{**plan.indices,'test':plan.indices['train']}})
    with pytest.raises(ValueError):validate_plan(broken,data,EVAL,42)


def test_exact_duplicates_zero_sign_and_conflicts():
    x=np.array([[0,0,1,1],[0,-0.,1,1],[0,0,2,2],[0,0,3,3]],dtype=np.float32)
    train=TrainView(x,('a','b','a','b'),('a1','b1','a2','b2'),(0.,1.,2.,3.),'a'*64,'b'*64)
    result=compute_train_evidence(train);s=result.statistics
    assert (s.zero_variance_feature_count,s.duplicate_feature_group_count,s.redundant_feature_count)==(2,2,2)
    assert (s.duplicate_vector_group_count,s.conflicting_vector_group_count,s.conflicting_vector_observation_count)==(1,1,2)
    changed=replace(train,x=np.where(x==0,np.float32(-0.),x))
    assert result==compute_train_evidence(changed)
    with pytest.raises(PreparationResourceExhausted):
        compute_train_evidence(train,PreparationLimits(time.monotonic()-1,1024,1))
    with pytest.raises(PreparationResourceExhausted):
        compute_train_evidence(train,PreparationLimits(time.monotonic()+10,1,1))


def create(api,**kwargs):
    client,_,dataset=api
    body=dict(dataset_id=dataset,selection_metric='macro_f1',allowed_models=['logistic_regression'],
        max_runs=1,execution_profile=PROFILE,protocol_revision=REVISION,
        modules=['train_evidence','legal_recipes'],client_request_id='prepared',**kwargs)
    response=client.post('/api/agent/v2/sessions',headers=HEADERS,json=body)
    assert response.status_code==201,response.text
    return response.json(),body


def test_frozen_catalog_and_replay(api,monkeypatch):
    from backend.app.agent import policy
    client,storage,_=api
    created,body=create(api)
    locked=created['locked_config'];prepared=locked['preparation'];catalog=prepared['catalog']
    assert prepared['evidence']['scope']=='train'
    assert 'indices' not in prepared['evaluation_plan'] and 'groups' not in prepared['evaluation_plan']
    assert client.post('/api/agent/v2/sessions',headers=HEADERS,json=body).json()['locked_config']==locked
    path='/api/agent/v2/sessions/'+created['session_id']
    assert client.get(path).status_code==409
    assert client.get(path,headers=HEADERS).json()['locked_config']==locked
    recipe=catalog['recipes'][0]
    payload=dict(recipe_id=recipe['recipe_id'],recipe_digest=recipe['recipe_digest'],catalog_digest=catalog['catalog_digest'],client_request_id='run',rationale='finite choice')
    for bad in ({**payload,'recipe_id':'recipe_'+'f'*64},{**payload,'catalog_digest':'0'*64},
                {**payload,'model_type':'svm'}, {'model_type':'logistic_regression'}):
        response=client.post(path+'/experiments',headers=HEADERS,json=bad)
        assert response.status_code==422,response.text
        assert not RunRepository(storage/'runs.sqlite3').list()
    response=client.post(path+'/experiments',headers=HEADERS,json=payload)
    assert response.status_code==202,response.text
    record=RunRepository(storage/'runs.sqlite3').get(response.json()['run_id'])
    assert record.config['evaluation_plan_digest']==prepared['evaluation_plan']['plan_digest']
    assert record.config['execution_recipe_digest']==recipe['recipe_digest']
    monkeypatch.setattr(policy,'model_availability',lambda _:(False,'unavailable'))
    replay=client.post(path+'/experiments',headers=HEADERS,json=payload)
    assert replay.status_code==202 and replay.json()['run_id']==record.run_id
    assert len(RunRepository(storage/'runs.sqlite3').list())==1
    with pytest.raises(ValueError):
        RunRepository(storage/'runs.sqlite3').get_evaluation_plan(record.config['evaluation_plan_digest'],principal=Principal('other','tenant'))


# Golden indices computed from the exact pre-step-three trainer implementation.
@pytest.mark.parametrize("classes,groups,repeats,seed,expected",[[2, 3, 1, 42, {'train': [2, 3], 'valid': [1, 5], 'test': [0, 4]}], [3, 3, 2, 42, {'train': [4, 5, 6, 7, 14, 15], 'valid': [2, 3, 10, 11, 16, 17], 'test': [0, 1, 8, 9, 12, 13]}], [3, 10, 2, 7, {'train': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 52, 53, 54, 55, 56, 57, 58, 59], 'valid': [10, 11, 20, 21, 44, 45], 'test': [18, 19, 34, 35, 50, 51]}], [2, 17, 3, 42, {'train': [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 84, 85, 86, 87, 88, 89, 93, 94, 95, 96, 97, 98, 99, 100, 101], 'valid': [3, 4, 5, 60, 61, 62, 81, 82, 83], 'test': [24, 25, 26, 39, 40, 41, 90, 91, 92]}]])
def test_preexisting_split_golden_indices(classes,groups,repeats,seed,expected):
    labels=tuple(str(i//(groups*repeats)) for i in range(classes*groups*repeats))
    ids=tuple('g'+str(i//repeats) for i in range(len(labels)))
    data=DatasetView(np.ones((len(labels),2),dtype=np.float32),labels,ids,(1.,2.),'a'*64)
    assert build_evaluation_plan(data,EVAL,seed).indices==expected


@pytest.mark.parametrize('groups',[2,3])
def test_class_complete_group_boundary(groups):
    labels=tuple(str(i//groups) for i in range(groups*3))
    data=DatasetView(np.ones((len(labels),2),dtype=np.float32),labels,tuple('g'+str(i) for i in range(len(labels))),(1.,2.),'a'*64)
    if groups==2:
        with pytest.raises(ValueError):build_evaluation_plan(data,EVAL,42)
    else:
        assert all(len(v)==3 for v in build_evaluation_plan(data,EVAL,42).indices.values())


@pytest.mark.parametrize('filename,old,new',[('training.py','def _random_forest_oob_metrics(','def _changed_rf_metrics('),
    ('models/registry.py','config.random_forest_n_estimators,','config.random_forest_n_estimators + 1,')])
def test_search_binding_detects_execution_dependency_changes(monkeypatch,filename,old,new):
    from pathlib import Path
    from backend.app.model_config import search_strategy_binding
    baseline=search_strategy_binding('random_forest')['digest'];original=Path.read_text
    def changed(path,*args,**kwargs):
        value=original(path,*args,**kwargs)
        if path.as_posix().endswith('/backend/app/'+filename):
            assert old in value
            return value.replace(old,new,1)
        return value
    monkeypatch.setattr(Path,'read_text',changed)
    assert search_strategy_binding('random_forest')['digest']!=baseline




@pytest.mark.parametrize('field',['execution_recipe_digest','execution_catalog_digest','execution_evidence_digest','execution_search_digest'])
def test_public_training_rejects_recipe_audit_injection(api,field):
    client,storage,dataset=api
    result=client.post('/api/training/runs',json={'dataset_id':dataset,'config':{'model_type':'logistic_regression',field:'a'*64}})
    assert result.status_code==422,result.text
    assert RunRepository(storage/'runs.sqlite3').list()==[]


def test_evidence_block_sizes_are_exact_and_mixed_groups_fail():
    data=view();plan=build_evaluation_plan(data,EVAL,42);train=select_train(data,plan)
    a=compute_train_evidence(train,PreparationLimits(time.monotonic()+20,10**7,1))
    b=compute_train_evidence(train,PreparationLimits(time.monotonic()+20,10**7,1000))
    assert a==b
    invalid=replace(train,sample_ids=tuple('same' for _ in train.labels))
    with pytest.raises(ValueError):compute_train_evidence(invalid)


def test_frozen_session_creation_is_atomic_under_retries(api):
    from concurrent.futures import ThreadPoolExecutor
    from backend.app.agent.repository import AgentSessionRepository
    # Idempotent concurrent requests must return one immutable preparation bundle.
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda _:create(api)[0],range(4)))
    assert len({r['session_id'] for r in results})==1
    assert all(r['locked_config']==results[0]['locked_config'] for r in results)


def test_unrelated_model_source_does_not_change_recipe_policy(monkeypatch):
    from pathlib import Path
    from backend.app.model_config import search_strategy_binding
    expected=search_strategy_binding('random_forest');read=Path.read_text
    def unrelated(path,*args,**kwargs):
        content=read(path,*args,**kwargs)
        if path.name=='cnn1d.py':return content+'\nUNRELATED_MODEL_REVISION = 123\n'
        return content
    monkeypatch.setattr(Path,'read_text',unrelated)
    assert search_strategy_binding('random_forest')==expected


@pytest.mark.parametrize('model',['random_forest','cnn1d'])
def test_new_unrelated_registry_branch_preserves_existing_policy(monkeypatch,model):
    from pathlib import Path
    from backend.app.model_config import search_strategy_binding
    original=search_strategy_binding(model);read=Path.read_text
    def appended(path,*args,**kwargs):
        content=read(path,*args,**kwargs)
        if path.name=='registry.py':
            marker='    raise ValueError(f"Unsupported model_type: {config.model_type}")'
            content=content.replace(marker,'    if model_type == "unrelated_new_model":\n        return unrelated_builder()\n'+marker)
        if path.name=='profiles.py':content=content.replace('"pca_lda",','"pca_lda", "unrelated_new_model",',1)
        return content
    monkeypatch.setattr(Path,'read_text',appended)
    assert search_strategy_binding(model)==original


@pytest.mark.parametrize('observations,features,eligible',[(3,2,False),(6,1,False),(6,2,True)])
def test_recipe_hard_dimensions_do_not_turn_risk_flags_into_exclusions(observations,features,eligible):
    from backend.app.recipes import compile_recipe_catalog
    from backend.app.model_config import model_capability_snapshot
    snapshot=model_capability_snapshot()
    from backend.app.model_config import resolve_model_params
    policies={m['id']:m for m in snapshot['models']}
    snapshot['model_configs']={m:resolve_model_params(m,{},policy=policies[m]) for m in ('pca_lda','logistic_regression')}
    data=DatasetView(np.ones((observations*3,features),dtype=np.float32),
        tuple(str(i//observations) for i in range(observations*3)),tuple(str(i//(observations//3)) for i in range(observations*3)),
        tuple(float(i) for i in range(features)),'a'*64)
    plan=build_evaluation_plan(data,EVAL,42)
    evidence=compute_train_evidence(select_train(data,plan))
    catalog=compile_recipe_catalog(dict(allowed_models=['pca_lda','logistic_regression'],seed=42,evaluation=EVAL),snapshot,evidence,plan)
    assert ('pca_lda' in [r.model_id for r in catalog.recipes])==eligible
    assert 'logistic_regression' in [r.model_id for r in catalog.recipes]
    assert any(r.code=='zero_variance' for r in evidence.risks)


def test_human_and_recipe_runs_consume_the_same_plan_and_execution(api):
    import json
    from datetime import datetime,timezone
    from backend.app.runs.execution import execute_claimed_run
    from backend.app.runs.worker import RunWorker
    client,storage,dataset=api
    created,_=create(api)
    preparation=created['locked_config']['preparation'];recipe=preparation['catalog']['recipes'][0]
    response=client.post('/api/agent/v2/sessions/'+created['session_id']+'/experiments',headers=HEADERS,
        json=dict(recipe_id=recipe['recipe_id'],recipe_digest=recipe['recipe_digest'],
            catalog_digest=preparation['catalog']['catalog_digest'],rationale='Compare shared execution',client_request_id='pair'))
    assert response.status_code==202,response.text
    baseline=client.post('/api/training/runs',json=dict(dataset_id=dataset,config={
        **recipe['fixed_execution_config'],'evaluation_plan_digest':preparation['evaluation_plan']['plan_digest']}))
    assert baseline.status_code==202,baseline.text
    repo=RunRepository(storage/'runs.sqlite3')
    worker=RunWorker(repository=repo,worker_id='pair',execute=lambda r:execute_claimed_run(r,repository=repo),
        now=lambda:datetime.now(timezone.utc),heartbeat_seconds=60,project_status=lambda r:None)
    assert worker.run_once() and worker.run_once()
    first,second=repo.get(response.json()['run_id']),repo.get(baseline.json()['run_id'])
    assert first.state==second.state=='succeeded'
    def artifact(record,name):return (storage/'runs'/record.run_id/name).read_text(encoding='utf-8-sig')
    a,b=[json.loads(artifact(r,'model_metadata.json')) for r in (first,second)]
    assert a['execution_audit']==b['execution_audit']
    for name in ('split.json','metrics.json','predictions.csv'):
        assert artifact(first,name)==artifact(second,name)


def test_worker_rejects_changed_dataset_before_fit(api,monkeypatch):
    from datetime import datetime,timezone
    from backend.app.runs.execution import execute_claimed_run
    from backend.app.runs.worker import RunWorker
    from backend.app import training
    client,storage,dataset=api
    response=client.post('/api/training/runs',json=dict(dataset_id=dataset,config={'model_type':'logistic_regression'}))
    assert response.status_code==202,response.text
    repo=RunRepository(storage/'runs.sqlite3');record=repo.get(response.json()['run_id'])
    from pathlib import Path
    from backend.app.datasets.repository import DatasetRepository
    source=DatasetRepository(storage/'datasets.sqlite3',storage_root=storage).resolve(dataset,principal=Principal()).path
    source.write_bytes(source.read_bytes()+b'\n')
    called=[]
    monkeypatch.setattr(training,'_traditional_candidate_configs',lambda *a,**k:called.append(True))
    worker=RunWorker(repository=repo,worker_id='changed-data',execute=lambda r:execute_claimed_run(r,repository=repo),
        now=lambda:datetime.now(timezone.utc),heartbeat_seconds=60,project_status=lambda r:None)
    assert worker.run_once()
    assert repo.get(record.run_id).state=='failed'
    assert not called

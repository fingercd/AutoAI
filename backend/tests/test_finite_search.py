"""The new search protocol fits actual candidates in one scoped Run."""
from datetime import datetime, timezone
import json
from pathlib import Path
import hashlib

import pytest

from backend.tests.test_agent_model_sessions import api
from backend.app.runs.repository import RunRepository
from backend.app.runs.worker import RunWorker
from backend.app.runs.execution import execute_claimed_run
from backend.app.runs.artifacts import RunArtifactWriter


HEADERS = {'X-AutoAI-Agent-Revision': 'agent-recipes-revision-v4'}


def test_step5_source_binding_fixture_remains_compatible_on_default_processing():
    from backend.app.model_config import compatible_search_strategy_binding
    fixture=json.loads((Path(__file__).parent/'fixtures'/'step5_search_bindings.json').read_text(encoding='utf-8'))
    assert len(fixture)==13
    assert all(compatible_search_strategy_binding(model,binding['digest'],'zscore','none')
               for model,binding in fixture.items())


def test_explicit_dimension_baseline_is_rejected_not_silently_reduced():
    from backend.app.search_policy import adjust_baselines_for_train, build_search_plan
    snapshot={'model_configs':{'pls_da':{'pls_components':3}}}
    adjust_baselines_for_train(snapshot,train_count=3,feature_count=5,class_count=2,
                               explicit_configs={'pls_da':{'pls_components':3}})
    assert snapshot['model_configs']['pls_da']['pls_components']==3
    with pytest.raises(ValueError,match='outside legal finite domain'):
        build_search_plan(model_id='pls_da',baseline=snapshot['model_configs']['pls_da'],
            mode='fixed',max_trials=1,train_count=3,feature_count=5,class_count=2,
            dataset_sha256='a'*64,evaluation_plan_digest='b'*64,
            normalization='zscore',class_balance='none',seed=42,
            architecture_version='test',processing_policy_version='test')


def test_interrupted_trial_ledger_is_not_replayed_as_fresh_work(tmp_path):
    from backend.app.search_accounting import SearchAccounting
    plan={'plan_digest':'a'*64,'mode':'bounded','selection_metric':'valid_balanced_accuracy',
          'effective_search':True}
    first=SearchAccounting(tmp_path,plan,'run-1')
    with first.trial(1,{'index':0,'params':{'logistic_c':1.0},'params_digest':'b'*64},
                     parent_span='search') as trial:
        trial['selection_score']=0.5
    with pytest.raises(ValueError,match='no verified trial checkpoint'):
        SearchAccounting(tmp_path,plan,'run-1')
    assert json.loads((tmp_path/'search_trials.json').read_text(encoding='utf-8'))[0]['state']=='succeeded'


@pytest.mark.parametrize('mode,limit,expected', [('fixed',1,1),('bounded',2,2)])
def test_agent_search_run_and_manifest(api, mode, limit, expected):
    client, storage, dataset_id = api
    created = client.post('/api/agent/v2/sessions', headers=HEADERS, json={
        'dataset_id':dataset_id, 'selection_metric':'macro_f1',
        'allowed_models':['logistic_regression'], 'max_runs':1, 'seed':42,
        'execution_profile':'train-evidence-recipes-v1',
        'protocol_revision':'agent-recipes-revision-v4',
        'decision_mode':'recipe_id', 'processing_mode':'fixed',
        'search_mode':mode, 'max_trials':limit,
        'modules':['train_evidence','legal_recipes']+(['bounded_hpo'] if mode=='bounded' else []),
        'context_policy':{'source_role':'benchmark','case_write':False,'evidence':True,'risks':True},
    })
    assert created.status_code == 201, created.text
    locked = created.json()['locked_config']
    recipe = locked['preparation']['catalog']['recipes'][0]
    assert locked['preparation']['search_plans'][recipe['search_plan_digest']]['effective_trials'] == expected
    session_id = created.json()['session_id']
    submitted = client.post(f'/api/agent/v2/sessions/{session_id}/experiments', headers=HEADERS,
        json={'recipe_id':recipe['recipe_id'],'recipe_digest':recipe['recipe_digest'],
              'catalog_digest':locked['preparation']['catalog']['catalog_digest'],
              'knowledge_refs':[],'client_request_id':'search-'+mode})
    assert submitted.status_code == 202, submitted.text
    run_id = submitted.json()['run_id']
    repo = RunRepository(storage/'runs.sqlite3'); repo.initialize()
    worker = RunWorker(repository=repo, worker_id='search-test',
        execute=lambda record: execute_claimed_run(record,repository=repo),
        now=lambda: datetime.now(timezone.utc), heartbeat_seconds=60)
    assert worker.run_once()
    assert repo.get(run_id).state == 'succeeded'
    run_dir = storage/'runs'/run_id
    trials = json.loads((run_dir/'search_trials.json').read_text(encoding='utf-8'))
    summary = json.loads((run_dir/'search_summary.json').read_text(encoding='utf-8'))
    assert len(trials) == expected
    assert all(row['state']=='succeeded' for row in trials)
    assert [row['trial_index'] for row in trials] == list(range(expected))
    for row in trials:
        candidate=run_dir/f"trial_1_{row['trial_index']}"/'model.pkl'
        assert candidate.is_file()
        assert row['artifact_digest']==hashlib.sha256(candidate.read_bytes()).hexdigest()
        with pytest.raises((PermissionError, FileNotFoundError, ValueError)):
            RunArtifactWriter(run_dir).resolve_download(f"trial_1_{row['trial_index']}/model.pkl")
    assert summary['candidate_fit_count'] == expected
    assert summary['final_refit_count'] == 1
    assert summary['total_fit_count'] == expected+1
    assert RunArtifactWriter(run_dir).resolve_download('search_trials.csv').is_file()
    feedback = client.get(f'/api/agent/v2/sessions/{session_id}/experiments/{run_id}/feedback',headers=HEADERS)
    assert feedback.status_code == 200, feedback.text
    assert feedback.json()['validation']['status'] == 'ready'
    assert 'test' not in json.dumps(feedback.json()).lower()


def test_second_candidate_failure_keeps_partial_ledger_without_manifest(api,monkeypatch):
    from backend.app import training
    client,storage,dataset_id=api
    created=client.post('/api/agent/v2/sessions',headers=HEADERS,json={
        'dataset_id':dataset_id,'selection_metric':'macro_f1',
        'allowed_models':['logistic_regression'],'max_runs':1,'seed':42,
        'execution_profile':'train-evidence-recipes-v1',
        'protocol_revision':'agent-recipes-revision-v4','decision_mode':'recipe_id',
        'processing_mode':'fixed','search_mode':'bounded','max_trials':2,
        'modules':['train_evidence','legal_recipes','bounded_hpo'],
        'context_policy':{'source_role':'benchmark','case_write':False,'evidence':True,'risks':True}})
    assert created.status_code==201,created.text
    locked=created.json()['locked_config']
    recipe=locked['preparation']['catalog']['recipes'][0]
    session_id=created.json()['session_id']
    submitted=client.post(f'/api/agent/v2/sessions/{session_id}/experiments',headers=HEADERS,
        json={'recipe_id':recipe['recipe_id'],'recipe_digest':recipe['recipe_digest'],
              'catalog_digest':locked['preparation']['catalog']['catalog_digest'],
              'knowledge_refs':[],'client_request_id':'fail-second'})
    assert submitted.status_code==202,submitted.text
    run_id=submitted.json()['run_id']
    original=training.build_traditional_model
    calls=[0]
    def fail_second(*args,**kwargs):
        calls[0]+=1
        if calls[0]==2:
            raise ValueError('controlled candidate failure')
        return original(*args,**kwargs)
    monkeypatch.setattr(training,'build_traditional_model',fail_second)
    repo=RunRepository(storage/'runs.sqlite3');repo.initialize()
    worker=RunWorker(repository=repo,worker_id='fail-second',
        execute=lambda record:execute_claimed_run(record,repository=repo),
        now=lambda:datetime.now(timezone.utc),heartbeat_seconds=60)
    assert worker.run_once()
    assert repo.get(run_id).state=='failed'
    run_dir=storage/'runs'/run_id
    trials=json.loads((run_dir/'search_trials.json').read_text(encoding='utf-8'))
    summary=json.loads((run_dir/'search_summary.json').read_text(encoding='utf-8'))
    assert [row['state'] for row in trials]==['succeeded','failed']
    assert summary['integrity']=='incomplete'
    assert summary['completed_trials']==1 and summary['failed_trials']==1
    assert not (run_dir/'manifest.json').exists()


def test_deep_candidate_optimizer_values_match_ledger(api,monkeypatch):
    import torch
    client,storage,dataset_id=api
    created=client.post('/api/agent/v2/sessions',headers=HEADERS,json={
        'dataset_id':dataset_id,'selection_metric':'macro_f1',
        'allowed_models':['cnn1d'],'max_runs':1,'seed':42,
        'execution_profile':'train-evidence-recipes-v1',
        'protocol_revision':'agent-recipes-revision-v4','decision_mode':'recipe_id',
        'processing_mode':'fixed','search_mode':'bounded','max_trials':2,
        'model_configs':{'cnn1d':{'epochs':1,'batch_size':8}},
        'modules':['train_evidence','legal_recipes','bounded_hpo'],
        'context_policy':{'source_role':'benchmark','case_write':False,'evidence':True,'risks':True}})
    assert created.status_code==201,created.text
    locked=created.json()['locked_config']
    recipe=locked['preparation']['catalog']['recipes'][0]
    session_id=created.json()['session_id']
    submitted=client.post(f'/api/agent/v2/sessions/{session_id}/experiments',headers=HEADERS,
        json={'recipe_id':recipe['recipe_id'],'recipe_digest':recipe['recipe_digest'],
              'catalog_digest':locked['preparation']['catalog']['catalog_digest'],
              'knowledge_refs':[],'client_request_id':'deep-optimizer'})
    assert submitted.status_code==202,submitted.text
    original=torch.optim.AdamW
    observed=[]
    def capture(parameters,**kwargs):
        optimizer=original(parameters,**kwargs)
        observed.append((optimizer,kwargs['lr'],kwargs['weight_decay']))
        return optimizer
    monkeypatch.setattr(torch.optim,'AdamW',capture)
    repo=RunRepository(storage/'runs.sqlite3');repo.initialize()
    worker=RunWorker(repository=repo,worker_id='deep-optimizer',
        execute=lambda record:execute_claimed_run(record,repository=repo),
        now=lambda:datetime.now(timezone.utc),heartbeat_seconds=60)
    assert worker.run_once()
    run_id=submitted.json()['run_id']
    assert repo.get(run_id).state=='succeeded',repo.get(run_id).error_details
    trials=json.loads((storage/'runs'/run_id/'search_trials.json').read_text(encoding='utf-8'))
    assert len(observed)==len(trials)==2
    assert observed[0][0] is not observed[1][0]
    assert [(lr,decay) for _,lr,decay in observed]==[
        (row['params']['learning_rate'],row['params']['weight_decay']) for row in trials]

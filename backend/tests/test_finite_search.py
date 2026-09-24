"""The new search protocol fits actual candidates in one scoped Run."""
from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
import hashlib

import pytest
from fastapi.testclient import TestClient

from backend.tests.test_agent_model_sessions import api
from backend.tests.modeling_data_factory import write_grouped_classification_csv
from backend.tests.test_agent_end_to_end import _isolate_storage
from backend.tests.test_knowledge_retrieval import configure_fixture_rag
from backend.app.main import app
from backend.app.runs.repository import RunRepository
from backend.app.runs.worker import RunWorker
from backend.app.runs.execution import execute_claimed_run
from backend.app.runs.artifacts import RunArtifactWriter


HEADERS = {'X-AutoAI-Agent-Revision': 'agent-recipes-revision-v4'}


def low_feature_api(monkeypatch, tmp_path, feature_count):
    configure_fixture_rag(monkeypatch, tmp_path)
    storage, _, _ = _isolate_storage(monkeypatch, tmp_path)
    source = write_grouped_classification_csv(tmp_path / 'small.csv',
        groups_per_class=5, repeats=2, feature_count=feature_count)
    client = TestClient(app)
    with source.open('rb') as handle:
        uploaded = client.post('/api/datasets/upload',
            files={'file': ('small.csv', handle, 'text/csv')})
    assert uploaded.status_code == 200, uploaded.text
    return client, storage, uploaded.json()['dataset_id']


def search_session_request(dataset_id, models, **extra):
    return dict(dataset_id=dataset_id, selection_metric='macro_f1',
        allowed_models=models, max_runs=1, seed=42,
        execution_profile='train-evidence-recipes-v1',
        protocol_revision='agent-recipes-revision-v4', decision_mode='recipe_id',
        processing_mode='fixed', search_mode='fixed', max_trials=1,
        modules=['train_evidence', 'legal_recipes'],
        context_policy={'source_role':'benchmark','case_write':False,
                        'evidence':True,'risks':True}, **extra)


def test_dimension_adjusted_default_replays_same_session(monkeypatch, tmp_path):
    client, _, dataset_id = low_feature_api(monkeypatch, tmp_path, 2)
    body = search_session_request(dataset_id, ['pls_da'], client_request_id='small-pls')
    first = client.post('/api/agent/v2/sessions', headers=HEADERS, json=body)
    assert first.status_code == 201, first.text
    assert first.json()['locked_config']['capability_snapshot']['model_configs']['pls_da']['pls_components'] == 2
    replay = client.post('/api/agent/v2/sessions', headers=HEADERS, json=body)
    assert replay.status_code == 201, replay.text
    assert replay.json()['session_id'] == first.json()['session_id']
    assert replay.json()['idempotent_replay'] is True
    changed = client.post('/api/agent/v2/sessions', headers=HEADERS,
        json={**body, 'selection_metric':'balanced_accuracy'})
    assert changed.status_code == 409
    assert changed.json()['detail']['code'] == 'agent_idempotency_conflict'


def test_pre_fix_dimension_hash_replays_from_frozen_evidence(monkeypatch, tmp_path):
    from backend.app.agent.contracts import CreateAgentSessionRequestV2, V2
    from backend.app.agent.service import _hash_payload
    from backend.app.datasets.repository import DatasetRepository

    client, storage, dataset_id = low_feature_api(monkeypatch, tmp_path, 2)
    body = search_session_request(dataset_id, ['pls_da'], client_request_id='old-small-pls')
    first = client.post('/api/agent/v2/sessions', headers=HEADERS, json=body)
    assert first.status_code == 201, first.text
    session_id = first.json()['session_id']
    adjusted = first.json()['locked_config']['capability_snapshot']['model_configs']
    legacy_body = CreateAgentSessionRequestV2.model_validate(body).model_dump(mode='json')
    legacy_body.update(contract_version=V2, model_configs=adjusted)
    legacy_hash = _hash_payload(legacy_body)
    with sqlite3.connect(storage / 'agent.sqlite3') as db:
        stable_hash = db.execute('SELECT payload_hash FROM agent_sessions_v1 WHERE session_id=?',
                                 (session_id,)).fetchone()[0]
        assert legacy_hash != stable_hash
        db.execute('UPDATE agent_sessions_v1 SET payload_hash=? WHERE session_id=?',
                   (legacy_hash, session_id))

    def no_dataset_recheck(*_args, **_kwargs):
        raise AssertionError('replay must use frozen preparation')

    monkeypatch.setattr(DatasetRepository, 'verify_integrity', no_dataset_recheck)

    replay = client.post('/api/agent/v2/sessions', headers=HEADERS, json=body)
    assert replay.status_code == 201, replay.text
    assert replay.json()['session_id'] == session_id
    assert replay.json()['idempotent_replay'] is True
    changed = client.post('/api/agent/v2/sessions', headers=HEADERS,
                          json={**body, 'selection_metric': 'balanced_accuracy'})
    assert changed.status_code == 409
    assert changed.json()['detail']['code'] == 'agent_idempotency_conflict'


def test_inapplicable_default_pca_does_not_block_legal_pool(monkeypatch, tmp_path):
    client, _, dataset_id = low_feature_api(monkeypatch, tmp_path, 1)
    body = search_session_request(dataset_id, ['pca_lda','logistic_regression'])
    response = client.post('/api/agent/v2/sessions', headers=HEADERS, json=body)
    assert response.status_code == 201, response.text
    catalog = response.json()['locked_config']['preparation']['catalog']
    assert catalog['excluded_models']['pca_lda'] == 'pca_lda_insufficient_training_dimensions'
    assert {row['model_id'] for row in catalog['recipes']} == {'logistic_regression'}
    only_pca = client.post('/api/agent/v2/sessions', headers=HEADERS,
        json=search_session_request(dataset_id, ['pca_lda']))
    assert only_pca.status_code == 422
    explicit = client.post('/api/agent/v2/sessions', headers=HEADERS,
        json=search_session_request(dataset_id, ['pca_lda','logistic_regression'],
            model_configs={'pca_lda':{'pca_components':2}}))
    assert explicit.status_code == 422


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
    timeline = json.loads((run_dir/'search_timeline.json').read_text(encoding='utf-8'))
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
    spans=timeline['spans']
    for stage in ('model_preprocessing_selection','model_preprocessing_refit',
                  'trial_fit','trial_validation'):
        matched=[span for span in spans if span['stage']==stage]
        assert len(matched)==(expected if stage.startswith('trial_') else 1)
        assert all(span['status']=='succeeded' and span['duration_seconds'] is not None
                   for span in matched)
    assert next(span for span in spans if span['stage']=='upstream_spectral_preprocessing')['status']=='not_applicable'
    trial_spans={span['span_id'] for span in spans if span['stage']=='trial_train_validation'}
    assert all(span['parent_span_id'] in trial_spans for span in spans
               if span['stage'] in ('trial_fit','trial_validation'))
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
    timeline=json.loads((storage/'runs'/run_id/'search_timeline.json').read_text(encoding='utf-8'))
    assert len(observed)==len(trials)==2
    assert observed[0][0] is not observed[1][0]
    assert [(lr,decay) for _,lr,decay in observed]==[
        (row['params']['learning_rate'],row['params']['weight_decay']) for row in trials]
    assert all(row['actual_epochs']==1 and row['training_batches']>0 for row in trials)
    assert len([span for span in timeline['spans'] if span['stage']=='trial_fit'])==2
    assert len([span for span in timeline['spans'] if span['stage']=='trial_validation'])==2


def test_failed_deep_batch_preserves_completed_progress(api, monkeypatch):
    import torch
    client, storage, dataset_id=api
    body=search_session_request(dataset_id,['cnn1d'],
        model_configs={'cnn1d':{'epochs':1,'batch_size':8}})
    created=client.post('/api/agent/v2/sessions',headers=HEADERS,json=body)
    assert created.status_code==201,created.text
    locked=created.json()['locked_config']
    recipe=locked['preparation']['catalog']['recipes'][0]
    submitted=client.post(f"/api/agent/v2/sessions/{created.json()['session_id']}/experiments",
        headers=HEADERS,json={'recipe_id':recipe['recipe_id'],
            'recipe_digest':recipe['recipe_digest'],
            'catalog_digest':locked['preparation']['catalog']['catalog_digest'],
            'knowledge_refs':[],'client_request_id':'partial-batch'})
    assert submitted.status_code==202,submitted.text
    original=torch.optim.AdamW.step
    calls=[0]
    def fail_second(self,*args,**kwargs):
        calls[0]+=1
        if calls[0]==2:
            raise ValueError('controlled optimizer failure')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(torch.optim.AdamW,'step',fail_second)
    repo=RunRepository(storage/'runs.sqlite3');repo.initialize()
    worker=RunWorker(repository=repo,worker_id='partial-batch',
        execute=lambda record:execute_claimed_run(record,repository=repo),
        now=lambda:datetime.now(timezone.utc),heartbeat_seconds=60)
    assert worker.run_once()
    run_id=submitted.json()['run_id']
    assert repo.get(run_id).state=='failed'
    run_dir=storage/'runs'/run_id
    trial=json.loads((run_dir/'search_trials.json').read_text(encoding='utf-8'))[0]
    summary=json.loads((run_dir/'search_summary.json').read_text(encoding='utf-8'))
    timeline=json.loads((run_dir/'search_timeline.json').read_text(encoding='utf-8'))
    assert trial['state']=='failed' and trial['actual_epochs']==0
    assert trial['training_batches']==1
    assert summary['training_batches']==summary['training_batches_known']==1
    assert summary['actual_epochs']==0 and summary['integrity']=='incomplete'
    assert any(span['stage']=='trial_fit' and span['status']=='failed'
               for span in timeline['spans'])
    assert not (run_dir/'manifest.json').exists()


def test_unentered_deep_trial_progress_stays_unknown(tmp_path):
    from backend.app.search_accounting import SearchAccounting
    plan={'plan_digest':'a'*64,'mode':'bounded','selection_metric':'best_valid_loss',
          'effective_search':True}
    accounting=SearchAccounting(tmp_path,plan,'partial')
    with pytest.raises(ValueError):
        with accounting.trial(1,{'index':0,'params':{},'params_digest':'b'*64},
                              parent_span='search'):
            raise ValueError('trainer did not enter')
    summary=json.loads((tmp_path/'search_summary.json').read_text(encoding='utf-8'))
    assert summary['actual_epochs'] is None and summary['training_batches'] is None
    assert summary['actual_epochs_unknown_trials']==1
    assert summary['training_batches_unknown_trials']==1

from datetime import datetime,timezone
import json
from pathlib import Path
import pytest
from backend.tests.test_agent_model_sessions import api
from backend.tests.test_finite_search import HEADERS,search_session_request
from backend.tests.modeling_data_factory import write_grouped_classification_csv
from backend.app.diagnostic_evidence import paired_metrics


@pytest.mark.parametrize('model',['logistic_regression','random_forest','cnn1d','pca_mlp'])
@pytest.mark.parametrize('mode',['fixed','bounded'])
def test_actual_search_trial_audit_and_fit_counts(api,model,mode):
    from backend.app.runs.repository import RunRepository
    from backend.app.runs.worker import RunWorker
    from backend.app.runs.execution import execute_claimed_run
    client,storage,dataset=api
    body=search_session_request(dataset,[model],model_configs={model:{'epochs':2,'batch_size':8}} if model in ('cnn1d','pca_mlp') else {})
    body.update(search_mode=mode,max_trials=2 if mode=='bounded' else 1,
        modules=['train_evidence','legal_recipes']+(['bounded_hpo'] if mode=='bounded' else []))
    c=client.post('/api/agent/v2/sessions',headers=HEADERS,json=body)
    assert c.status_code==201,c.text
    catalog=c.json()['locked_config']['preparation']['catalog'];recipe=catalog['recipes'][0]
    response=client.post('/api/agent/v2/sessions/'+c.json()['session_id']+'/experiments',headers=HEADERS,
        json=dict(recipe_id=recipe['recipe_id'],recipe_digest=recipe['recipe_digest'],catalog_digest=catalog['catalog_digest'],knowledge_refs=[],client_request_id='audit-search'))
    assert response.status_code==202,response.text
    rid=response.json()['run_id'];repo=RunRepository(storage/'runs.sqlite3')
    worker=RunWorker(repository=repo,worker_id='audit-search',now=lambda:datetime.now(timezone.utc),execute=lambda run:execute_claimed_run(run,repository=repo))
    assert worker.run_once();assert repo.get(rid).state=='succeeded'
    docs={p.name:json.loads(p.read_text()) for p in (storage/'runs'/rid).glob('*.json')}
    assert paired_metrics(docs)['status']=='ready'
    audit=docs['training_validation_audit.json']['folds'][0];summary=docs['search_summary.json']
    assert audit['selected_trial_index']==summary['selected'][0]['trial_index']
    expected=2 if mode=='bounded' else 1
    assert summary['candidate_fit_count']==expected
    assert summary['final_refit_count']==(0 if model in ('cnn1d','pca_mlp') else 1)
    if model in ('cnn1d','pca_mlp'):assert 1<=audit['best_epoch']<=2
    if model=='random_forest':assert audit['selection_criterion']=='oob_balanced_accuracy'


def test_external_test_private_audit_does_not_use_test_metrics(tmp_path):
    from backend.app.training import _run_legacy_training
    source=write_grouped_classification_csv(tmp_path/'train.csv',groups_per_class=10,repeats=2,feature_count=8)
    test=write_grouped_classification_csv(tmp_path/'test.csv',groups_per_class=4,repeats=2,feature_count=8)
    out=tmp_path/'run'
    _run_legacy_training(source,dict(model_type='logistic_regression',test_data_path=str(test),split_train=8,split_valid=2,split_test=0,feature_selection_enabled=False),output_dir=out)
    docs={p.name:json.loads(p.read_text()) for p in out.glob('*.json')}
    assert docs['training_validation_audit.json']['evaluation_strategy']=='external_test_holdout'
    assert paired_metrics(docs)['status']=='ready'
    before=paired_metrics(docs);docs['metrics.json']['test']={'PRIVATE_TEST_CANARY':'injected'}
    assert paired_metrics(docs)==before


@pytest.mark.parametrize('state',['failed','cancelled','queued','running'])
def test_failure_projection_never_guesses_from_private_errors(state):
    from types import SimpleNamespace
    from backend.app.agent.diagnosis import feedback_evidence
    record=SimpleNamespace(state=state,run_id='run-1',error_details=dict(code='dependency_unavailable',message='PRIVATE_CANARY /users/private/data.csv',type='ModuleNotFoundError'))
    evidence=feedback_evidence(record=record,assessment=None,awareness='off')
    assert evidence['metric_pairs']==[]
    assert evidence['failure']['reason_code']==({'failed':'training_failure','cancelled':'cancelled'}.get(state))
    assert 'PRIVATE_CANARY' not in json.dumps(evidence)
    assert 'ModuleNotFoundError' not in json.dumps(evidence)


def test_early_stopped_deep_audit_uses_restored_best_epoch(tmp_path,monkeypatch):
    from backend.app import training
    source=write_grouped_classification_csv(tmp_path/'early-stop.csv',groups_per_class=10,repeats=2,feature_count=8)
    losses=[]
    def plateau(*args,**kwargs):
        losses.append(0.5)
        return 0.5
    monkeypatch.setattr(training,'_evaluate_deep_loss',plateau)
    out=tmp_path/'run'
    training._run_legacy_training(source,dict(model_type='cnn1d',epochs=4,batch_size=8,
        early_stopping_patience=1,feature_selection_enabled=False),output_dir=out)
    docs={p.name:json.loads(p.read_text()) for p in out.glob('*.json')}
    assert len(losses)==2
    import csv
    with (out/'history.csv').open(encoding='utf-8-sig') as history:
        assert len(list(csv.DictReader(history)))==2
    assert docs['training_validation_audit.json']['folds'][0]['best_epoch']==1
    assert paired_metrics(docs)['status']=='ready'


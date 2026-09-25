"""Real fits plus isolated failure injection; no synthetic success claims."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import sqlite3

import pytest

from backend.tests.test_agent_model_sessions import api
from backend.tests.test_step8_guard_integration import submit, HEADERS
from backend.app.runs.repository import RunRepository
from backend.app.runs.worker import RunWorker, execute_with_budget_supervision


def worker(storage, execute=None):
    repo = RunRepository(storage/'runs.sqlite3')
    return repo, RunWorker(repository=repo, worker_id='guard-fault', now=lambda:datetime.now(timezone.utc),
        execute=execute or (lambda run:execute_with_budget_supervision(run, repository=repo, agent_database=storage/'agent.sqlite3')))


@pytest.mark.parametrize('fault', ['config','metric','split','processing','selected','missing','usage','report_io','claim'])
def test_postfit_fault_never_refunds_or_publishes(api, monkeypatch, fault):
    from backend.app.runs.artifacts import RunArtifactWriter
    from backend.app.runs import guard_store
    client, storage, _ = api
    base, run_id, _ = submit(api)
    repo = RunRepository(storage/'runs.sqlite3')
    def execute(record):
        result = execute_with_budget_supervision(record, repository=repo, agent_database=storage/'agent.sqlite3')
        root = storage/'runs'/run_id
        names = {'config':'config.json','metric':'metrics.json','split':'split.json',
            'processing':'model_metadata.json','selected':'search_summary.json'}
        if fault in names:
            path = root/names[fault]; body = json.loads(path.read_bytes())
            if fault == 'config': body['normalization'] = 'none'
            elif fault == 'metric': body['valid']['macro_f1'] = True
            elif fault == 'split': body[0]['splits']['train'] = []
            elif fault == 'processing': body['execution_audit']['processing_execution']['folds'] = []
            elif fault == 'selected': body['selected'][0]['params']['logistic_c'] = 999
            path.write_text(json.dumps(body), encoding='utf-8')
            RunArtifactWriter(root).finalize(run_id=run_id)
        elif fault == 'missing': (root/'model.pkl').unlink()
        elif fault == 'usage': result['_guard_usage']['model_fits'] += 1
        elif fault == 'report_io':
            def fail(*a, **kw): raise OSError('PRIVATE_CANARY_REPORT_IO')
            monkeypatch.setattr(guard_store, 'insert', fail)
        elif fault == 'claim': repo.request_cancel(run_id, now=datetime.now(timezone.utc))
        return result
    _, runner = worker(storage, execute)
    assert runner.run_once()
    assert repo.get(run_id).state != 'succeeded'
    assert not runner.run_once()  # Restart cannot refit the terminal Run.
    session = client.get(base, headers=HEADERS).json()
    assert session['best_run_id'] is None
    with sqlite3.connect(storage/'agent.sqlite3') as db:
        assert dict(db.execute("SELECT dimension,actual FROM task_budget_reservations_v1 WHERE dimension IN ('experiments','model_fits')")) == {'experiments':1,'model_fits':2}
        assert db.execute('SELECT state FROM task_training_executions_v1').fetchone()[0] == 'settled'


def test_finalize_repeat_delete_and_collector_share_current_eligibility(api):
    from scripts.agent_ablation import terminal_results
    client, storage, _ = api
    base, run_id, _ = submit(api)
    repo, runner = worker(storage); assert runner.run_once()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _:client.post(base+'/finalize', headers=HEADERS,json={'selected_run_id':run_id}), range(2)))
    assert [r.status_code for r in results] == [200,200]
    record = {'run_id':run_id, 'session_id':base.rsplit('/',1)[1], 'versions':{'protocol_revision':'agent-recipes-revision-v6'}}
    terminal_results(client, record)
    assert record['offline_test']
    original = (storage/'runs'/run_id/'model.pkl').read_bytes()
    (storage/'runs'/run_id/'model.pkl').write_bytes(b'!'+original[1:])
    with pytest.raises(ValueError, match='currently unavailable'): terminal_results(client, record)
    assert client.get(base, headers=HEADERS).json()['selected_run_id'] == run_id


def test_finalize_and_delete_are_serialized(api, monkeypatch):
    from threading import Event
    from backend.app.agent import service
    client, storage, _ = api
    base, run_id, _ = submit(api)
    _, runner = worker(storage); assert runner.run_once()
    entered, release = Event(), Event()
    original = service.assess_candidate
    def paused(**kwargs):
        result = original(**kwargs)
        if kwargs.get('stage') == 'finalize':
            entered.set(); assert release.wait(10)
        return result
    monkeypatch.setattr(service, 'assess_candidate', paused)
    with ThreadPoolExecutor(max_workers=2) as pool:
        final = pool.submit(client.post, base+'/finalize', headers=HEADERS, json={'selected_run_id':run_id})
        assert entered.wait(10)
        deletion = pool.submit(client.delete, '/api/training/runs/'+run_id)
        assert not deletion.done()
        release.set()
        assert final.result().status_code == 200
        assert deletion.result().status_code == 200
    current = client.get(base, headers=HEADERS).json()
    assert current['best_run_id'] is None and current['selected_run_id'] == run_id


@pytest.mark.parametrize('strategy', ['stratified_holdout','leave_one_sample_id_cv','external_test_holdout'])
def test_manual_training_uses_guard_without_agent_binding(api, strategy):
    from backend.app.runs.execution import execute_claimed_run
    client, storage, dataset = api
    config = {'model_type':'logistic_regression','split_mode':strategy,'feature_selection_enabled':False}
    body = {'dataset_id':dataset,'config':config}
    if strategy == 'external_test_holdout':
        from backend.tests.modeling_data_factory import write_grouped_classification_csv
        source = write_grouped_classification_csv(storage/'external.csv',groups_per_class=3,repeats=1,feature_count=128)
        with source.open('rb') as handle:
            response = client.post('/api/datasets/upload',files={'file':('external.csv',handle,'text/csv')})
        body['test_dataset_id'] = response.json()['dataset_id']
    response = client.post('/api/training/runs',json=body)
    assert response.status_code == 202, response.text
    run_id = response.json()['run_id']
    repo = RunRepository(storage/'runs.sqlite3')
    _, runner = worker(storage, lambda r:execute_claimed_run(r, repository=repo))
    assert runner.run_once()
    record = repo.get(run_id)
    assert record.state == 'succeeded', record.error_details
    assert record.publication_report_id and not record.config.get('execution_budget_task_id')
    cv = json.loads((storage/'runs'/run_id/'cv_metrics.json').read_bytes())
    assert cv['strategy'] == strategy

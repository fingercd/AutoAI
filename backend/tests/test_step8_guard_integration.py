from datetime import datetime, timezone
import json
import sqlite3

import pytest

from backend.tests.test_agent_model_sessions import api
from backend.tests.test_step7_v5_session import _request
from backend.app.runs.repository import RunRepository
from backend.app.runs.worker import RunWorker, execute_with_budget_supervision

HEADERS = {'X-AutoAI-Agent-Revision':'agent-recipes-revision-v6'}


def submit(api, *, model='logistic_regression', mode='fixed', guard='on'):
    client, storage, dataset = api
    body = _request(dataset)
    body.update(protocol_revision='agent-recipes-revision-v6', allowed_models=[model],
                fail_fast_guard=guard, search_mode=mode, max_trials=2 if mode=='bounded' else 1)
    body['budget_policy']['dimensions']['model_fits']['limit'] = 4
    body['budget_policy']['dimensions']['training_epochs']['limit'] = 4
    if model == 'cnn1d': body['model_configs'] = {model:{'epochs':2}}
    if guard == 'on': body['modules'].append('fail_fast_guard')
    if mode == 'bounded': body['modules'].append('bounded_hpo')
    created = client.post('/api/agent/v2/sessions', headers=HEADERS, json=body)
    assert created.status_code == 201, created.text
    session = created.json()
    catalog = session['locked_config']['preparation']['catalog']
    recipe = catalog['recipes'][0]
    base = f"/api/agent/v2/sessions/{session['session_id']}"
    payload = dict(recipe_id=recipe['recipe_id'], recipe_digest=recipe['recipe_digest'],
        catalog_digest=catalog['catalog_digest'], knowledge_refs=[], client_request_id='guard-run')
    queued = client.post(base+'/experiments', headers=HEADERS, json=payload)
    assert queued.status_code == 202, queued.text
    return base, queued.json()['run_id'], body


@pytest.mark.parametrize('model,mode,guard', [('logistic_regression','fixed','on'),('cnn1d','bounded','off')])
def test_real_supervised_guard_publication_and_damage(api, model, mode, guard):
    client, storage, _ = api
    base, run_id, _ = submit(api, model=model, mode=mode, guard=guard)
    repo = RunRepository(storage/'runs.sqlite3')
    worker = RunWorker(repository=repo, worker_id='guard-worker', now=lambda:datetime.now(timezone.utc),
        execute=lambda record:execute_with_budget_supervision(record, repository=repo, agent_database=storage/'agent.sqlite3'))
    assert worker.run_once()
    record = repo.get(run_id)
    assert record.state == 'succeeded', record.error_details
    feedback = client.get(base+f'/experiments/{run_id}/feedback', headers=HEADERS)
    assert feedback.status_code == 200, feedback.text
    assert feedback.json()['extensions']['guard']['eligibility'] == 'eligible'
    assert client.get(base, headers=HEADERS).json()['best_run_id'] == run_id
    finalized = client.post(base+'/finalize', headers=HEADERS, json={'selected_run_id':run_id})
    assert finalized.status_code == 200, finalized.text
    before = (storage/'runs'/run_id/'model_metadata.json').read_bytes()
    (storage/'runs'/run_id/'model_metadata.json').write_bytes(b'!'+before[1:])
    invalid = client.get(base+f'/experiments/{run_id}/feedback', headers=HEADERS).json()
    assert invalid['validation']['status'] == 'unavailable'
    assert invalid['extensions']['guard']['eligibility'] == 'ineligible'
    session = client.get(base, headers=HEADERS).json()
    assert session['best_run_id'] is None and session['selected_run_id'] == run_id
    assert client.post(base+'/finalize', headers=HEADERS, json={'selected_run_id':run_id}).status_code == 422
    assert repo.get(run_id).state == 'succeeded'
    with sqlite3.connect(storage/'agent.sqlite3') as db:
        assert db.execute('SELECT state FROM task_training_executions_v1').fetchone()[0] == 'settled'


def test_guard_switch_is_frozen_and_old_worker_cannot_claim(api):
    client, storage, _ = api
    base, run_id, body = submit(api)
    changed = {**body, 'fail_fast_guard':'off', 'modules':['train_evidence','legal_recipes']}
    assert client.post('/api/agent/v2/sessions', headers=HEADERS, json=changed).status_code == 409
    repo = RunRepository(storage/'runs.sqlite3')
    now = datetime.now(timezone.utc)
    repo.record_worker_heartbeat(worker_id='old', now=now, contract_version='training-worker-budget-v1')
    with pytest.raises(sqlite3.IntegrityError): repo.claim_next(worker_id='old', now=now)
    assert repo.get(run_id).state == 'queued'


@pytest.mark.parametrize('fault', ['manifest_return', 'settled_return', 'unknown_exit'])
def test_real_child_completion_cannot_publish_across_parent_failure(api, monkeypatch, fault):
    from backend.app.runs import supervisor
    from backend.app.runs.supervisor import ChildExecutionError, SupervisionUncertain, TerminationEvidence
    client, storage, _ = api
    base, run_id, _ = submit(api)
    repo = RunRepository(storage/'runs.sqlite3')
    original = supervisor.supervise
    if fault != 'settled_return':
        def interrupted(*args, **kwargs):
            original(*args, **kwargs)  # Actual fit and actual process-tree exit.
            if fault == 'unknown_exit':
                # Inject lost exit evidence, not a claim that a live tree was observed.
                raise SupervisionUncertain(TerminationEvidence('lost_evidence', datetime.now(timezone.utc).isoformat(), None, None, None))
            raise ChildExecutionError('InjectedParentLoss', 'Manifest return lost')
        monkeypatch.setattr(supervisor, 'supervise', interrupted)
    def execute(record):
        result = execute_with_budget_supervision(record, repository=repo, agent_database=storage/'agent.sqlite3')
        if fault == 'settled_return':
            raise RuntimeError('Injected crash after settlement')
        return result
    worker = RunWorker(repository=repo, worker_id='fault-worker', now=lambda:datetime.now(timezone.utc), execute=execute)
    if fault == 'unknown_exit':
        with pytest.raises(SupervisionUncertain): worker.run_once()
    else:
        assert worker.run_once()
    assert repo.get(run_id).state != 'succeeded'
    assert client.get(base, headers=HEADERS).json()['best_run_id'] is None
    with sqlite3.connect(storage/'agent.sqlite3') as db:
        state = db.execute('SELECT state FROM task_training_executions_v1').fetchone()[0]
        assert state == ('unknown_pending' if fault=='unknown_exit' else 'settled')
        actual = dict(db.execute("SELECT dimension,actual FROM task_budget_reservations_v1 WHERE dimension IN ('model_fits','experiments')"))
        assert actual == {'model_fits':2, 'experiments':1}
    # Manifest alone never grants candidate status.
    assert (storage/'runs'/run_id/'manifest.json').exists()


def test_queued_dataset_replacement_zero_fit_retains_attempt(api):
    from backend.app.datasets.repository import DatasetRepository
    client, storage, dataset = api
    base, run_id, _ = submit(api, guard='off')
    data = DatasetRepository(storage/'datasets.sqlite3', storage_root=storage).resolve_system(dataset, legacy_path=None)
    data.path.write_bytes(data.path.read_bytes()+b'\n')
    repo = RunRepository(storage/'runs.sqlite3')
    worker = RunWorker(repository=repo, worker_id='changed-worker', now=lambda:datetime.now(timezone.utc),
        execute=lambda record:execute_with_budget_supervision(record, repository=repo, agent_database=storage/'agent.sqlite3'))
    assert worker.run_once()
    assert repo.get(run_id).state == 'failed'
    details=repo.get(run_id).error_details
    assert details['code']=='guard_dataset_changed' and details['stage']=='pre_fit'
    assert details['report_id'].startswith('guard-')
    feedback=client.get(base+f'/experiments/{run_id}/feedback',headers=HEADERS).json()
    assert feedback['extensions']['guard']['report_id']==details['report_id']
    assert str(data.path) not in json.dumps(feedback)
    with sqlite3.connect(storage/'agent.sqlite3') as db:
        assert db.execute('SELECT count(*) FROM task_training_events_v1').fetchone()[0] == 0
        amounts = dict(db.execute("SELECT dimension,actual FROM task_budget_reservations_v1 WHERE dimension IN ('model_fits','experiments')"))
        assert amounts == {'model_fits':0,'experiments':1}
    assert client.get(base, headers=HEADERS).json()['remaining_runs'] == 0

"""Independent acceptance probes: expected invariants, not implementation snapshots.

All stores/data are pytest-created. No production artifact is modified.
"""
import json
import sqlite3
from datetime import datetime, timezone

import pytest

from backend.tests.test_agent_model_sessions import api
from agent_poc.tests.conftest import budget_config
from backend.tests.test_step8_guard_integration import submit, HEADERS
from backend.app.runs.repository import RunRepository
from backend.app.runs.worker import RunWorker, execute_with_budget_supervision


def runner(storage, execute=None):
    repo = RunRepository(storage / 'runs.sqlite3')
    return repo, RunWorker(repository=repo, worker_id='independent-probe',
        now=lambda: datetime.now(timezone.utc), execute=execute or (
            lambda r: execute_with_budget_supervision(r, repository=repo,
                agent_database=storage / 'agent.sqlite3')))


@pytest.mark.parametrize('fault', ['missing_best_params', 'test_as_final_fit',
    'final_missing', 'final_duplicate', 'final_outside', 'final_bool', 'final_scope',
    'stages_missing', 'stage_extra', 'stage_type', 'stage_scope', 'stage_count', 'stage_indices',
    'best_empty', 'best_partial', 'best_extra', 'best_changed'])
def test_incomplete_or_wrong_execution_evidence_cannot_publish(api, fault):
    from backend.app.runs.artifacts import RunArtifactWriter
    from backend.app.runs.guard import check_outputs
    client, storage, _ = api
    base, run_id, _ = submit(api, model='svm' if fault.startswith('best_') else 'logistic_regression')
    repo = RunRepository(storage / 'runs.sqlite3')

    def execute(record):
        result = execute_with_budget_supervision(record, repository=repo,
            agent_database=storage / 'agent.sqlite3')
        root = storage / 'runs' / run_id
        folds = json.loads((root / 'split.json').read_bytes())
        cv = json.loads((root / 'cv_metrics.json').read_bytes())
        metadata = json.loads((root / 'model_metadata.json').read_bytes())
        fold = folds[0]
        if fault == 'missing_best_params':
            assert folds[0]['best_params']
            folds[0].pop('best_params')
        elif fault == 'test_as_final_fit':
            assert set(folds[0]['final_fit_indices']).isdisjoint(folds[0]['splits']['test'])
            folds[0]['final_fit_indices'] = folds[0]['splits']['test']
        elif fault == 'final_missing': fold['final_fit_indices'].pop()
        elif fault == 'final_duplicate': fold['final_fit_indices'].append(fold['final_fit_indices'][0])
        elif fault == 'final_outside': fold['final_fit_indices'][-1] = 999999
        elif fault == 'final_bool': fold['final_fit_indices'][0] = True
        elif fault == 'final_scope': metadata['execution_audit']['final_fit_scope'] = 'test'
        elif fault.startswith('stage'):
            stages = fold['processing_execution']['stages']
            if fault == 'stages_missing': stages.clear()
            elif fault == 'stage_extra': stages.append(dict(stages[0]))
            elif fault == 'stage_type': stages[0]['stage'] = 'test'
            elif fault == 'stage_scope': stages[0]['fit_scope'] = 'test'
            elif fault == 'stage_count': stages[0]['fit_count'] -= 1
            elif fault == 'stage_indices': stages[0]['fit_indices_digest'] = '0' * 64
            processing = metadata['execution_audit']['processing_execution']
            processing['folds'][0]['stages'] = [{k:s[k] for k in (
                'stage','fit_scope','fit_indices_digest','normalizer_digest','weighting_digest')} for s in stages]
            from backend.app.model_config import semantic_digest
            processing['digest'] = semantic_digest({k:v for k,v in processing.items() if k != 'digest'})
        elif fault == 'best_empty': fold['best_params'] = {}
        elif fault == 'best_partial': fold['best_params'] = {'svm_c': fold['best_params']['svm_c']}
        elif fault == 'best_extra': fold['best_params']['unknown'] = 1
        elif fault == 'best_changed': fold['best_params']['svm_c'] = 999
        cv['folds'] = folds
        for name, body in [('split.json', folds), ('cv_metrics.json', cv), ('model_metadata.json', metadata)]:
            (root / name).write_text(json.dumps(body), encoding='utf-8')
        # Model a faulty producer's final output, not a byte/hash mismatch.
        # This is the same real check_outputs entry used by child publication.
        RunArtifactWriter(root).finalize(run_id=run_id)
        checked = check_outputs(record=record, repository=repo, run_dir=root,
            stage='pre_publish', usage=result['_guard_usage'])
        result['guard_report_id'] = checked.report_id
        return result

    _, worker = runner(storage, execute)
    assert worker.run_once()
    record = repo.get(run_id)
    feedback = client.get(base + f'/experiments/{run_id}/feedback', headers=HEADERS).json()
    finalized = client.post(base + '/finalize', headers=HEADERS,
        json={'selected_run_id': run_id})
    print(json.dumps({'probe': fault, 'state': record.state,
        'guard': feedback.get('extensions', {}).get('guard', {}).get('eligibility'),
        'finalize_http': finalized.status_code}))
    assert record.state != 'succeeded', 'Semantically invalid execution evidence was published and selectable'
    assert record.error_details['code'] == 'guard_execution_audit_mismatch'
    assert finalized.status_code != 200
    assert feedback['validation']['status'] != 'ready'
    assert not worker.run_once()
    with sqlite3.connect(storage / 'agent.sqlite3') as db:
        assert db.execute('SELECT state FROM task_training_executions_v1').fetchone()[0] == 'settled'
        assert dict(db.execute("SELECT dimension,actual FROM task_budget_reservations_v1 WHERE dimension IN ('experiments','model_fits')")) == {'experiments':1,'model_fits':2}


def test_rules_source_drift_preserves_readonly_finalized_session(api, monkeypatch):
    from backend.app.runs import guard
    client, storage, _ = api
    base, run_id, _ = submit(api)
    repo, worker = runner(storage)
    assert worker.run_once() and repo.get(run_id).state == 'succeeded'
    assert client.post(base + '/finalize', headers=HEADERS,
        json={'selected_run_id': run_id}).status_code == 200
    original_run = repo.get(run_id)
    from backend.app.runs.contracts import Principal
    publication = repo.get_guard_report(original_run.publication_report_id, principal=Principal())
    # Equivalent to a later code release computing a different source digest.
    # Persisted policy, reports, Run, Manifest and files are left unchanged.
    monkeypatch.setattr(guard, 'RULES_DIGEST', 'f' * 64)
    try:
        response = client.get(base, headers=HEADERS)
    except Exception as exc:
        print(json.dumps({'probe': 'rules_drift', 'exception_type': type(exc).__name__,
            'run_state': repo.get(run_id).state}))
        pytest.fail('Read-only finalized Session raises ' + type(exc).__name__ + ' after rules source drift')
    print(json.dumps({'probe': 'rules_drift', 'session_get_http': response.status_code,
        'run_state': repo.get(run_id).state}))
    assert response.status_code == 200
    assert response.json()['selected_run_id'] == run_id
    assert response.json()['best_run_id'] is None
    feedback = client.get(base + f'/experiments/{run_id}/feedback', headers=HEADERS).json()
    assert feedback['validation']['status'] == 'unavailable'
    assert feedback['extensions']['guard']['status'] == 'unavailable'
    assert feedback['extensions']['guard']['eligibility'] == 'unavailable'
    assert repo.get(run_id).guard_policy == original_run.guard_policy
    assert repo.get_guard_report(original_run.publication_report_id, principal=Principal()) == publication


def test_deterministic_admission_failure_closes_session(api, tmp_path, budget_config, monkeypatch):
    from backend.app import training
    from backend.app.runs.guard import GuardError
    from agent_poc.clients.autoai_client import AutoAIClient
    from agent_poc.orchestration.llm import LLMAdapter, LLMConfig
    from agent_poc.orchestration.runtime import RuntimeConfig, start_task
    from agent_poc.tests.test_knowledge_graph import KnowledgeProvider
    from agent_poc.tests.test_review_recovery import CountTransport

    def fail_preflight(*args, **kwargs):
        assert kwargs['record'].run_id is None
        raise GuardError('guard_resource_limit', stage='admission')

    monkeypatch.setattr(training, 'prepare_training_inputs', fail_preflight)
    wire, storage, dataset = api
    config = LLMConfig('http://scripted.invalid/v1', 'fixture',
        prompt_version='agent-decision-budget-v1', **budget_config)
    runtime = RuntimeConfig('http://backend.invalid', 'local', config)
    transport = CountTransport(wire)
    client = AutoAIClient(runtime.backend_url, transport=transport, api_version='v2',
        execution_profile='train-evidence-recipes-v1', protocol_revision='agent-recipes-revision-v6',
        processing_mode='fixed', search_mode='fixed', max_trials=1, max_retries=0)
    provider = KnowledgeProvider('json_action')
    state = start_task(runtime, dataset_id=dataset, allowed_models=['logistic_regression'],
        processing_mode='fixed', search_mode='fixed', max_trials=1, decision_mode='recipe_id',
        knowledge=False, budget_awareness='off', fail_fast_guard='on', storage=tmp_path/'graph',
        thread_id='admission-rejection', client=client, llm=LLMAdapter(config, transport=provider), wait=True)
    session_id = state['identity']['session_id']
    response = wire.get('/api/agent/v2/sessions/' + session_id, headers=HEADERS)
    assert response.status_code == 200
    backend = response.json()
    print(json.dumps({'probe': 'admission_rejection', 'lifecycle': state['lifecycle'],
        'backend_state': backend['state'], 'runs': len(RunRepository(storage/'runs.sqlite3').list()),
        'llm_calls': len(provider.contexts)}))
    assert backend['state'] == 'terminated', 'Known no-Run Guard rejection left backend Session open'
    assert state['lifecycle']['status'] == 'completed'
    assert len(RunRepository(storage/'runs.sqlite3').list()) == 0
    assert len(provider.contexts) == 1
    assert state['guard']['reports'][0]['checks'][0]['reason_code'] == 'guard_resource_limit'
    assert state['guard']['reports'][0]['report_id'].startswith('guard-')
    assert state['budget']['llm_calls']['actual'] == 1
    assert state['budget']['api_calls']['actual'] == len(transport.requests)
    from agent_poc.orchestration.runtime import resume_task
    before = len(transport.requests)
    again = resume_task(runtime, storage=tmp_path/'graph', thread_id='admission-rejection',
                        client=client, llm=LLMAdapter(config, transport=provider), wait=True)
    assert again == state and len(transport.requests) == before and len(provider.contexts) == 1
    with sqlite3.connect(storage/'agent.sqlite3') as db:
        assert db.execute('SELECT COUNT(*) FROM task_training_executions_v1').fetchone()[0] == 0


def test_queued_old_policy_cannot_enter_child_or_fit(api, monkeypatch):
    from backend.app.runs import guard, supervisor
    client, storage, _ = api
    base, run_id, _ = submit(api)
    monkeypatch.setattr(guard, 'RULES_DIGEST', 'f' * 64)
    def forbidden(*args, **kwargs):
        pytest.fail('old policy entered child execution')
    monkeypatch.setattr(supervisor, 'supervise', forbidden)
    repo, worker = runner(storage)
    assert worker.run_once()
    record = repo.get(run_id)
    assert record.state == 'failed' and record.error_details['code'] == 'guard_check_unavailable'
    assert client.get(base, headers=HEADERS).status_code == 200
    with sqlite3.connect(storage/'agent.sqlite3') as db:
        assert db.execute('SELECT state FROM task_training_executions_v1').fetchone()[0] == 'settled'
        assert db.execute("SELECT actual FROM task_budget_reservations_v1 WHERE dimension='model_fits'").fetchone()[0] == 0


@pytest.mark.parametrize('corruption', ['digest', 'version'])
def test_corrupt_historical_policy_has_safe_error(api, corruption):
    client, storage, _ = api
    base, _, _ = submit(api)
    with sqlite3.connect(storage/'agent.sqlite3') as db:
        row = db.execute('SELECT frozen_preparation_json FROM agent_sessions_v1').fetchone()
        frozen = json.loads(row[0])
        if corruption == 'digest': frozen['guard_policy_digest'] = '0' * 64
        else: frozen['guard_policy']['rules_version'] = 'PRIVATE_UNKNOWN_VERSION'
        db.execute('UPDATE agent_sessions_v1 SET frozen_preparation_json=?', (json.dumps(frozen),))
    response = client.get(base, headers=HEADERS)
    assert response.status_code == 409
    assert response.json()['detail']['code'] == 'guard_report_binding_mismatch'
    assert 'PRIVATE_UNKNOWN_VERSION' not in response.text

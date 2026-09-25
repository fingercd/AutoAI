"""The exporter reads legacy WAL state without migrating or exposing proposals."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest
from backend.tests.test_agent_model_sessions import api

from scripts.task_cost_report import build_report, export_report


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources(tmp_path: Path):
    agent = tmp_path / 'agent.sqlite'
    db = sqlite3.connect(agent)
    db.execute('''CREATE TABLE agent_sessions_v1
        (session_id TEXT,owner_id TEXT,tenant_id TEXT)''')
    db.execute('''INSERT INTO agent_sessions_v1 VALUES('session-1',NULL,NULL)''')
    db.commit()
    db.close()
    journal = tmp_path / 'calls.sqlite'
    live = sqlite3.connect(journal)
    live.execute('PRAGMA journal_mode=WAL')
    live.execute('''CREATE TABLE orchestration_calls_v1
        (id INTEGER,thread_id TEXT,operation_id TEXT,kind TEXT,name TEXT,
        status TEXT,input_tokens INTEGER,output_tokens INTEGER,total_tokens INTEGER,
        proposal_json TEXT,task_id TEXT,session_id TEXT,run_id TEXT)''')
    live.execute('''INSERT INTO orchestration_calls_v1 VALUES
        (1,'thread-1','op-1','llm','choose','confirmed',10,NULL,10,
         'private prompt 123','task-1','session-1',NULL)''')
    live.execute('''INSERT INTO orchestration_calls_v1 VALUES
        (2,'thread-1','op-2','llm','finalize','failed',20,5,25,
         'private prompt 456','task-1','session-1',NULL)''')
    live.commit()
    return agent, journal, live


def test_readonly_wal_repeat_export_and_metric_coverage(tmp_path):
    agent, journal, live = sources(tmp_path)
    before = {path.name: digest(path) for path in (agent, journal, journal.with_name('calls.sqlite-wal'))}
    options = dict(journal=journal, agent_db=agent, thread_id='thread-1',
                   session_id='session-1', task_id='task-1',
                   owner_id=None, tenant_id=None)
    one, two = tmp_path / 'report-1', tmp_path / 'report-2'
    export_report(one, **options)
    export_report(two, **options)
    assert sorted(path.name for path in one.iterdir()) == sorted(path.name for path in two.iterdir())
    assert (one / 'coverage.json').read_bytes() == (two / 'coverage.json').read_bytes()
    coverage = json.loads((one / 'coverage.json').read_text(encoding='utf-8'))
    assert coverage['input_tokens']['actual_total'] == 30
    assert coverage['output_tokens']['known_subtotal'] == 5
    assert coverage['output_tokens']['actual_total'] is None
    assert coverage['output_tokens']['unknown_count'] == 1
    rows = list(csv.DictReader((one / 'llm_requests.csv').open(encoding='utf-8')))
    assert all(row['dispatch_evidence'] == 'unknown' for row in rows)
    assert all(row['cached_tokens'] == '' for row in rows)
    assert 'private prompt' not in ''.join(path.read_text(encoding='utf-8') for path in one.iterdir())
    assert before == {path.name: digest(path) for path in (agent, journal, journal.with_name('calls.sqlite-wal'))}
    live.close()


def test_scope_mismatch_does_not_emit_report(tmp_path):
    agent, journal, live = sources(tmp_path)
    with pytest.raises(ValueError, match='cost_report_scope_mismatch'):
        build_report(journal=journal, agent_db=agent, thread_id='thread-1',
                     session_id='session-1', task_id='task-1',
                     owner_id='another', tenant_id=None)
    live.close()


def test_training_scope_refit_and_parent_spans_remain_separate(tmp_path):
    agent, journal, live = sources(tmp_path)
    db = sqlite3.connect(agent)
    db.execute('''CREATE TABLE agent_experiment_reservations_v1
        (session_id TEXT,run_id TEXT,owner_id TEXT,tenant_id TEXT)''')
    db.execute("INSERT INTO agent_experiment_reservations_v1 VALUES('session-1','run-1',NULL,NULL)")
    db.commit()
    db.close()
    run_dir, runs_db, plan = bound_run(tmp_path)
    (run_dir / 'search_summary.json').write_text(json.dumps(dict(
        schema_version='search-summary-v1', run_id='run-1', plan_digest=plan['plan_digest'],
        integrity='complete', final_refit_count=1,
        selection_metric='oob_balanced_accuracy_then_accuracy')), encoding='utf-8')
    (run_dir / 'search_trials.json').write_text(json.dumps([dict(
        run_id='run-1', fold_index=0, trial_index=0, state='succeeded', actual_epochs=0,
        training_batches=0, started_at='2026-09-25T00:00:01+00:00',
        ended_at='2026-09-25T00:00:04+00:00', duration_seconds=3.0)]), encoding='utf-8')
    (run_dir / 'search_timeline.json').write_text(json.dumps(dict(
        schema_version='search-timeline-v1', run_id='run-1', spans=[
        dict(span_id='parent', run_id='run-1', parent_span_id=None, stage='search', status='succeeded',
             started_at='2026-09-25T00:00:00+00:00', ended_at='2026-09-25T00:00:05+00:00',
             duration_seconds=5.0, duration_clock='process_monotonic'),
        dict(span_id='child', run_id='run-1', parent_span_id='parent', stage='trial_fit', status='succeeded',
             started_at='2026-09-25T00:00:01+00:00', ended_at='2026-09-25T00:00:04+00:00',
             duration_seconds=3.0, duration_clock='process_monotonic')])), encoding='utf-8')
    files = build_report(journal=journal, agent_db=agent, thread_id='thread-1',
        session_id='session-1', task_id='task-1', owner_id=None, tenant_id=None,
        run_id='run-1', run_dir=run_dir, runs_db=runs_db)
    usage = list(csv.DictReader(files['training_usage.csv'].splitlines()))
    assert [row['fit_purpose'] for row in usage] == ['candidate', 'final_refit']
    assert usage[0]['measurement_scope'] == 'train_oob'
    events = list(csv.DictReader(files['timeline_events.csv'].splitlines()))
    assert events[1]['parent_event_id'] == 'parent'
    assert 'sum_duration_seconds' not in json.loads(files['cost_summary.json'])
    live.close()


def bound_run(tmp_path: Path):
    plan = {'schema_version': 'search-plan-v1', 'plan_digest': 'plan-1'}
    run_dir = tmp_path / 'runs' / 'run-1'
    run_dir.mkdir(parents=True)
    (run_dir / 'search_plan.json').write_text(json.dumps(plan), encoding='utf-8')
    runs_db = tmp_path / 'runs.sqlite3'
    with sqlite3.connect(runs_db) as db:
        db.execute('''CREATE TABLE runs (run_id TEXT,state TEXT,owner_id TEXT,
            tenant_id TEXT,config_json TEXT,manifest_name TEXT)''')
        db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?)',
                   ('run-1', 'running', None, None,
                    json.dumps({'execution_search_plan': plan}), None))
    return run_dir, runs_db, plan


def bound_report_options(tmp_path: Path):
    agent, journal, live = sources(tmp_path)
    with sqlite3.connect(agent) as db:
        db.execute('''CREATE TABLE agent_experiment_reservations_v1
            (session_id TEXT,run_id TEXT,owner_id TEXT,tenant_id TEXT)''')
        db.execute("INSERT INTO agent_experiment_reservations_v1 VALUES('session-1','run-1',NULL,NULL)")
    run_dir, runs_db, plan = bound_run(tmp_path)
    options = dict(journal=journal, agent_db=agent, thread_id='thread-1',
        session_id='session-1', task_id='task-1', owner_id=None, tenant_id=None,
        run_id='run-1', run_dir=run_dir, runs_db=runs_db)
    return options, live, plan


def test_wrong_run_directory_and_foreign_wait_are_rejected(tmp_path):
    options, live, _ = bound_report_options(tmp_path)
    foreign = tmp_path / 'runs' / 'run-2'
    foreign.mkdir()
    (foreign / 'search_summary.json').write_text(json.dumps({
        'schema_version': 'search-summary-v1', 'run_id': 'run-2',
        'plan_digest': 'plan-1'}), encoding='utf-8')
    with pytest.raises(ValueError, match='cost_report_run_source_mismatch'):
        build_report(**{**options, 'run_dir': foreign})
    live.execute('''CREATE TABLE orchestration_monitor_waits_v1
        (id INTEGER,thread_id TEXT,operation_id TEXT,run_id TEXT,task_id TEXT,
        session_id TEXT,status TEXT)''')
    live.execute("INSERT INTO orchestration_monitor_waits_v1 VALUES(1,'thread-1','foreign','run-2','task-2','session-2','succeeded')")
    live.commit()
    with pytest.raises(ValueError, match='cost_report_wait_binding_mismatch'):
        build_report(**options)
    live.close()


def test_foreign_wait_rejected_even_without_calls_and_missing_identity_rejected(tmp_path):
    options, live, _ = bound_report_options(tmp_path)
    live.execute('DELETE FROM orchestration_calls_v1')
    live.execute('''CREATE TABLE orchestration_monitor_waits_v1
        (id INTEGER,thread_id TEXT,operation_id TEXT,run_id TEXT,task_id TEXT,
        session_id TEXT,status TEXT)''')
    live.execute("INSERT INTO orchestration_monitor_waits_v1 VALUES(1,'thread-1','foreign','run-2','task-2','session-2','succeeded')")
    live.commit()
    with pytest.raises(ValueError, match='cost_report_wait_binding_mismatch'):
        build_report(**options)
    live.execute("UPDATE orchestration_monitor_waits_v1 SET run_id='run-1',task_id=NULL,session_id='session-1'")
    live.commit()
    with pytest.raises(ValueError, match='cost_report_wait_binding_mismatch'):
        build_report(**options)
    live.close()


@pytest.mark.parametrize('tamper', ['summary_run', 'summary_plan', 'trial_run',
                                    'timeline_run', 'missing_source'])
def test_search_file_identity_mismatch_is_rejected(tmp_path, tamper):
    options, live, plan = bound_report_options(tmp_path)
    run_dir = options['run_dir']
    (run_dir / 'search_summary.json').write_text(json.dumps({
        'schema_version': 'search-summary-v1', 'run_id': 'run-1',
        'plan_digest': plan['plan_digest'], 'integrity': 'incomplete'}), encoding='utf-8')
    (run_dir / 'search_trials.json').write_text(json.dumps([
        {'run_id': 'run-1', 'fold_index': 0, 'trial_index': 0}]), encoding='utf-8')
    (run_dir / 'search_timeline.json').write_text(json.dumps({
        'schema_version': 'search-timeline-v1', 'run_id': 'run-1',
        'spans': [{'run_id': 'run-1', 'span_id': 'span-1'}]}), encoding='utf-8')
    if tamper == 'summary_run':
        target, field, value = 'search_summary.json', 'run_id', 'run-2'
    elif tamper == 'summary_plan':
        target, field, value = 'search_summary.json', 'plan_digest', 'foreign-plan'
    elif tamper == 'trial_run':
        target, field, value = 'search_trials.json', 'run_id', 'run-2'
    elif tamper == 'timeline_run':
        target, field, value = 'search_timeline.json', 'run_id', 'run-2'
    else:
        (run_dir / 'search_plan.json').unlink()
        target = None
    if target:
        path = run_dir / target
        body = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(body, list):
            body[0][field] = value
        else:
            body[field] = value
        path.write_text(json.dumps(body), encoding='utf-8')
    with pytest.raises(ValueError, match='cost_report_(search_source|plan_mismatch)'):
        build_report(**options)
    live.close()


def test_successful_run_requires_verifiable_manifest(tmp_path):
    options, live, plan = bound_report_options(tmp_path)
    directory = options['run_dir']
    (directory / 'search_summary.json').write_text(json.dumps({
        'schema_version': 'search-summary-v1', 'run_id': 'run-1',
        'plan_digest': plan['plan_digest'], 'integrity': 'complete',
        'final_refit_count': 0}), encoding='utf-8')
    (directory / 'search_trials.json').write_text('[]', encoding='utf-8')
    (directory / 'search_timeline.json').write_text(json.dumps({
        'schema_version': 'search-timeline-v1', 'run_id': 'run-1', 'spans': []}),
        encoding='utf-8')
    with sqlite3.connect(options['runs_db']) as db:
        db.execute("UPDATE runs SET state='succeeded',manifest_name='manifest.json'")
    with pytest.raises(ValueError, match='cost_report_manifest_unverified'):
        build_report(**options)
    entries = {name: {'size_bytes': (directory / name).stat().st_size,
                      'sha256': digest(directory / name)}
               for name in ('search_plan.json', 'search_summary.json',
                            'search_trials.json', 'search_timeline.json')}
    (directory / 'manifest.json').write_text(json.dumps({
        'run_id': 'run-1', 'artifacts': entries}), encoding='utf-8')
    assert json.loads(build_report(**options)['cost_summary.json'])['source']['run_plan_digest'] == 'plan-1'
    (directory / 'search_trials.json').write_text('[ ]\n', encoding='utf-8')
    with pytest.raises(ValueError, match='cost_report_manifest_unverified'):
        build_report(**options)
    live.close()


def test_live_report_uses_durable_fit_entry_and_preserves_unknown_hold(tmp_path):
    from backend.tests.test_step7_training_budget import _setup
    from agent_poc.orchestration.persistence import CallJournal

    agent, policy, reservation, ledger = _setup(tmp_path)
    ledger.bind()
    ledger.enter('run-1:fit:0:trial:0', dimension='model_fits', kind='trial')
    journal = tmp_path / 'journal.sqlite3'
    call_journal = CallJournal(journal, 'thread-fit')
    call_journal.bind_budget(policy)
    call_journal.close()
    run_dir = tmp_path / 'runs' / 'run-1'
    run_dir.mkdir(parents=True)
    with sqlite3.connect(agent) as db:
        session_id = db.execute('''SELECT session_id FROM agent_sessions_v1
            WHERE budget_task_id=?''', (policy.task_id,)).fetchone()[0]
        plan = json.loads(db.execute('''SELECT compiled_config_json FROM
            agent_experiment_reservations_v1 WHERE reservation_id=?''',
            (reservation.experiment_id,)).fetchone()[0])['execution_search_plan']
    (run_dir / 'search_plan.json').write_text(json.dumps(plan), encoding='utf-8')
    runs_db = tmp_path / 'runs.sqlite3'
    with sqlite3.connect(runs_db) as db:
        db.execute('''CREATE TABLE runs (run_id TEXT,state TEXT,owner_id TEXT,
            tenant_id TEXT,config_json TEXT,manifest_name TEXT)''')
        db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?)',
                   ('run-1', 'running', None, None,
                    json.dumps({'execution_search_plan': plan}), None))
    options = dict(journal=journal, agent_db=agent, thread_id='thread-fit',
        session_id=session_id, task_id=policy.task_id, owner_id=None,
        tenant_id=None, run_id='run-1', run_dir=run_dir, runs_db=runs_db)
    report = build_report(**options)
    summary = json.loads(report['cost_summary.json'])
    fits = summary['budget_dimensions']['model_fits']
    assert summary['schema_version'] == 'task-cost-report-v1'
    assert summary['budget_dimensions']['llm_calls']['limit'] == policy.limits['llm_calls']
    assert (fits['known_actual'], fits['held_reserved'], fits['held_unknown']) == (1, 1, 0)
    assert fits['known_subtotal'] == 1 and fits['actual_total'] is None
    usage = list(csv.DictReader(report['training_usage.csv'].splitlines()))
    assert usage[0]['event_id'] == 'run-1:fit:0:trial:0'
    assert usage[0]['state'] == 'entered'
    ledger.settle(exit_confirmed=False)
    held = json.loads(build_report(**options)['cost_summary.json'])
    assert held['budget_dimensions']['model_fits']['held_unknown'] == 1
    assert held['budget_dimensions']['model_fits']['actual_total'] is None
    assert held['report_status'] == 'partial'
    with sqlite3.connect(agent) as db:
        db.execute("UPDATE task_training_events_v1 SET payload_digest='corrupt'")
    with pytest.raises(ValueError, match='cost_report_training_source_mismatch'):
        export_report(tmp_path / 'invalid-report', **options)
    assert not (tmp_path / 'invalid-report').exists()


def test_terminated_without_run_has_settled_v1_report(api, tmp_path):
    from backend.tests.test_step7_v5_session import _request, HEADERS
    from backend.app.agent.budget import BudgetPolicy
    from agent_poc.orchestration.persistence import CallJournal

    client, storage, dataset = api
    request = _request(dataset, awareness='off')
    created = client.post('/api/agent/v2/sessions', headers=HEADERS, json=request)
    assert created.status_code == 201, created.text
    session_id = created.json()['session_id']
    stopped = client.post(f'/api/agent/v2/sessions/{session_id}/terminate',
        headers=HEADERS, json={'client_request_id':'report-stop',
                               'reason':'budget_exhausted'})
    assert stopped.status_code == 200, stopped.text
    journal_path = tmp_path / 'calls.sqlite3'
    journal = CallJournal(journal_path, 'report-thread')
    journal.bind_budget(BudgetPolicy.from_dict(request['budget_policy']))
    journal.close()
    report = build_report(journal=journal_path,
        agent_db=storage / 'agent.sqlite3', thread_id='report-thread',
        session_id=session_id, task_id='session-budget-v5',
        owner_id=None, tenant_id=None)
    summary = json.loads(report['cost_summary.json'])
    assert summary['schema_version'] == 'task-cost-report-v1'
    assert summary['task_status'] == 'terminated'
    assert summary['report_status'] == 'settled'
    assert summary['duration_scopes']['task_wall_duration']['known_count'] == 1


def test_training_events_and_termination_are_scoped_to_requested_task(tmp_path):
    from backend.tests.test_step7_training_budget import _setup
    from scripts.task_cost_report import _backend_budget

    agent, policy, reservation, ledger = _setup(tmp_path)
    ledger.bind()
    ledger.enter('own-fit', dimension='model_fits', kind='trial')
    with sqlite3.connect(agent) as db:
        session = db.execute('SELECT session_id FROM agent_sessions_v1 WHERE budget_task_id=?',
                             (policy.task_id,)).fetchone()[0]
        db.execute('''INSERT INTO task_training_executions_v1
            VALUES('foreign-reservation','other-task','other-run','other-claim',
                   'unknown_pending','2026-09-27T00:00:00+00:00',NULL)''')
        payload = hashlib.sha256(json.dumps({'dimension':'model_fits','kind':'trial'},
                                            sort_keys=True).encode()).hexdigest()
        db.execute('''INSERT INTO task_training_events_v1 VALUES
            ('foreign-reservation','other-fit',?,'model_fits','trial','entered',
             '2026-09-27T00:00:00+00:00',NULL)''', (payload,))
        db.execute('''INSERT INTO task_training_terminations_v1 VALUES
            ('foreign-reservation','guardian_lost','2026-09-27T00:00:00+00:00',NULL,NULL,NULL)''')
    result = _backend_budget(agent, session_id=session, task_id=policy.task_id, run_id='run-1')
    assert [row['event_id'] for row in result['events']] == ['own-fit']
    assert result['terminations'] == []
    assert result['dimensions']['model_fits']['known_actual'] == 1
    with sqlite3.connect(agent) as db:
        db.execute("UPDATE task_training_events_v1 SET payload_digest='corrupt' WHERE event_id='own-fit'")
    with pytest.raises(ValueError, match='cost_report_training_source_mismatch'):
        _backend_budget(agent, session_id=session, task_id=policy.task_id, run_id='run-1')

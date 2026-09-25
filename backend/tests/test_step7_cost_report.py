"""The exporter reads legacy WAL state without migrating or exposing proposals."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

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
    run_dir = tmp_path / 'run-1'
    run_dir.mkdir()
    (run_dir / 'search_summary.json').write_text(json.dumps(dict(
        integrity='complete', final_refit_count=1,
        selection_metric='oob_balanced_accuracy_then_accuracy')), encoding='utf-8')
    (run_dir / 'search_trials.json').write_text(json.dumps([dict(
        fold_index=0, trial_index=0, state='succeeded', actual_epochs=0,
        training_batches=0, started_at='2026-09-25T00:00:01+00:00',
        ended_at='2026-09-25T00:00:04+00:00', duration_seconds=3.0)]), encoding='utf-8')
    (run_dir / 'search_timeline.json').write_text(json.dumps(dict(spans=[
        dict(span_id='parent', parent_span_id=None, stage='search', status='succeeded',
             started_at='2026-09-25T00:00:00+00:00', ended_at='2026-09-25T00:00:05+00:00',
             duration_seconds=5.0, duration_clock='process_monotonic'),
        dict(span_id='child', parent_span_id='parent', stage='trial_fit', status='succeeded',
             started_at='2026-09-25T00:00:01+00:00', ended_at='2026-09-25T00:00:04+00:00',
             duration_seconds=3.0, duration_clock='process_monotonic')])), encoding='utf-8')
    files = build_report(journal=journal, agent_db=agent, thread_id='thread-1',
        session_id='session-1', task_id='task-1', owner_id=None, tenant_id=None,
        run_id='run-1', run_dir=run_dir)
    usage = list(csv.DictReader(files['training_usage.csv'].splitlines()))
    assert [row['fit_purpose'] for row in usage] == ['candidate', 'final_refit']
    assert usage[0]['measurement_scope'] == 'train_oob'
    events = list(csv.DictReader(files['timeline_events.csv'].splitlines()))
    assert events[1]['parent_event_id'] == 'parent'
    assert 'sum_duration_seconds' not in json.loads(files['cost_summary.json'])
    live.close()

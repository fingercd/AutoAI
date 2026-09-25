"""Read-only, scope-bound task cost snapshot; no model or proposal decoding."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any


SCHEMA = 'task-cost-report-v1'
CALL_FIELDS = ('id', 'thread_id', 'operation_id', 'kind', 'name', 'status',
               'started_at', 'ended_at', 'attempt_index', 'task_id', 'session_id',
               'run_id', 'input_tokens', 'output_tokens', 'total_tokens',
               'token_status', 'error_code', 'request_started_at_utc',
               'response_ended_at_utc', 'request_duration_seconds',
               'measurement_version', 'model_id_sha256', 'model_id', 'protocol', 'phase',
               'request_outcome', 'budget_phase')
CALL_FIELDS += ('cached_tokens', 'prepare_duration_seconds', 'parse_duration_seconds')
WAIT_FIELDS = ('operation_id', 'run_id', 'task_id', 'session_id', 'status',
               'started_at_utc', 'ended_at_utc', 'duration_seconds',
               'duration_clock', 'reason_code')


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _optional_sha(path: Path) -> str | None:
    return _sha(path) if path.is_file() else None


def readonly_snapshot(path: Path) -> sqlite3.Connection:
    """SQLite backup includes committed WAL pages; source is opened mode=ro."""
    if not path.is_file():
        raise FileNotFoundError(path)
    source = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    target = sqlite3.connect(':memory:')
    try:
        source.backup(target)
    finally:
        source.close()
    target.row_factory = sqlite3.Row
    return target


def _rows(db: sqlite3.Connection, table: str, fields: tuple[str, ...],
          *, where: str = '', params: tuple = ()) -> list[dict[str, Any]]:
    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is None:
        return []
    present = {row['name'] for row in db.execute(f'PRAGMA table_info({table})')}
    columns = [name for name in fields if name in present]
    selected = ','.join(columns)
    return [{name: row[name] if name in columns else None for name in fields}
            for row in db.execute(f'SELECT {selected} FROM {table} {where}', params)]


def call_rows(journal: Path, thread_id: str) -> tuple[list[dict], list[dict]]:
    db = readonly_snapshot(journal)
    try:
        calls = _rows(db, 'orchestration_calls_v1', CALL_FIELDS,
                      where='WHERE thread_id=? ORDER BY id', params=(thread_id,))
        waits = _rows(db, 'orchestration_monitor_waits_v1', WAIT_FIELDS,
                      where='WHERE thread_id=? ORDER BY id', params=(thread_id,))
    finally:
        db.close()
    return calls, waits


def _bound_session(agent_db: Path, *, session_id: str, run_id: str | None,
                   owner_id: str | None, tenant_id: str | None,
                   task_id: str | None) -> None:
    db = readonly_snapshot(agent_db)
    try:
        columns = {item['name'] for item in db.execute('PRAGMA table_info(agent_sessions_v1)')}
        task_column = 'budget_task_id' if 'budget_task_id' in columns else 'NULL AS budget_task_id'
        row = db.execute(f'''SELECT owner_id,tenant_id,{task_column} FROM agent_sessions_v1
            WHERE session_id=?''', (session_id,)).fetchone()
        if row is None or (row['owner_id'], row['tenant_id']) != (owner_id, tenant_id):
            raise ValueError('cost_report_scope_mismatch')
        if row['budget_task_id'] is not None and row['budget_task_id'] != task_id:
            raise ValueError('cost_report_task_mismatch')
        if run_id is not None:
            found = db.execute('''SELECT 1 FROM agent_experiment_reservations_v1
                WHERE session_id=? AND run_id=? AND owner_id IS ? AND tenant_id IS ?''',
                (session_id, run_id, owner_id, tenant_id)).fetchone()
            if found is None:
                raise ValueError('cost_report_run_mismatch')
    finally:
        db.close()


def _backend_budget(agent_db: Path, *, session_id: str,
                    task_id: str | None, run_id: str | None) -> dict | None:
    """Read one consistent backend snapshot without migrating the source DB."""
    db = readonly_snapshot(agent_db)
    try:
        columns = {item['name'] for item in db.execute('PRAGMA table_info(agent_sessions_v1)')}
        if 'budget_task_id' not in columns:
            return None
        session = db.execute('''SELECT state,finalized_at,terminated_at,budget_task_id,budget_policy_digest FROM
            agent_sessions_v1 WHERE session_id=?''', (session_id,)).fetchone()
        if session is None or session['budget_task_id'] is None:
            return None
        if session['budget_task_id'] != task_id:
            raise ValueError('cost_report_task_mismatch')
        from backend.app.agent.budget import DIMENSIONS, BudgetError, dimension_summary
        try:
            policy = db.execute('''SELECT policy_json,policy_digest FROM
                task_budget_policies_v1 WHERE task_id=? AND owner='backend' ''',
                (task_id,)).fetchone()
            if policy is None or policy['policy_digest'] != session['budget_policy_digest']:
                raise ValueError('cost_report_policy_mismatch')
            dimensions = {name: dimension_summary(db, task_id=task_id,
                owner='backend', dimension=name) for name, (_, owner) in DIMENSIONS.items()
                if owner == 'backend'}
        except (sqlite3.OperationalError, KeyError, BudgetError) as exc:
            raise ValueError('cost_report_policy_mismatch') from exc
        reservations = _rows(db, 'task_budget_reservations_v1',
            ('operation_id','attempt_id','dimension','amount','actual','held','status','source_ref'),
            where="WHERE task_id=? AND owner='backend' ORDER BY operation_id,dimension",
            params=(task_id,))
        ids = {row['operation_id'] for row in reservations if row['dimension'] == 'model_fits'}
        mapped = {row['reservation_id']: row['run_id'] for row in db.execute('''
            SELECT reservation_id,run_id FROM agent_experiment_reservations_v1
            WHERE session_id=? AND state IN ('bound','compensation_required')''',
            (session_id,))}
        executions = _rows(db, 'task_training_executions_v1',
            ('reservation_id','task_id','run_id','state','started_at','finished_at'),
            where='WHERE task_id=? ORDER BY reservation_id', params=(task_id,))
        if any(row['reservation_id'] not in ids or
               mapped.get(row['reservation_id']) != row['run_id'] for row in executions):
            raise ValueError('cost_report_training_source_mismatch')
        if run_id is not None and any(row['run_id'] != run_id for row in executions):
            raise ValueError('cost_report_training_source_mismatch')
        event_ids = tuple(row['reservation_id'] for row in executions)
        events = _rows(db, 'task_training_events_v1',
            ('reservation_id','event_id','payload_digest','dimension','kind','status','entered_at','completed_at'),
            where='''WHERE reservation_id IN (SELECT reservation_id
                FROM task_training_executions_v1 WHERE task_id=?)
                ORDER BY reservation_id,event_id''', params=(task_id,))
        if any(row['reservation_id'] not in event_ids or
               row['dimension'] not in ('model_fits','training_epochs') or
               row['payload_digest'] != hashlib.sha256(json.dumps({
                   'dimension': row['dimension'], 'kind': row['kind']},
                   sort_keys=True).encode('utf-8')).hexdigest() for row in events):
            raise ValueError('cost_report_training_source_mismatch')
        terminations = _rows(db, 'task_training_terminations_v1',
            ('reservation_id','reason','requested_at','exited_at','latency_seconds','exit_code'),
            where='''WHERE reservation_id IN (SELECT reservation_id
                FROM task_training_executions_v1 WHERE task_id=?)
                ORDER BY reservation_id''', params=(task_id,))
        if any(row['reservation_id'] not in event_ids for row in terminations):
            raise ValueError('cost_report_training_source_mismatch')
        return dict(task_status='terminated' if session['terminated_at'] is not None else session['state'],
                    completed_at=session['terminated_at'] or session['finalized_at'],
                    policy=json.loads(policy['policy_json']),
                    policy_digest=policy['policy_digest'], dimensions=dimensions,
                    reservations=reservations, executions=executions,
                    events=events, terminations=terminations)
    finally:
        db.close()


def _journal_budget(journal: Path, *, thread_id: str, task_id: str | None,
                    policy_digest: str | None) -> dict | None:
    db = readonly_snapshot(journal)
    try:
        if db.execute('''SELECT 1 FROM sqlite_master WHERE type='table'
            AND name='task_budget_journal_bindings_v1' ''').fetchone() is None:
            return None
        row = db.execute('''SELECT task_id,canonical_path,policy_digest FROM
            task_budget_journal_bindings_v1 WHERE thread_id=?''',
            (thread_id,)).fetchone()
        if row is None:
            return None
        if (row['task_id'] != task_id or row['policy_digest'] != policy_digest or
                Path(row['canonical_path']).resolve() != journal.resolve()):
            raise ValueError('cost_report_journal_binding_mismatch')
        from backend.app.agent.budget import DIMENSIONS, BudgetError, dimension_summary
        try:
            return {name: dimension_summary(db, task_id=task_id,
                owner='journal', dimension=name) for name, (_, owner) in DIMENSIONS.items()
                if owner == 'journal'}
        except (sqlite3.OperationalError, KeyError, BudgetError) as exc:
            raise ValueError('cost_report_journal_policy_mismatch') from exc
    finally:
        db.close()


def _bound_run(runs_db: Path, *, run_id: str, run_dir: Path,
               owner_id: str | None, tenant_id: str | None) -> dict:
    """Resolve one Run from its read-only registry; never trust a supplied directory alone."""
    if not run_id or Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('cost_report_run_mismatch')
    expected = (runs_db.parent / 'runs').resolve() / run_id
    if run_dir.resolve() != expected or not expected.is_dir():
        raise ValueError('cost_report_run_source_mismatch')
    db = readonly_snapshot(runs_db)
    try:
        row = db.execute('''SELECT state,owner_id,tenant_id,config_json,manifest_name
            FROM runs WHERE run_id=?''', (run_id,)).fetchone()
        if row is None or (row['owner_id'], row['tenant_id']) != (owner_id, tenant_id):
            raise ValueError('cost_report_run_source_mismatch')
        config = json.loads(row['config_json'])
        plan = config.get('execution_search_plan')
        if not isinstance(plan, dict) or not isinstance(plan.get('plan_digest'), str):
            raise ValueError('cost_report_plan_unverified')
        return dict(state=row['state'], manifest_name=row['manifest_name'], plan=plan)
    finally:
        db.close()


def _verified_search_files(run_dir: Path, run_id: str, run: dict) -> tuple[Any, Any, Any]:
    plan = _json(run_dir / 'search_plan.json')
    summary = _json(run_dir / 'search_summary.json')
    trials = _json(run_dir / 'search_trials.json')
    timeline = _json(run_dir / 'search_timeline.json')
    if plan != run['plan']:
        raise ValueError('cost_report_plan_mismatch')
    digest = plan['plan_digest']
    if summary is not None and (type(summary) is not dict or
            summary.get('schema_version') != 'search-summary-v1' or
            summary.get('run_id') != run_id or summary.get('plan_digest') != digest):
        raise ValueError('cost_report_search_source_mismatch')
    if trials is not None and (type(trials) is not list or any(
            type(item) is not dict or item.get('run_id') != run_id for item in trials)):
        raise ValueError('cost_report_search_source_mismatch')
    if timeline is not None and (type(timeline) is not dict or
            timeline.get('schema_version') != 'search-timeline-v1' or
            timeline.get('run_id') != run_id or type(timeline.get('spans')) is not list or
            any(type(span) is not dict or span.get('run_id') != run_id
                for span in timeline['spans'])):
        raise ValueError('cost_report_search_source_mismatch')
    if run['state'] == 'succeeded':
        if (run['manifest_name'] != 'manifest.json' or
                summary is None or trials is None or timeline is None or
                summary.get('integrity') != 'complete'):
            raise ValueError('cost_report_manifest_unverified')
        from backend.app.runs.artifacts import RunArtifactWriter
        reader = RunArtifactWriter(run_dir)
        try:
            manifest = reader.load_manifest()
            if manifest.get('run_id') != run_id:
                raise ValueError('cost_report_manifest_unverified')
            for name in ('search_plan.json', 'search_summary.json',
                         'search_trials.json', 'search_timeline.json'):
                entry, path = reader._resolve_manifest_entry(manifest, name)
                if (type(entry.get('size_bytes')) is not int or
                        type(entry.get('sha256')) is not str or
                        len(entry['sha256']) != 64):
                    raise ValueError('cost_report_manifest_unverified')
                reader._verify_entry(path, entry, verify_hash=True)
        except (OSError, ValueError) as error:
            raise ValueError('cost_report_manifest_unverified') from error
    return summary, trials, timeline


def _measure(rows: list[dict], field: str, *, applicable=lambda row: True) -> dict:
    selected = [row for row in rows if applicable(row)]
    known = [row[field] for row in selected if row[field] is not None]
    return dict(known_subtotal=sum(known), actual_total=sum(known) if len(known) == len(selected) else None,
                known_count=len(known), unknown_count=len(selected)-len(known),
                not_applicable_count=len(rows)-len(selected), applicable_count=len(selected))


def _duration(start: str | None, end: str | None) -> float | None:
    if start is None or end is None:
        return None
    try:
        first = datetime.fromisoformat(start.replace('Z', '+00:00'))
        last = datetime.fromisoformat(end.replace('Z', '+00:00'))
        seconds = (last - first).total_seconds()
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else None


def _csv(name: str, rows: list[dict], fields: tuple[str, ...]) -> tuple[str, str]:
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=_fields(rows, fields), extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return name, output.getvalue()


def _fields(rows: list[dict], fallback: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(key for row in rows for key in row)) if rows else fallback


def _report_dimension(value: dict) -> dict:
    unknown = (value['unknown_count'] > 0 or value['held_unknown'] > 0 or
               value['held_reserved'] > 0)
    return {**value, 'known_subtotal': value['known_actual'],
            'actual_total': None if unknown else value['known_actual'],
            'measurement_status': 'partial' if unknown else
                'not_started' if value['known_count'] == 0 else 'known',
            'not_applicable_count': 0, 'source_version': 'task-budget-ledger-v1'}


def build_report(*, journal: Path, agent_db: Path, thread_id: str,
                 session_id: str, task_id: str | None,
                 owner_id: str | None, tenant_id: str | None,
                 run_id: str | None = None, run_dir: Path | None = None,
                 runs_db: Path | None = None) -> dict[str, str]:
    _bound_session(agent_db, session_id=session_id, run_id=run_id,
                   owner_id=owner_id, tenant_id=tenant_id, task_id=task_id)
    backend_budget = _backend_budget(agent_db, session_id=session_id,
                                     task_id=task_id, run_id=run_id)
    journal_dimensions = _journal_budget(journal, thread_id=thread_id,
        task_id=task_id, policy_digest=backend_budget['policy_digest'] if backend_budget else None)
    if backend_budget is None and journal_dimensions is not None:
        raise ValueError('cost_report_policy_mismatch')
    calls, waits = call_rows(journal, thread_id)
    if any(row['task_id'] != task_id or
           row['run_id'] not in (None, run_id) or
           (row['session_id'] != session_id and not
            (row['session_id'] is None and row['run_id'] is None and
             row['name'] in ('inspect_ml_capabilities','start_ml_session')))
           for row in calls):
        raise ValueError('cost_report_call_binding_mismatch')
    if any((wait['task_id'], wait['session_id'], wait['run_id']) !=
           (task_id, session_id, run_id) for wait in waits):
        raise ValueError('cost_report_wait_binding_mismatch')
    if run_dir is not None:
        if run_id is None or runs_db is None:
            raise ValueError('cost_report_run_source_required')
        run = _bound_run(runs_db, run_id=run_id, run_dir=run_dir,
                         owner_id=owner_id, tenant_id=tenant_id)
        summary, trials, timeline = _verified_search_files(run_dir, run_id, run)
    else:
        summary = trials = timeline = None
    requests = []
    attempts = []
    for row in calls:
        # An old "dispatched" row was written before prompt preparation. Its
        # status alone cannot prove a physical HTTP send.
        dispatched = ('known' if row['budget_phase'] is not None and row['status'] not in ('prepared', 'prepare_failed')
                      else 'not_sent' if row['status'] == 'prepare_failed' else 'unknown')
        base = dict(event_id=f"call-{row['id']}", operation_id=row['operation_id'],
                    attempt_index=row['attempt_index'], kind=row['kind'], name=row['name'],
                    status=row['status'], dispatch_evidence=dispatched,
                    started_at=row['started_at'], ended_at=row['ended_at'],
                    request_started_at_utc=row['request_started_at_utc'],
                    response_ended_at_utc=row['response_ended_at_utc'],
                    request_duration_seconds=row['request_duration_seconds'],
                    error_code=row['error_code'], budget_phase=row['budget_phase'])
        attempts.append(base)
        if row['kind'] == 'llm':
            requests.append({**base, 'phase': row['phase'], 'protocol': row['protocol'],
                             'model_id_sha256': row['model_id_sha256'],
                             'measurement_version': row['measurement_version'],
                             'request_outcome': row['request_outcome'],
                             'prepare_duration_seconds': row['prepare_duration_seconds'],
                             'parse_duration_seconds': row['parse_duration_seconds'],
                             'input_tokens': row['input_tokens'],
                             'output_tokens': row['output_tokens'],
                             'total_tokens': row['total_tokens'],
                             'cached_tokens': row['cached_tokens'],
                             'cached_token_status': 'known' if row['cached_tokens'] is not None else 'unknown',
                             'token_status': row['token_status'] or 'unknown'})
    training = []
    for trial in trials or []:
        training.append(dict(event_id=f"trial-{trial.get('fold_index')}-{trial.get('trial_index')}",
            **({'record_type': 'search_trial'} if backend_budget is not None else {}),
            run_id=trial['run_id'], fold_index=trial.get('fold_index'), trial_index=trial.get('trial_index'),
            fit_purpose='candidate', state=trial.get('state'),
            fit_started=True if backend_budget is None and trial.get('state') == 'succeeded' else None,
            fit_start_status=('known' if trial.get('state') == 'succeeded' else 'unknown')
                if backend_budget is None else 'see_budget_events',
            entered_epochs=trial.get('actual_epochs') if backend_budget is None and
                trial.get('state') == 'succeeded' else None,
            completed_epochs=trial.get('actual_epochs') if backend_budget is None and
                trial.get('state') == 'succeeded' else None,
            training_batches=trial.get('training_batches'),
            started_at=trial.get('started_at'), ended_at=trial.get('ended_at'),
            duration_seconds=trial.get('duration_seconds'),
            measurement_scope=('train_oob' if summary and summary.get('selection_metric') ==
                               'oob_balanced_accuracy_then_accuracy' else
                               'valid_loss' if summary and summary.get('selection_metric') == 'best_valid_loss'
                               else 'valid' if summary else None)))
    if backend_budget is None and summary is not None and summary.get('integrity') == 'complete':
        for index in range(summary.get('final_refit_count') or 0):
            training.append(dict(event_id=f'final-refit-{index}', run_id=run_id,
                fold_index=None, trial_index=None, fit_purpose='final_refit',
                state='succeeded', fit_started=True, fit_start_status='known',
                entered_epochs=None, completed_epochs=None, training_batches=None,
                started_at=None, ended_at=None, duration_seconds=None,
                measurement_scope='train_valid_refit'))
    if backend_budget is not None:
        execution_runs = {row['reservation_id']: row['run_id']
                          for row in backend_budget['executions']}
        for entry in backend_budget['events']:
            training.append(dict(event_id=entry['event_id'], record_type='budget_event',
                run_id=execution_runs[entry['reservation_id']],
                reservation_id=entry['reservation_id'], dimension=entry['dimension'],
                fit_purpose=entry['kind'] if entry['dimension'] == 'model_fits' else None,
                state=entry['status'], fit_started=True if entry['dimension'] == 'model_fits' else None,
                fit_start_status='known' if entry['dimension'] == 'model_fits' else 'not_applicable',
                entered_epochs=1 if entry['dimension'] == 'training_epochs' else None,
                completed_epochs=(1 if entry['status'] == 'completed' else 0)
                    if entry['dimension'] == 'training_epochs' else None,
                started_at=entry['entered_at'], ended_at=entry['completed_at'],
                duration_seconds=None, measurement_scope='train'))
    events = []
    for wait in waits:
        events.append(dict(event_id=f"wait-{wait['operation_id']}", parent_event_id=None,
            stage='monitor_wait', status=wait['status'], started_at=wait['started_at_utc'],
            ended_at=wait['ended_at_utc'], duration_seconds=wait['duration_seconds'],
            duration_clock=wait['duration_clock'], reason_code=wait['reason_code']))
    for span in (timeline or {}).get('spans', []):
        events.append(dict(event_id=span.get('span_id'), parent_event_id=span.get('parent_span_id'),
            stage=span.get('stage'), status=span.get('status'),
            started_at=span.get('started_at'), ended_at=span.get('ended_at'),
            duration_seconds=span.get('duration_seconds'), duration_clock=span.get('duration_clock'),
            reason_code=None))
    events.sort(key=lambda row: (str(row['started_at']), str(row['event_id'])))
    coverage = {field: _measure(requests, field) for field in
                ('input_tokens', 'output_tokens', 'cached_tokens', 'total_tokens',
                 'request_duration_seconds')}
    coverage.update(trial_duration=_measure(training, 'duration_seconds'),
                    entered_epochs=_measure(training, 'entered_epochs',
                                            applicable=lambda row: row.get('dimension') == 'training_epochs'
                                            if backend_budget is not None else
                                            row['fit_purpose'] == 'candidate' and
                                            summary is not None and
                                            summary.get('selection_metric') == 'best_valid_loss'))
    if backend_budget is not None:
        workers = [{**entry, 'duration_seconds':_duration(entry['started_at'],
            entry['finished_at'])} for entry in backend_budget['executions']]
        coverage['worker_parent_duration'] = _measure(workers, 'duration_seconds')
        completed_at = backend_budget['completed_at']
        task_started = datetime.fromtimestamp(backend_budget['policy']['started_at'],
            timezone.utc).isoformat()
        coverage['task_wall_duration'] = _measure([{
            'duration_seconds':_duration(task_started, completed_at)}], 'duration_seconds')
    coverage['monitor_wait_duration'] = _measure(waits, 'duration_seconds')
    search_parents = [event for event in events if event['stage'] == 'search']
    coverage['search_parent_duration'] = _measure(search_parents, 'duration_seconds')
    dimensions = ({name: _report_dimension(value) for name, value in
                   {**backend_budget['dimensions'], **journal_dimensions}.items()}
        if backend_budget is not None and journal_dimensions is not None else
        {name: _report_dimension(value) for name, value in
         backend_budget['dimensions'].items()} if backend_budget is not None else None)
    settled = (backend_budget is not None and
        journal_dimensions is not None and
        backend_budget['task_status'] in ('finalized', 'terminated') and
        (not backend_budget['executions'] or run_dir is not None and
         run['state'] in ('succeeded', 'failed', 'cancelled')) and
        all(row['held_reserved'] == 0 and row['held_unknown'] == 0
            for row in dimensions.values()) and
        all(row['state'] == 'settled' for row in backend_budget['executions']))
    cost_summary = dict(schema_version=SCHEMA if backend_budget else 'task-cost-snapshot-v1',
        aggregation_version='task-cost-reduction-v1',
        source=dict(journal_sha256=_sha(journal), agent_db_sha256=_sha(agent_db),
                    runs_db_sha256=_sha(runs_db) if run_dir is not None else None,
                    run_plan_digest=run['plan']['plan_digest'] if run_dir is not None else None,
                    journal_wal_sha256=_optional_sha(journal.with_name(journal.name + '-wal')),
                    agent_wal_sha256=_optional_sha(agent_db.with_name(agent_db.name + '-wal')),
                    search_summary_sha256=_optional_sha(run_dir / 'search_summary.json') if run_dir else None,
                    search_trials_sha256=_optional_sha(run_dir / 'search_trials.json') if run_dir else None,
                    run_timeline_sha256=_optional_sha(run_dir / 'search_timeline.json') if run_dir else None,
                    call_snapshot_sha256=hashlib.sha256(json.dumps(calls, sort_keys=True,
                        ensure_ascii=True, allow_nan=False).encode('utf-8')).hexdigest()),
        binding=dict(thread_id=thread_id, task_id=task_id, session_id=session_id,
                     run_id=run_id, owner_id=owner_id, tenant_id=tenant_id),
        **(dict(task_status=backend_budget['task_status'],
            budget_policy_digest=backend_budget['policy_digest'],
            budget_dimensions=dimensions,
            duration_scopes={name:coverage[name] for name in
                ('task_wall_duration','worker_parent_duration',
                 'search_parent_duration','monitor_wait_duration')},
            training_executions=backend_budget['executions'],
            training_terminations=backend_budget['terminations'])
            if backend_budget else {}),
        call_records=len(calls), llm_request_records=len(requests),
        training_trial_records=len(training), search_summary=summary,
        metrics=coverage, report_status=('settled' if settled else 'partial')
            if backend_budget else 'partial_budget_settlement_unverified',
        warnings=['旧 journal 状态不证明实际物理发送；未知用量未按零计算。',
                  '父子时间段重叠，不能直接相加；货币、GPU 秒不可用。'] +
                 (['后端与调用账本是独立 SQLite 快照，跨库没有原子时间点。']
                  if backend_budget else []))
    markdown = (f'# 任务成本报告\n\n任务：{task_id or "历史未绑定"}；Session：{session_id}；'
                f'Run：{run_id or "无"}。\n\n'
                f'状态：{cost_summary["report_status"]}；LLM 记录 {len(requests)} 条；'
                f'训练明细 {len(training)} 条。'
                '覆盖率详见 coverage.json。\n\n'
                '未知成本保留为未知，预留量不算实际支出。父子时间段不可求和。'
                '货币、电费与 GPU 秒不可用。\n')
    files = {
        'cost_summary.json': json.dumps(cost_summary, ensure_ascii=False, indent=2, allow_nan=False),
        'coverage.json': json.dumps(coverage, ensure_ascii=False, indent=2, allow_nan=False),
        'cost_report.md': markdown,
    }
    files.update([_csv('llm_requests.csv', requests, tuple(requests[0]) if requests else
                       tuple(dict(event_id=None, operation_id=None, input_tokens=None, output_tokens=None, cached_tokens=None, total_tokens=None))),
                  _csv('operation_attempts.csv', attempts, tuple(attempts[0]) if attempts else
                       tuple(dict(event_id=None, operation_id=None, status=None, dispatch_evidence=None))),
                  _csv('training_usage.csv', training, tuple(training[0]) if training else
                       tuple(dict(event_id=None, run_id=None, fit_purpose=None, state=None))),
                  _csv('timeline_events.csv', events, tuple(events[0]) if events else
                       tuple(dict(event_id=None, parent_event_id=None, stage=None, status=None)))])
    return files


def export_report(output: Path, **kwargs) -> None:
    files = build_report(**kwargs)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='step7-report-', dir=output.parent) as folder:
        temporary = Path(folder)
        for name, content in files.items():
            (temporary / name).write_text(content, encoding='utf-8')
        os.rename(temporary, output)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', required=True, type=Path)
    parser.add_argument('--agent-db', required=True, type=Path)
    parser.add_argument('--thread-id', required=True)
    parser.add_argument('--session-id', required=True)
    parser.add_argument('--task-id')
    parser.add_argument('--owner-id')
    parser.add_argument('--tenant-id')
    parser.add_argument('--run-id')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--runs-db', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    export_report(args.output, journal=args.journal, agent_db=args.agent_db,
                  thread_id=args.thread_id, session_id=args.session_id,
                  task_id=args.task_id, owner_id=args.owner_id, tenant_id=args.tenant_id,
                  run_id=args.run_id, run_dir=args.run_dir, runs_db=args.runs_db)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

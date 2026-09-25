"""Read-only, scope-bound task cost snapshot; no model or proposal decoding."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any


SCHEMA = 'task-cost-snapshot-v1'
CALL_FIELDS = ('id', 'thread_id', 'operation_id', 'kind', 'name', 'status',
               'started_at', 'ended_at', 'attempt_index', 'task_id', 'session_id',
               'run_id', 'input_tokens', 'output_tokens', 'total_tokens',
               'token_status', 'error_code', 'request_started_at_utc',
               'response_ended_at_utc', 'request_duration_seconds',
               'measurement_version', 'model_id_sha256', 'model_id', 'protocol', 'phase',
               'request_outcome', 'budget_phase')
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


def _measure(rows: list[dict], field: str, *, applicable=lambda row: True) -> dict:
    selected = [row for row in rows if applicable(row)]
    known = [row[field] for row in selected if row[field] is not None]
    return dict(known_subtotal=sum(known), actual_total=sum(known) if len(known) == len(selected) else None,
                known_count=len(known), unknown_count=len(selected)-len(known),
                not_applicable_count=len(rows)-len(selected), applicable_count=len(selected))


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else None


def _csv(name: str, rows: list[dict], fields: tuple[str, ...]) -> tuple[str, str]:
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return name, output.getvalue()


def build_report(*, journal: Path, agent_db: Path, thread_id: str,
                 session_id: str, task_id: str | None,
                 owner_id: str | None, tenant_id: str | None,
                 run_id: str | None = None, run_dir: Path | None = None) -> dict[str, str]:
    _bound_session(agent_db, session_id=session_id, run_id=run_id,
                   owner_id=owner_id, tenant_id=tenant_id, task_id=task_id)
    calls, waits = call_rows(journal, thread_id)
    if any(row['session_id'] not in (None, session_id) or
           row['task_id'] not in (None, task_id) or
           row['run_id'] not in (None, run_id) for row in calls):
        raise ValueError('cost_report_call_binding_mismatch')
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
                    error_code=row['error_code'])
        attempts.append(base)
        if row['kind'] == 'llm':
            requests.append({**base, 'phase': row['phase'], 'protocol': row['protocol'],
                             'model_id_sha256': row['model_id_sha256'],
                             'measurement_version': row['measurement_version'],
                             'request_outcome': row['request_outcome'],
                             'input_tokens': row['input_tokens'],
                             'output_tokens': row['output_tokens'],
                             'total_tokens': row['total_tokens'],
                             'cached_tokens': None,
                             'cached_token_status': 'unknown',
                             'token_status': row['token_status'] or 'unknown'})
    summary = _json(run_dir / 'search_summary.json') if run_dir else None
    trials = _json(run_dir / 'search_trials.json') if run_dir else None
    timeline = _json(run_dir / 'search_timeline.json') if run_dir else None
    training = []
    for trial in trials or []:
        training.append(dict(event_id=f"trial-{trial.get('fold_index')}-{trial.get('trial_index')}",
            run_id=run_id, fold_index=trial.get('fold_index'), trial_index=trial.get('trial_index'),
            fit_purpose='candidate', state=trial.get('state'),
            fit_started=True if trial.get('state') == 'succeeded' else None,
            fit_start_status='known' if trial.get('state') == 'succeeded' else 'unknown',
            entered_epochs=trial.get('actual_epochs') if trial.get('state') == 'succeeded' else None,
            completed_epochs=trial.get('actual_epochs') if trial.get('state') == 'succeeded' else None,
            training_batches=trial.get('training_batches'),
            started_at=trial.get('started_at'), ended_at=trial.get('ended_at'),
            duration_seconds=trial.get('duration_seconds'),
            measurement_scope=('train_oob' if summary and summary.get('selection_metric') ==
                               'oob_balanced_accuracy_then_accuracy' else
                               'valid_loss' if summary and summary.get('selection_metric') == 'best_valid_loss'
                               else 'valid' if summary else None)))
    if summary is not None and summary.get('integrity') == 'complete':
        for index in range(summary.get('final_refit_count') or 0):
            training.append(dict(event_id=f'final-refit-{index}', run_id=run_id,
                fold_index=None, trial_index=None, fit_purpose='final_refit',
                state='succeeded', fit_started=True, fit_start_status='known',
                entered_epochs=None, completed_epochs=None, training_batches=None,
                started_at=None, ended_at=None, duration_seconds=None,
                measurement_scope='train_valid_refit'))
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
                                            applicable=lambda row: row['fit_purpose'] == 'candidate' and
                                            summary is not None and
                                            summary.get('selection_metric') == 'best_valid_loss'))
    cost_summary = dict(schema_version=SCHEMA, aggregation_version='task-cost-reduction-v1',
        source=dict(journal_sha256=_sha(journal), agent_db_sha256=_sha(agent_db),
                    journal_wal_sha256=_optional_sha(journal.with_name(journal.name + '-wal')),
                    agent_wal_sha256=_optional_sha(agent_db.with_name(agent_db.name + '-wal')),
                    search_summary_sha256=_optional_sha(run_dir / 'search_summary.json') if run_dir else None,
                    search_trials_sha256=_optional_sha(run_dir / 'search_trials.json') if run_dir else None,
                    run_timeline_sha256=_optional_sha(run_dir / 'search_timeline.json') if run_dir else None,
                    call_snapshot_sha256=hashlib.sha256(json.dumps(calls, sort_keys=True,
                        ensure_ascii=True, allow_nan=False).encode('utf-8')).hexdigest()),
        binding=dict(thread_id=thread_id, task_id=task_id, session_id=session_id,
                     run_id=run_id, owner_id=owner_id, tenant_id=tenant_id),
        call_records=len(calls), llm_request_records=len(requests),
        training_trial_records=len(training), search_summary=summary,
        metrics=coverage, report_status='partial_budget_settlement_unverified',
        warnings=['旧 journal 状态不证明实际物理发送；未知用量未按零计算。',
                  '父子时间段重叠，不能直接相加；货币、GPU 秒不可用。'])
    markdown = (f'# 任务成本报告\n\n任务：{task_id or "历史未绑定"}；Session：{session_id}；'
                f'Run：{run_id or "无"}。\n\n'
                f'LLM 记录 {len(requests)} 条；训练 trial 记录 {len(training)} 条。'
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
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    export_report(args.output, journal=args.journal, agent_db=args.agent_db,
                  thread_id=args.thread_id, session_id=args.session_id,
                  task_id=args.task_id, owner_id=args.owner_id, tenant_id=args.tenant_id,
                  run_id=args.run_id, run_dir=args.run_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

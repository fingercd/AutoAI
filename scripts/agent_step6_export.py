"""Export step-six raw request and Run span facts without prompts or credentials.

Usage: python -m scripts.agent_step6_export --journal PATH --thread-id ID
       [--run-dir PATH] --output PATH
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .task_cost_report import call_rows


def export(*, journal_path: Path, thread_id: str, run_dir: Path | None = None) -> dict:
    if not journal_path.is_file():
        raise FileNotFoundError(journal_path)
    calls, monitor_waits = call_rows(journal_path, thread_id)
    requests = [dict(thread_id=thread_id, task_id=row['task_id'],
                     session_id=row['session_id'], run_id=row['run_id'],
                     call_id=row['id'], operation_id=row['operation_id'],
                     retry_index=row['attempt_index'], phase=row['phase'] or row['name'],
                     status=row['status'], request_started_at_utc=row['request_started_at_utc'],
                     response_ended_at_utc=row['response_ended_at_utc'],
                     request_duration_seconds=row['request_duration_seconds'],
                     measurement_version=row['measurement_version'] or 'legacy-wrapper-unknown',
                     model_id_sha256=row['model_id_sha256'], model_id=row['model_id'],
                     protocol=row['protocol'], request_outcome=row['request_outcome'],
                     input_tokens=row['input_tokens'], output_tokens=row['output_tokens'],
                     total_tokens=row['total_tokens'], token_status=row['token_status'] or 'unknown',
                     error_code=row['error_code'])
                for row in calls if row['kind'] == 'llm']
    monitor_waits = [dict(thread_id=thread_id, **row) for row in monitor_waits]
    timeline = None
    if run_dir is not None:
        path = run_dir / 'search_timeline.json'
        if not path.is_file():
            raise FileNotFoundError(path)
        timeline = json.loads(path.read_text(encoding='utf-8'))
    return dict(schema_version='step6-raw-measurement-export-v1', thread_id=thread_id,
                llm_requests=requests, monitor_waits=monitor_waits, run_timeline=timeline,
                clock_note='UTC orders events; durations use each process monotonic clock. '
                           'Parent and child durations overlap and must not be summed.')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--thread-id', required=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    body = export(journal_path=args.journal, thread_id=args.thread_id,
                  run_dir=args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, ensure_ascii=False, indent=2,
                                      allow_nan=False), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

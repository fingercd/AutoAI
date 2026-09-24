"""Export step-six raw request and Run span facts without prompts or credentials.

Usage: python -m scripts.agent_step6_export --journal PATH --thread-id ID
       [--run-dir PATH] --output PATH
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_poc.orchestration.persistence import CallJournal


def export(*, journal_path: Path, thread_id: str, run_dir: Path | None = None) -> dict:
    if not journal_path.is_file():
        raise FileNotFoundError(journal_path)
    journal = CallJournal(journal_path, thread_id)
    try:
        requests = journal.export_llm_requests()
        monitor_waits = journal.export_monitor_waits()
    finally:
        journal.close()
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

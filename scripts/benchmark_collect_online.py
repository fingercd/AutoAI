#!/usr/bin/env python3
"""Node2 SwanLab bridge: one independent SDK process per task.

Run with --root <suite-root>. --metadata-check performs only online logging
acceptance, with no classifier fit, evaluation, or LLM request.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from backend.app.benchmark_tracking import (OnlineJournal, SwanLabBackend, TaskCollector,
    atomic_json, read_json, OnlineUnavailable)


def run_task(journal, task_id, project, workspace):
    directory = journal.task_dir(task_id)
    with (directory / 'consumer.lock').open('a+b') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return
        if (directory / 'verified.json').exists(): return
        task = read_json(directory / 'task.json')
        backend = SwanLabBackend(directory, task, project, workspace)
        collector = TaskCollector(directory, backend)
        while True:
            try:
                if backend.run is None: backend.open()
                collector.process_once()
                if collector.finished: return
            except Exception as exc:
                prior = read_json(directory / 'status.json', {})
                collector.offline(exc)
                if prior.get('error_class') != type(exc).__name__:
                    print(f'SwanLab outbox retained; task={task_id}; error={type(exc).__name__}', flush=True)
            time.sleep(5)


class Manager:
    def __init__(self, journal, project, workspace):
        self.journal, self.project, self.workspace = journal, project, workspace
        self.children = {}

    def tick(self):
        online = True
        active = 0
        for directory in sorted(self.journal.tasks.iterdir()):
            task = read_json(directory / 'task.json')
            if task is None or (directory / 'verified.json').exists(): continue
            active += 1
            child = self.children.get(directory.name)
            if child is None or child.poll() is not None:
                output = (directory / 'collector.log').open('a', encoding='utf-8')
                try:
                    self.children[directory.name] = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                        '--root', str(self.journal.root), '--task', task['task_id'], '--project', self.project,
                        '--workspace', self.workspace], stdout=output, stderr=subprocess.STDOUT)
                finally: output.close()
            status = read_json(directory / 'status.json', {})
            if status.get('online') is not True or not 0 <= time.time() - status.get('api_verified_at', 0) <= 30:
                online = False
        atomic_json(self.journal.directory / 'heartbeat.json', dict(updated_at=time.time(),
            online=online, active_tasks=active, pid=os.getpid(), node=os.uname().nodename))


def metadata_check(journal, manager, timeout):
    task_id = 'metadata-bridge-acceptance-20261010'
    journal.start_task(task_id, dict(metadata_only=True, classification_experiment=False,
        purpose='SwanLab persistent-outbox online bridge acceptance'))
    deadline = time.monotonic() + timeout
    emitted = False
    while time.monotonic() < deadline:
        manager.tick()
        if not emitted:
            try: journal.ensure_online(task_id, timeout=0)
            except OnlineUnavailable: pass
            else:
                journal.emit(task_id, 'bridge.checked', dict(upload_verified=1, model_id='metadata', budget=0))
                journal.finish_task(task_id, dict(metadata_only=1, online_bridge_checks=1, classifier_fits=0, lm_requests=0))
                emitted = True
        try:
            result = journal.wait_verified(task_id, timeout=0)
        except RuntimeError: pass
        else:
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return 0
        time.sleep(1)
    raise RuntimeError('Online metadata acceptance timed out; existing task and outbox are retained')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task')
    parser.add_argument('--project', default='AutoAI-benchmark-v3')
    parser.add_argument('--workspace', default='ryfrcd693')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--metadata-check', action='store_true')
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    journal = OnlineJournal(args.root)
    if args.task:
        run_task(journal, args.task, args.project, args.workspace)
        return
    with (journal.directory / 'manager.lock').open('a+b') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('An online collector already owns this root') from None
        manager = Manager(journal, args.project, args.workspace)
        if args.metadata_check: return metadata_check(journal, manager, args.timeout)
        while True:
            manager.tick()
            if args.once: return
            time.sleep(5)


if __name__ == '__main__':
    main()

"""Durable tracking tests use a fake cloud; no fit, LLM or network access."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest
from backend.app.benchmark_tracking import (OnlineJournal, OnlineUnavailable, OnlineVerificationFailed,
    TaskCollector, SwanLabBackend, atomic_json, read_json, event_metrics)


class Cloud:
    def __init__(self):
        self.summary = {}; self.state = 'RUNNING'; self.accept = True
        self.calls = []; self.fail = False
        self.ref = dict(run_id='id', path='account/project/id', url='https://swanlab.cn/account/project/runs/id')
    def log(self, values, sequence):
        self.calls.append((sequence, values))
        if self.accept:
            self.summary.update({key: dict(value=value, step=sequence) for key, value in values.items()})
    def query(self):
        if self.fail: raise ConnectionError('fake offline')
        return dict(state=self.state, summary=self.summary, cloud_ref=self.ref)
    def finish(self): self.state = 'FINISHED'


def started(tmp_path):
    journal = OnlineJournal(tmp_path)
    journal.start_task('task', {'seed': 42})
    cloud = Cloud(); collector = TaskCollector(journal.task_dir('task'), cloud)
    return journal, cloud, collector


def heartbeat(journal, **extra):
    atomic_json(journal.directory / 'heartbeat.json', dict(online=True, updated_at=time.time(), **extra))


def test_start_and_finish_are_idempotent_and_frozen(tmp_path):
    journal, cloud, collector = started(tmp_path)
    first = journal.start_task('task', {'seed': 42})
    assert first == journal.start_task('task', {'seed': 42})
    with pytest.raises(ValueError): journal.start_task('task', {'seed': 7})
    final = journal.finish_task('task', {'fits': 3})
    assert final == journal.finish_task('task', {'fits': 3})
    with pytest.raises(ValueError): journal.finish_task('task', {'fits': 4})
    with pytest.raises(ValueError): journal.emit('task', 'fit.completed', {'fits': 1})


@pytest.mark.parametrize('data', [{'headers': {'Authorization': 'forbidden'}}, {'token': 'forbidden'},
                                  {'duration': float('nan')}, {'path': Path('fixture')}])
def test_unsafe_and_non_json_events_never_persist(tmp_path, data):
    journal, cloud, collector = started(tmp_path)
    with pytest.raises(ValueError): journal.emit('task', 'lm.request', data)
    assert len(list((journal.task_dir('task') / 'events').glob('*.json'))) == 1


def test_concurrent_producers_keep_unique_ordered_events(tmp_path):
    journal, cloud, collector = started(tmp_path)
    with ThreadPoolExecutor(max_workers=6) as pool:
        ids = list(pool.map(lambda i: journal.emit('task', 'fit.completed', {'fits': i}), range(20)))
    events = [read_json(path) for path in sorted((journal.task_dir('task') / 'events').glob('*.json'))]
    assert len(set(ids)) == 20
    assert [event['sequence'] for event in events] == list(range(1, 22))


def test_online_gate_requires_cloud_ack_and_fresh_manager(tmp_path):
    journal, cloud, collector = started(tmp_path)
    heartbeat(journal)
    with pytest.raises(OnlineUnavailable): journal.ensure_online('task', timeout=0)
    collector.process_once()
    assert journal.ensure_online('task', timeout=0) == cloud.ref
    atomic_json(journal.directory / 'heartbeat.json', dict(online=True, updated_at=time.time() - 16))
    with pytest.raises(OnlineUnavailable): journal.ensure_online('task', timeout=0)


def test_pending_log_is_not_acked_before_cloud_readback_and_retries_same_step(tmp_path):
    journal, cloud, collector = started(tmp_path)
    cloud.accept = False
    with pytest.raises(OnlineUnavailable): collector.process_once()
    assert not (journal.task_dir('task') / 'acks').exists()
    cloud.accept = True
    collector.process_once()
    assert [step for step, _ in cloud.calls] == [1, 1]
    assert len(list((journal.task_dir('task') / 'acks').glob('*.json'))) == 1


def test_outage_retains_events_and_finish_can_be_written_locally(tmp_path):
    journal, cloud, collector = started(tmp_path)
    collector.process_once(); heartbeat(journal)
    assert journal.ensure_online('task', timeout=0)
    journal.emit('task', 'fit.completed', {'actual_fits': 1})
    cloud.fail = True
    with pytest.raises(ConnectionError) as error: collector.process_once()
    collector.offline(error.value)
    with pytest.raises(OnlineUnavailable): journal.ensure_online('task', timeout=0)
    journal.finish_task('task', {'actual_fits': 1})
    assert len(list((journal.task_dir('task') / 'events').glob('*.json'))) == 3
    with pytest.raises(OnlineVerificationFailed): journal.wait_verified('task', timeout=0)
    cloud.fail = False
    collector.process_once()
    assert journal.wait_verified('task', timeout=0)['verified'] is True


def test_final_numeric_values_and_finished_state_are_both_required(tmp_path):
    journal, cloud, collector = started(tmp_path)
    collector.process_once()
    journal.finish_task('task', {'metrics': {'macro_f1': .75}, 'actual_fits': 3})
    cloud.finish = lambda: None
    with pytest.raises(OnlineUnavailable): collector.process_once()
    assert not (journal.task_dir('task') / 'verified.json').exists()
    cloud.state = 'FINISHED'
    cloud.summary['summary/actual_fits']['value'] = 2
    cloud.accept = False
    with pytest.raises(OnlineUnavailable): collector.process_once()
    cloud.summary['summary/actual_fits']['value'] = 3
    collector.process_once()
    verified = journal.wait_verified('task', timeout=0)
    assert verified['summary']['metrics']['macro_f1'] == .75
    assert 'timing/task_wall_seconds' in verified['verified_numeric_summary']


def test_lm_missing_usage_is_counted_and_known_totals_are_explicit(tmp_path):
    journal, cloud, collector = started(tmp_path)
    data = dict(model_id='qwen35_9b', budget=3, usage=dict(prompt_tokens=10, completion_tokens=None, total_tokens=None))
    journal.emit('task', 'lm.request', data); journal.emit('task', 'lm.request', data)
    collector.process_once()
    prefix = 'model/qwen35_9b/budget/3/lm/request/'
    assert cloud.summary[prefix + 'input_tokens_known_requests']['value'] == 2
    assert cloud.summary[prefix + 'output_tokens_missing_requests']['value'] == 2
    assert cloud.summary[prefix + 'request_count']['value'] == 2
    assert prefix + 'usage/completion_tokens' not in cloud.summary


def test_recovery_after_start_manifest_crash_reuses_original_event(tmp_path):
    journal, cloud, collector = started(tmp_path)
    directory = journal.task_dir('task')
    first = read_json(directory / 'task.json')['start_event_id']
    (directory / 'task.json').unlink()
    assert journal.start_task('task', {'seed': 42}) == first
    assert len(list((directory / 'events').glob('*.json'))) == 1


def test_unknown_cloud_init_never_creates_second_run(tmp_path, monkeypatch):
    journal, cloud, collector = started(tmp_path)
    calls = []
    def initialize(**kwargs):
        calls.append(kwargs)
        raise ConnectionError('unknown init outcome')
    api = SimpleNamespace(run=lambda path: SimpleNamespace(state=''))
    monkeypatch.setitem(sys.modules, 'swanlab', SimpleNamespace(init=initialize, Api=lambda: api))
    task = read_json(journal.task_dir('task') / 'task.json')
    backend = SwanLabBackend(journal.task_dir('task'), task)
    with pytest.raises(ConnectionError): backend.open()
    saved = read_json(backend.binding_path)
    with pytest.raises(OnlineUnavailable): SwanLabBackend(backend.directory, task).open()
    assert len(calls) == 1
    assert read_json(backend.binding_path)['run_id'] == saved['run_id']


def test_existing_cloud_run_resumes_exact_saved_id(tmp_path, monkeypatch):
    journal, cloud, collector = started(tmp_path)
    directory = journal.task_dir('task')
    atomic_json(directory / 'cloud-binding.json', dict(state='bound', run_id='saved-id', path='account/project/saved-id'))
    calls = []
    def initialize(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(mode='online', id='saved-id', url='https://fixture.invalid/saved-id')
    monkeypatch.setitem(sys.modules, 'swanlab', SimpleNamespace(init=initialize,
        Api=lambda: SimpleNamespace(run=lambda path: SimpleNamespace(state='RUNNING'))))
    backend = SwanLabBackend(directory, read_json(directory / 'task.json'))
    backend.open()
    assert calls[0]['id'] == 'saved-id'
    assert calls[0]['resume'] == 'must'


def test_final_numeric_lists_are_verified_without_key_collisions(tmp_path):
    journal, cloud, collector = started(tmp_path)
    collector.process_once()
    summary = {'rows': [{'metric': .5}, {'metric': .9}], 'a/b': 1, 'a_b': 2}
    journal.finish_task('task', summary)
    collector.process_once()
    values = journal.wait_verified('task', timeout=0)['verified_numeric_summary']
    assert values['summary/rows/0/metric'] == .5
    assert values['summary/rows/1/metric'] == .9
    assert values['summary/a~1b'] == 1
    assert values['summary/a_b'] == 2


def test_task_freeze_distinguishes_json_integer_and_boolean(tmp_path):
    journal = OnlineJournal(tmp_path)
    journal.start_task('task', {'setting': 1})
    with pytest.raises(ValueError): journal.start_task('task', {'setting': True})


def test_finished_cloud_cannot_authorize_new_training(tmp_path):
    journal, cloud, collector = started(tmp_path)
    cloud.state = 'FINISHED'
    collector.process_once(); heartbeat(journal)
    with pytest.raises(OnlineUnavailable): journal.ensure_online('task', timeout=0)


def test_fit_counters_deduplicate_runs_and_duration_sum_is_not_wall(tmp_path):
    journal, cloud, collector = started(tmp_path)
    data = dict(run_id='r1', model_id='svm', budget=3, actual_fits=4, started_fits=4, epochs=0, wall_seconds=8)
    journal.emit('task', 'fit.completed', data); journal.emit('task', 'fit.completed', data)
    journal.emit('task', 'fit.completed', {**data, 'run_id': 'r2', 'epochs': 5, 'wall_seconds': 6})
    collector.process_once()
    assert cloud.summary['accounting/fit/actual_fits']['value'] == 8
    assert cloud.summary['accounting/fit/epochs']['value'] == 5
    assert cloud.summary['accounting/fit/fit_duration_sum_seconds']['value'] == 14
    assert 'timing/task_wall_seconds' not in cloud.summary


def test_native_lm_input_output_fields_accumulate_known_and_missing_counts(tmp_path):
    journal, cloud, collector = started(tmp_path)
    journal.emit('task', 'lm.request', dict(input_tokens=20, output_tokens=10, total_tokens=30, token_status='known'))
    journal.emit('task', 'lm.request', dict(input_tokens=None, output_tokens=None, total_tokens=None, token_status='unknown'))
    collector.process_once()
    assert cloud.summary['accounting/lm/input_tokens_known_tokens_sum']['value'] == 20
    assert cloud.summary['accounting/lm/output_tokens_known_tokens_sum']['value'] == 10
    assert cloud.summary['accounting/lm/total_tokens_missing_requests']['value'] == 1
    assert cloud.summary['accounting/lm/request_count']['value'] == 2
    assert cloud.summary['accounting/lm/usage_unknown_requests']['value'] == 1

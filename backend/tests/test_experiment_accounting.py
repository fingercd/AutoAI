"""Offline timing and physical-work checks using a counter-only toy fit."""
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import time

import pytest

from backend.app import experiment_accounting as module
from backend.app import benchmark_tracking
from backend.app.search_accounting import SearchAccounting


class FakeJournal:
    calls = []
    failure = None
    def __init__(self, root):
        self.root = root
    def _call(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        if name == self.failure:
            raise RuntimeError('fake-' + name + '-failure')
    def start_task(self, *args, **kwargs): return self._call('start_task', *args, **kwargs)
    def ensure_online(self, *args, **kwargs): return self._call('ensure_online', *args, **kwargs)
    def emit(self, *args, **kwargs): return self._call('emit', *args, **kwargs)
    def finish_task(self, *args, **kwargs): return self._call('finish_task', *args, **kwargs)
    def wait_verified(self, *args, **kwargs): return self._call('wait_verified', *args, **kwargs)


@pytest.fixture(autouse=True)
def offline_journal(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTOAI_BENCHMARK_ROOT', str(tmp_path / 'online-fixture'))
    FakeJournal.calls, FakeJournal.failure = [], None
    monkeypatch.setattr(benchmark_tracking, 'OnlineJournal', FakeJournal)
    monkeypatch.setattr(module, 'start_cuda_clock', lambda: None)


def summaries(directory):
    return (json.loads((directory / 'telemetry/experiment_summary.json').read_text()),
            json.loads((directory / 'telemetry/experiment_timeline.json').read_text()))


class ToyModel:
    def __init__(self): self.calls = 0
    def fit(self): self.calls += 1; return self


@pytest.mark.parametrize('explicit_id', [True, False])
def test_normal_entrypoint_optional_run_id_output_signature(tmp_path, explicit_id):
    @module.record_training
    def run(data_path, config_data=None, run_id=None, *, output_dir=None):
        assert module.account() is not None
        model = ToyModel()
        module.measured_model_fit(model.fit)
        return run_id, Path(output_dir), model.calls
    options = dict(output_dir=tmp_path / 'run')
    if explicit_id:
        options['run_id'] = 'ordinary-id'
    run_id, directory, count = run('unused.csv', {'execution_device': 'cpu'}, **options)
    summary, timeline = summaries(directory)
    assert summary['run_id'] == run_id
    assert count == 1
    assert summary['model_fits_completed'] == 1
    assert summary['model_fits_started'] == 1
    assert summary['status'] == 'succeeded'
    assert summary['wall_seconds'] >= max(row['duration_seconds'] for row in timeline['spans'])
    assert module.account() is None
    assert [row[0] for row in FakeJournal.calls] == ['start_task', 'ensure_online', 'emit']


def test_failed_fit_retains_root_duration_counters_and_releases_active_account(tmp_path):
    @module.record_training
    def run(data_path, config_data=None, run_id=None, *, output_dir=None):
        def counter_failure():
            time.sleep(.002)
            raise ValueError('toy-counter-failure')
        module.measured_model_fit(counter_failure)
    with pytest.raises(ValueError, match='toy-counter-failure'):
        run('unused', {}, 'failed-run', output_dir=tmp_path / 'run')
    summary, timeline = summaries(tmp_path / 'run')
    assert summary['status'] == 'failed'
    assert summary['model_fits_started'] == 1
    assert summary['model_fits_completed'] == 0
    root = [row for row in timeline['spans'] if row['stage'] == 'total_training'][0]
    assert root['status'] == 'failed'
    assert root['duration_seconds'] > 0
    assert module.ACTIVE_ACCOUNT.get() is None


@pytest.mark.parametrize('failure', ['ensure_online', 'emit'])
def test_online_gate_or_upload_failure_keeps_trace_and_releases_account(tmp_path, failure):
    FakeJournal.failure = failure
    invoked = []
    @module.record_training
    def run(data_path, config_data=None, run_id=None, *, output_dir=None):
        invoked.append(True)
        return 'completed'
    with pytest.raises(RuntimeError, match='fake-' + failure):
        run('unused', {}, 'journal-failure', output_dir=tmp_path / 'run')
    summary, timeline = summaries(tmp_path / 'run')
    assert summary['wall_seconds'] >= 0
    assert module.account() is None
    assert invoked == ([] if failure == 'ensure_online' else [True])
    if failure == 'emit':
        assert summary.get('tracking_status') == 'failed'


def test_initial_cuda_sync_failure_has_failed_root_trace(tmp_path, monkeypatch):
    from backend.app import execution_runtime
    def unavailable(*args): raise RuntimeError('mock-sync-failure')
    monkeypatch.setattr(execution_runtime, 'synchronize_device', unavailable)
    @module.record_training
    def run(data_path, config_data=None, run_id=None, *, output_dir=None):
        pytest.fail('sync gate failed before body')
    with pytest.raises(RuntimeError, match='mock-sync-failure'):
        run('unused', {'execution_device': 'cuda'}, 'sync-failure', output_dir=tmp_path / 'run')
    summary, timeline = summaries(tmp_path / 'run')
    assert summary['status'] == 'failed'
    assert module.account() is None
    root = [row for row in timeline['spans'] if row['stage'] == 'total_training']
    assert len(root) == 1
    assert root[0]['status'] == 'failed'
    assert root[0]['duration_seconds'] >= 0


def test_exit_cuda_sync_failure_does_not_claim_success(tmp_path, monkeypatch):
    from backend.app import execution_runtime
    calls = []
    def sync(*args):
        calls.append(True)
        if len(calls) == 2: raise RuntimeError('mock-exit-sync-failure')
    monkeypatch.setattr(execution_runtime, 'synchronize_device', sync)
    @module.record_training
    def run(data_path, config_data=None, run_id=None, *, output_dir=None): return 'body-complete'
    with pytest.raises(RuntimeError, match='mock-exit-sync-failure'):
        run('unused', {'execution_device': 'cuda'}, 'exit-sync-failure', output_dir=tmp_path / 'run')
    summary, timeline = summaries(tmp_path / 'run')
    assert summary['status'] == 'failed'
    assert module.account() is None
    assert timeline['spans'][0]['status'] == 'failed'


def search_plan(deep=True):
    return dict(plan_digest='d'*64, mode='bounded', effective_search=True,
                selection_metric='valid_balanced_accuracy', effective_trials=2,
                fit_strategy='train_best_epoch_no_refit' if deep else 'train_valid_refit')


def test_cached_historical_epochs_and_batches_are_not_physical_work(tmp_path):
    ledger = SearchAccounting(tmp_path, search_plan(), 'r')
    for index, cached in enumerate((True, False)):
        with ledger.trial(1, dict(index=index, params={}, params_digest=str(index)), parent_span='p') as row:
            row.update(actual_epochs=5, training_batches=20, cached=cached)
            if cached:
                row.update(source_epochs=5, executed_epochs=0, executed_training_batches=0)
    result = ledger.finish(selected=[dict(fold_index=1, trial_index=1)], refit_count=0)
    assert result['actual_epochs'] == 5
    assert result['training_batches'] == 20
    assert result['candidate_fit_count'] == 1
    assert result['candidate_evaluation_count'] == 2
    assert result['cache_hit_count'] == 1
    assert result['total_fit_count'] == 1


def test_incomplete_search_reports_physical_fits_excluding_cache(tmp_path):
    ledger = SearchAccounting(tmp_path, search_plan(), 'r')
    with ledger.trial(1, dict(index=0, params={}, params_digest='0'), parent_span='p') as row:
        row.update(actual_epochs=5, training_batches=20, cached=True, executed_epochs=0, executed_training_batches=0)
    with pytest.raises(ValueError):
        with ledger.trial(1, dict(index=1, params={}, params_digest='1'), parent_span='p') as row:
            row.update(actual_epochs=1, training_batches=4, cached=False)
            raise ValueError('counter-failure')
    summary = json.loads((tmp_path / 'search_summary.json').read_text())
    assert summary['candidate_fit_count'] == 1
    assert summary['actual_epochs_known'] == 1
    assert summary['training_batches_known'] == 4


def test_publication_tracking_failure_preserves_manifest_hash(tmp_path):
    directory = tmp_path / 'run'
    account = module.ExperimentAccount(directory / 'telemetry', 'r', {})
    account.status = 'succeeded'
    account.flush()
    manifest = directory / 'manifest.json'
    manifest.write_text(json.dumps({'artifacts': ['metrics.json'], 'run_id': 'r'}))
    before = hashlib.sha256(manifest.read_bytes()).hexdigest()
    FakeJournal.failure = 'wait_verified'
    with pytest.raises(RuntimeError, match='wait_verified'):
        module.finish_publication(directory, {}, time.monotonic())
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == before
    summary, _ = summaries(directory)
    assert summary['publication_seconds'] >= 0
    assert summary['total_wall_seconds'] >= summary['wall_seconds']
    assert summary.get('tracking_status') == 'failed'


def test_base_hpo_keeps_frozen_default_first_and_stable_source():
    from backend.app.benchmark_protocol import build_benchmark_search_plan
    from backend.app.search_policy import BASELINES, _digest_source
    baseline = BASELINES['logistic_regression']
    kwargs = dict(model_id='logistic_regression', baseline=baseline, mode='bounded',
        train_count=20, feature_count=10, class_count=2, dataset_sha256='a'*64,
        evaluation_plan_digest='b'*64, normalization='zscore', class_balance='none', seed=42,
        architecture_version='fixture-arch', processing_policy_version='fixture-processing')
    one = build_benchmark_search_plan(**kwargs, max_trials=1)
    three = build_benchmark_search_plan(**kwargs, max_trials=3)
    assert one['candidates'][0] == three['candidates'][0]
    assert one['candidates'][0]['params'] == baseline
    assert one['source_digest'] == three['source_digest'] == _digest_source()
    assert three['effective_trials'] == 3
    assert [row['params']['logistic_c'] for row in three['candidates']] == [1., .1, 10.]


def test_ordinary_entrypoint_default_output_directory_and_signature(tmp_path, monkeypatch):
    from backend.app import paths
    monkeypatch.setattr(paths, 'RUNS_DIR', tmp_path / 'runs')
    def implementation(data_path, config_data=None, run_id=None, *, output_dir=None):
        return run_id, Path(output_dir)
    wrapped = module.record_training(implementation)
    assert inspect.signature(wrapped) == inspect.signature(implementation)
    run_id, directory = wrapped('unused')
    assert directory == tmp_path / 'runs' / run_id
    assert summaries(directory)[0]['status'] == 'succeeded'
    assert module.account() is None


def test_root_clock_and_physical_accounting_include_budget_cache_hit(tmp_path, monkeypatch):
    import torch
    from backend.app.benchmark_cache import cache_deep_fit, active_cache
    from backend.app.training import TrainConfig
    root = tmp_path / 'suite'
    approval = root / 'approval'
    approval.mkdir(parents=True)
    approved = Path(__file__).resolve().parents[2] / 'work/benchmark-20261010/approval/protocol.json'
    shutil.copyfile(approved, approval / 'protocol.json')
    protocol = json.loads((approval / 'protocol.json').read_text())
    monkeypatch.setenv('AUTOAI_BENCHMARK_ROOT', str(root))
    context = dict(suite_root=str(root), bundle_id='toy:round01', task_id='fixture', round_id='01',
                   source_digest='a'*64, protocol_digest=protocol['approval_digest'])
    calls = []
    @cache_deep_fit
    @module.measured_deep_fit
    def counter_fit(*, config, trial_fact=None):
        calls.append(True)
        trial_fact.update(actual_epochs=2, training_batches=6)
        return torch.nn.Linear(2, 1), [{'best_valid_loss': 2.}, {'best_valid_loss': 1.}]
    @module.record_training
    def run(data_path, config_data=None, run_id=None, *, output_dir=None):
        active_cache().normalizer = {'mode': 'none'}
        fact = {}
        counter_fit(config=TrainConfig(), trial_fact=fact)
        return fact
    raw = dict(benchmark_context=context, evaluation_plan_digest='b'*64,
               benchmark_dataset_sha256='c'*64, max_trials=1, execution_device='cpu', model_type='cnn1d')
    first = run('unused', raw, 'budget-one', output_dir=tmp_path / 'one')
    second = run('unused', {**raw, 'max_trials': 3}, 'budget-three', output_dir=tmp_path / 'three')
    first_summary, first_timeline = summaries(tmp_path / 'one')
    second_summary, second_timeline = summaries(tmp_path / 'three')
    assert calls == [True]
    assert first_summary['model_fits_completed'] == 1
    assert first_summary['executed_epochs'] == 2
    assert first_summary['training_batches'] == 6
    assert second_summary['model_fits_started'] == second_summary['model_fits_completed'] == 0
    assert second_summary['executed_epochs'] == second_summary['training_batches'] == 0
    assert second['source_epochs'] == 2 and second['executed_epochs'] == 0
    assert second_summary['wall_seconds'] > 0
    assert any(row['stage'] == 'total_training' for row in second_timeline['spans'])
    record = first_summary['default_weight_ref']
    assert record['kind'] == 'deep_candidate'
    assert record['source_run_id'] == 'budget-one'
    import pickle
    with (root / 'weights' / record['relative_path']).open('rb') as handle:
        payload = pickle.load(handle)['payload']
    assert payload['normalizer'] == {'mode': 'none'}
    assert module.account() is None


def test_journal_finish_failure_is_durable_and_manifest_unchanged(tmp_path):
    directory = tmp_path / 'run'
    current = module.ExperimentAccount(directory / 'telemetry', 'r', {})
    current.status = 'succeeded'
    current.flush()
    manifest = directory / 'manifest.json'
    manifest.write_text('{"run_id":"r","artifacts":[]}')
    before = manifest.read_bytes()
    FakeJournal.failure = 'finish_task'
    with pytest.raises(RuntimeError, match='finish_task'):
        module.finish_publication(directory, {}, time.monotonic())
    assert manifest.read_bytes() == before
    assert summaries(directory)[0].get('tracking_status') == 'failed'

def test_cuda_event_interval_is_separate_from_wall_time(tmp_path, monkeypatch):
    recorded = []
    class Event:
        def record(self): recorded.append(True)
        def elapsed_time(self, other): return 12.5
    current = module.ExperimentAccount(tmp_path, 'cuda-clock', {'execution_device':'cuda'})
    monkeypatch.setattr(current, 'synchronize', lambda: None)
    monkeypatch.setattr(module, 'start_cuda_clock', lambda: (Event(), Event()))
    with current.span('gpu-operation'):
        pass
    row = current.spans[-1]
    assert row['status'] == 'succeeded'
    assert row['gpu_elapsed_seconds'] == .0125
    assert row['duration_seconds'] >= 0
    assert 'kernel active time' in row['gpu_timing_scope']
    assert len(recorded) == 1

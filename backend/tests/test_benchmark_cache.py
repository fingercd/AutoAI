"""Offline cache contract checks: toy objects and serialization, no learner training."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pytest
import torch

from backend.app import benchmark_cache as module
from backend.app.benchmark_cache import FittedCache, fitted_cache_scope, cache_deep_fit, cache_final_fit
from backend.app.benchmark_resources import WeightStore, CapacityBlocked
from backend.app.evaluation_plan import digest
from backend.app.training import TrainConfig, TraditionalSelection


class ToyModel:
    def __init__(self):
        self.fit_calls = 0
    def fit(self, *args):
        self.fit_calls += 1
        return self


@pytest.fixture
def binding(tmp_path, monkeypatch):
    root = tmp_path / 'suite'
    approval = root / 'approval'
    approval.mkdir(parents=True)
    approved = Path(__file__).resolve().parent / 'fixtures/benchmark_protocol_v3.json'
    shutil.copyfile(approved, approval / 'protocol.json')
    protocol = json.loads((approval / 'protocol.json').read_text())
    monkeypatch.setenv('AUTOAI_BENCHMARK_ROOT', str(root))
    context = dict(suite_root=str(root), bundle_id='fixture:toy:round01', task_id='fixture',
                   round_id='01', protocol_digest=protocol['approval_digest'], source_digest='a' * 64)
    raw = dict(evaluation_plan_digest='b' * 64, benchmark_dataset_sha256='c' * 64,
               _cache_run_id='source-run-budget1')
    return context, raw


def test_budget_one_and_three_share_default_and_normalizer_binding(binding):
    context, raw = binding
    one = TrainConfig(cpu_threads=4)
    one.search_mode, one.max_trials = 'fixed', 1
    cache = FittedCache(context, raw)
    normalizer = {'mode': 'zscore', 'mean': np.array([1., 2.]), 'std': np.array([3., 4.])}
    cache.normalizer = normalizer
    cache.save('traditional_candidate', one, {'model': ToyModel(), 'valid_eval': {'balanced_accuracy': .5}})
    three = TrainConfig(**{k: v for k, v in one.__dict__.items() if k not in ('search_mode', 'max_trials')})
    three.search_mode, three.max_trials = 'bounded', 3
    payload = FittedCache(context, raw).load('traditional_candidate', three)
    assert isinstance(payload['model'], ToyModel)
    np.testing.assert_array_equal(payload['normalizer']['mean'], normalizer['mean'])
    record = cache.latest_artifact
    assert record['kind'] == 'traditional_candidate'
    assert record['source_run_id'] == 'source-run-budget1'
    assert record['key'] == cache.key('traditional_candidate', three)
    assert cache.load('traditional_final', three) is None


@pytest.mark.parametrize('field,value', [('learning_rate', .009), ('seed', 43),
    ('execution_device', 'cuda'), ('cpu_threads', 1), ('normalization', 'none')])
def test_changed_fit_configuration_never_reuses(binding, field, value):
    context, raw = binding
    config = TrainConfig(cpu_threads=4)
    cache = FittedCache(context, raw)
    cache.save('traditional_candidate', config, {'model': ToyModel()})
    changed = TrainConfig(**{**config.__dict__, field: value})
    assert cache.load('traditional_candidate', changed) is None


@pytest.mark.parametrize('field', ['evaluation_plan_digest', 'benchmark_dataset_sha256', 'source_digest', 'bundle_id'])
def test_changed_data_plan_source_or_pair_binding_never_reuses(binding, field):
    context, raw = binding
    cache = FittedCache(context, raw)
    config = TrainConfig()
    cache.save('traditional_candidate', config, {'model': ToyModel()})
    changed_context, changed_raw = deepcopy(context), deepcopy(raw)
    target = changed_context if field in context else changed_raw
    target[field] = 'd' * 64 if field != 'bundle_id' else 'other-pair'
    assert FittedCache(changed_context, changed_raw).load('traditional_candidate', config) is None


def test_model_blob_tamper_refused_before_unpickle(binding, monkeypatch):
    cache = FittedCache(*binding)
    config = TrainConfig()
    cache.save('traditional_candidate', config, {'model': ToyModel()})
    file = cache.store.root / cache.latest_artifact['relative_path']
    file.write_bytes(file.read_bytes() + b'tampered')
    unpickled = []
    monkeypatch.setattr(module.pickle, 'load', lambda handle: unpickled.append(True))
    with pytest.raises(ValueError, match='changed'):
        cache.load('traditional_candidate', config)
    assert unpickled == []


def test_tensor_checkpoint_tamper_refused_before_unpickle_or_torch_load(binding, monkeypatch):
    cache = FittedCache(*binding)
    config = TrainConfig()
    cache.save('deep_candidate', config, {'model': torch.nn.Linear(2, 1),
        'history': [{'best_valid_loss': 1.}], 'fit_seconds': 2., 'actual_epochs': 1, 'training_batches': 3})
    payload = cache.load('deep_candidate', config)
    path = cache.store.root / payload['checkpoint']['relative_path']
    path.write_bytes(path.read_bytes() + b'tampered')
    unpickled, loaded = [], []
    actual_unpickle = module.pickle.load
    monkeypatch.setattr(module.pickle, 'load', lambda *args, **kwargs: unpickled.append(True) or actual_unpickle(*args, **kwargs))
    actual_torch_load = torch.load
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: loaded.append(True) or actual_torch_load(*args, **kwargs))
    @cache_deep_fit
    def fit(*, config, trial_fact=None):
        pytest.fail('tampered cached fit cannot train again')
    token = module.ACTIVE_CACHE.set(cache)
    try:
        with pytest.raises(ValueError, match='changed'):
            fit(config=config)
    finally:
        module.ACTIVE_CACHE.reset(token)
    assert loaded == []
    assert unpickled == []


def test_cache_storage_limit_and_failed_temporary_cleanup(binding):
    cache = FittedCache(*binding)
    cache.store = WeightStore(cache.store.root, limit_bytes=500, high=400, low=350, retained_limit=100)
    with pytest.raises(CapacityBlocked):
        cache.save('traditional_candidate', TrainConfig(), {'blob': b'x' * 600})
    snapshot = cache.store.snapshot()
    assert snapshot['actual_bytes'] == 0
    assert snapshot['active_reservations'] == 0
    assert not list(cache.index.glob('*.json'))


def test_deep_nonfinite_parameters_rejected_before_any_artifact_write(binding):
    cache = FittedCache(*binding)
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(float('nan'))
    with pytest.raises(ValueError, match='nonfinite'):
        cache.save('deep_candidate', TrainConfig(), {'model': model})
    assert cache.store.snapshot()['actual_bytes'] == 0
    assert cache.store.snapshot()['active_reservations'] == 0


def test_deep_reuse_restores_cpu_parameters_and_keeps_physical_epochs_zero(binding):
    config = TrainConfig()
    calls = []
    model = torch.nn.Linear(2, 1)
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    @cache_deep_fit
    def fit(*, config, trial_fact=None):
        calls.append('counter-only')
        trial_fact.update(actual_epochs=2, training_batches=6)
        return model, [{'best_valid_loss': 2.}, {'best_valid_loss': 1.}]
    with fitted_cache_scope(*binding) as cache:
        cache.normalizer = {'mode': 'none'}
        first_fact = {}
        first, history = fit(config=config, trial_fact=first_fact)
        with torch.no_grad():
            model.weight.fill_(99)
        hit_fact = {}
        restored, reused_history = fit(config=config, trial_fact=hit_fact)
        assert calls == ['counter-only']
        assert hit_fact['source_epochs'] == 2
        assert hit_fact['executed_epochs'] == 0
        assert hit_fact['executed_training_batches'] == 0
        assert reused_history == history
        for name, value in initial.items():
            torch.testing.assert_close(restored.state_dict()[name], value)
        assert all(t.device.type == 'cpu' for t in restored.state_dict().values())
    assert module.active_cache() is None


def test_deep_cache_gpu_restore_requests_cuda_with_cpu_fixture_tensors(binding, monkeypatch):
    from backend.app import execution_runtime
    config = TrainConfig(execution_device='cuda')
    cache = FittedCache(*binding)
    cache.save('deep_candidate', config, {'model': torch.nn.Linear(2, 1),
        'history': [{'best_valid_loss': 1.}], 'fit_seconds': 2., 'actual_epochs': 1, 'training_batches': 3})
    actual_load = execution_runtime.load_deep_checkpoint
    calls = []
    def restore_cpu_fixture(model, path, *, execution_device):
        calls.append(execution_device)
        return actual_load(model, path, execution_device='cpu')
    monkeypatch.setattr(execution_runtime, 'load_deep_checkpoint', restore_cpu_fixture)
    @cache_deep_fit
    def fit(*, config, trial_fact=None):
        pytest.fail('cache hit must not fit')
    token = module.ACTIVE_CACHE.set(cache)
    try:
        model, _ = fit(config=config, trial_fact={})
    finally:
        module.ACTIVE_CACHE.reset(token)
    assert calls == ['cuda']
    assert all(t.device.type == 'cpu' for t in model.state_dict().values())


def test_final_refit_selected_config_and_selection_wrapper_share_identity(binding):
    config = TrainConfig(model_type='logistic_regression')
    selection = TraditionalSelection(config, .5, .5, [], ToyModel(), {})
    calls = []
    @cache_final_fit
    def refit(*, selected_config):
        calls.append(True)
        return ToyModel(), {'mode': 'none'}, np.array([0, 1, 2])
    with fitted_cache_scope(*binding) as cache:
        expected = refit(selected_config=selection)
        restored = refit(selected_config=config)
        assert calls == [True]
        assert cache.final_cache_hit is True
        assert restored[1] == expected[1]
        np.testing.assert_array_equal(restored[2], expected[2])
        assert cache.latest_artifact['kind'] == 'traditional_final'


def test_deep_invalid_evidence_is_rejected_before_saving_model(binding):
    @cache_deep_fit
    def fit(*, config, trial_fact=None):
        trial_fact.update(actual_epochs=1, training_batches=1)
        return torch.nn.Linear(2, 1), [{'best_valid_loss': float('inf')}]
    with fitted_cache_scope(*binding) as cache:
        with pytest.raises(ValueError, match='finite validation'):
            fit(config=TrainConfig(), trial_fact={})
        assert cache.store.snapshot()['actual_bytes'] == 0



def test_checkpoint_single_bounded_write_matches_torch_and_preserves_source(binding, monkeypatch):
    from backend.app.benchmark_resources import _BoundedFile
    cache = FittedCache(*binding)
    config = TrainConfig()
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.BatchNorm1d(3))
    model.eval()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    devices = {name: tensor.device for name, tensor in model.state_dict().items()}
    with io.BytesIO() as buffer:
        torch.save(model.state_dict(), buffer)
        reference_bytes = buffer.getvalue()

    cpu_calls = []
    original_cpu = torch.nn.Module.cpu
    def cpu(instance):
        cpu_calls.append(instance)
        return original_cpu(instance)
    monkeypatch.setattr(torch.nn.Module, 'cpu', cpu)
    reservation_paths, writes = {}, []
    original_reserve = cache.store.reserve
    def reserve(relative_path, *args, **kwargs):
        reservation = original_reserve(relative_path, *args, **kwargs)
        reservation_paths[reservation.token] = relative_path
        return reservation
    original_write = _BoundedFile.write
    def write(handle, data):
        writes.append((reservation_paths[handle.reservation.token], memoryview(data).nbytes))
        return original_write(handle, data)
    monkeypatch.setattr(cache.store, 'reserve', reserve)
    monkeypatch.setattr(_BoundedFile, 'write', write)
    cache.save('deep_candidate', config, {'model': model})
    loaded = cache.load('deep_candidate', config)
    checkpoint = cache.store.root / loaded['checkpoint']['relative_path']
    checkpoint_writes = [(path, size) for path, size in writes if path.startswith('checkpoints/')]
    assert checkpoint_writes == [(loaded['checkpoint']['relative_path'], len(reference_bytes))]
    assert checkpoint.read_bytes() == reference_bytes
    tensors = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert model.training is False
    assert len(cpu_calls) == 1
    assert cpu_calls[0] is not model
    assert loaded['model'] is not model
    for name, expected in before.items():
        assert model.state_dict()[name].device == devices[name]
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)
        torch.testing.assert_close(loaded['model'].state_dict()[name], expected, rtol=0, atol=0)
        torch.testing.assert_close(tensors[name], expected, rtol=0, atol=0)


@pytest.mark.parametrize('failure_stage', ['reservation', 'payload_write'])
def test_checkpoint_quota_failure_cleans_own_temporary_and_preserves_protected_assets(
        binding, monkeypatch, failure_stage):
    cache = FittedCache(*binding)
    cache.store = WeightStore(cache.store.root, limit_bytes=2000, high=1500, low=1000,
                             retained_limit=100)
    for name, pinned in [('active', False), ('pinned', True), ('retained', False)]:
        cache.store.save(f'weights/{name}', lambda handle: handle.write(b'x' * 100),
                         expected_bytes=100, group_id=name, pinned=pinned)
    cache.store.retain_representative('weights/retained', 'toy', 1)
    cache.store.release_group('retained')
    cache.store.release_group('pinned')
    if failure_stage == 'payload_write':
        original_save = cache.store.save
        def underestimate(relative_path, writer, expected_bytes, group_id, **metadata):
            if relative_path.startswith('checkpoints/'):
                expected_bytes = 1
            return original_save(relative_path, writer, expected_bytes, group_id, **metadata)
        monkeypatch.setattr(cache.store, 'save', underestimate)

    model = torch.nn.Linear(64, 64)
    before = model.weight.detach().clone()
    with pytest.raises(CapacityBlocked):
        cache.save('deep_candidate', TrainConfig(), {'model': model})
    snapshot = cache.store.snapshot()
    assert snapshot['actual_bytes'] == 300
    assert snapshot['accounted_bytes'] <= snapshot['limit_bytes']
    assert snapshot['active_reservations'] == 0
    assert not list((cache.store.root / '.tmp').glob('*'))
    assert not list((cache.store.root / 'checkpoints').glob('*'))
    assert not list(cache.index.glob('*.json'))
    for name in ('active', 'pinned', 'retained'):
        assert (cache.store.root / f'weights/{name}').read_bytes() == b'x' * 100
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
    if failure_stage == 'payload_write':
        events = [json.loads(line) for line in cache.store._events_path.read_text().splitlines()]
        assert any(event['event'] == 'write_capacity_blocked' for event in events)
        assert any(event['event'] == 'reservation_aborted' for event in events)

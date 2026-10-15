import concurrent.futures
import json
import multiprocessing
import pickle
from pathlib import Path
import threading
import time

import pytest

from backend.app.benchmark_resources import CapacityBlocked, ResourceScheduler, WeightStore


def store_at(path, **kwargs):
    return WeightStore(path, limit_bytes=1000, high=800, low=600, retained_limit=150, **kwargs)


def test_write_extends_before_write_and_failure_removes_only_own_tmp(tmp_path):
    store = store_at(tmp_path / 'payload')
    artifact = store.save('weights/a', lambda f: f.write(b'a' * 700), 100, 'active')
    assert artifact['bytes'] == 700
    assert store.snapshot()['accounted_bytes'] == 700
    with pytest.raises(CapacityBlocked):
        store.save('checkpoints/b', lambda f: f.write(b'b' * 301), 100, 'other')
    assert store.snapshot()['actual_bytes'] == 700
    assert store.snapshot()['active_reservations'] == 0
    assert (store.root / 'weights/a').exists()
    assert not list((store.root / '.tmp').iterdir())
    events = [json.loads(line) for line in store._events_path.read_text().splitlines()]
    assert any(event['event'] == 'write_capacity_blocked' for event in events)


def test_seek_truncate_and_unregistered_files_are_accounted(tmp_path):
    store = store_at(tmp_path / 'payload')
    cache = store.root / 'cache'
    cache.mkdir()
    (cache / 'external').write_bytes(b'e' * 100)
    with store.open_writer('learnedtransforms/a', 30, 'g') as handle:
        handle.seek(899)
        handle.write(b'x')
        with pytest.raises(CapacityBlocked):
            handle.truncate(901)
    assert store.snapshot()['actual_bytes'] == 1000
    store.release_group('g')
    assert (cache / 'external').exists()
    assert store.snapshot()['actual_bytes'] == 100


def test_reservations_survive_reopen_and_protect_active_payloads(tmp_path):
    store = store_at(tmp_path / 'payload')
    reservation = store.reserve('a', 600, 'g')
    with reservation.open_writer() as f:
        f.write(b'x' * 300)
    restored = store_at(store.root)
    assert restored.snapshot()['accounted_bytes'] == 600
    assert restored.cleanup(force=True)['capacity_blocked'] is False
    with pytest.raises(CapacityBlocked):
        restored.reserve('b', 401, 'h')
    with pytest.raises(RuntimeError):
        restored.release_group('g')
    reservation.commit()
    assert restored.snapshot()['reserved_bytes'] == 0
    assert restored.snapshot()['actual_bytes'] == 300


def test_cleanup_preserves_active_pinned_retained_and_manifest_is_truthful(tmp_path):
    store = store_at(tmp_path / 'payload')
    store.save('weights/representative', lambda f: f.write(b'r' * 100), 100, 'g')
    assert store.retain_representative('weights/representative', 'model', 4)
    store.save('weights/pinned', lambda f: f.write(b'p' * 200), 200, 'g', pinned=True)
    store.save('weights/temporary', lambda f: f.write(b't' * 300), 300, 'g')
    with pytest.raises(RuntimeError):
        store.finalized_manifest()
    result = store.release_group('g')
    assert result['removed_bytes'] == 300
    manifest = {a['path']: a for a in store.finalized_manifest()['artifacts']}
    assert manifest['weights/temporary']['available'] is False
    assert manifest['weights/temporary']['required'] is False
    assert manifest['weights/temporary']['sha256']
    assert manifest['weights/representative']['required'] is True
    assert manifest['weights/pinned']['required'] is True
    assert store.cleanup(force=True)['removed_bytes'] == 0


def test_representative_uses_development_order_and_retained_cap(tmp_path):
    store = store_at(tmp_path / 'payload')
    for name in ('later', 'first', 'oversized'):
        size = 151 if name == 'oversized' else 100
        store.save(name, lambda f, n=size: f.write(b'x' * n), size, name)
    assert store.retain_representative('later', 'm', 7)
    assert store.retain_representative('first', 'm', 2)
    assert not store.retain_representative('later', 'm', 7)
    assert not store.retain_representative('oversized', 'other', 1)
    assert store.snapshot()['retained_bytes'] == 100
    store.release_group('later')
    assert not (store.root / 'later').exists()
    store.release_group('first')
    assert (store.root / 'first').exists()


def test_path_escape_and_symlink_are_rejected(tmp_path):
    store = store_at(tmp_path / 'payload')
    outside = tmp_path / 'outside'
    outside.mkdir()
    (store.root / 'link').symlink_to(outside)
    for name in ('../outside/a', str(outside / 'a'), 'link/a'):
        with pytest.raises(ValueError):
            store.reserve(name, 10, 'g')
    (outside / 'keep').write_bytes(b'keep')
    store.cleanup(force=True)
    assert (outside / 'keep').read_bytes() == b'keep'


def _process_write(root, barrier, queue, number):
    store = WeightStore(Path(root), limit_bytes=1000, high=800, low=600, retained_limit=150)
    barrier.wait()
    try:
        store.save(f'weights/{number}', lambda f: f.write(b'x' * 300), 1, str(number))
        queue.put('saved')
    except CapacityBlocked:
        queue.put('blocked')


def test_process_race_never_crosses_payload_limit(tmp_path):
    root = tmp_path / 'payload'
    store = store_at(root)
    context = multiprocessing.get_context('fork')
    barrier, queue = context.Barrier(8), context.Queue()
    processes = [context.Process(target=_process_write, args=(str(root), barrier, queue, n)) for n in range(8)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    results = [queue.get(timeout=1) for _ in processes]
    assert results.count('saved') == 3
    assert results.count('blocked') == 5
    snapshot = store.snapshot()
    assert snapshot['actual_bytes'] == 900
    assert snapshot['accounted_bytes'] <= snapshot['limit_bytes']
    assert snapshot['active_reservations'] == 0


def test_pickle_writer_real_seekable_handle(tmp_path):
    store = store_at(tmp_path / 'payload')
    value = {'config': [1, 2, 3], 'weights': b'abc'}
    store.save('weights/model.pkl', lambda f: pickle.dump(value, f), 1, 'g')
    with (store.root / 'weights/model.pkl').open('rb') as f:
        assert pickle.load(f) == value


def gpu_sample(pids=(), util=30, used=100):
    return {gpu: {'utilization_percent': util, 'memory_used_mb': used,
                  'memory_total_mb': 1000, 'process_pids': list(pids)} for gpu in (1, 2, 3, 5)}


def test_shared_cpu_budget_and_initial_gpu_slots():
    scheduler = ResourceScheduler(sampler=gpu_sample, start_monitor=False)
    scheduler.sample()
    gpu = [scheduler.allocate('gpu') for _ in range(8)]
    assert all(gpu)
    assert scheduler.allocate('gpu') is None
    cpu = [scheduler.allocate('cpu') for _ in range(6)]
    assert all(cpu)
    assert scheduler.metrics()['cpu/threads_in_use'] == 32
    assert scheduler.allocate('cpu') is None
    gpu[0].release()
    gpu[0].release()
    assert scheduler.metrics()['cpu/threads_in_use'] == 31
    assert scheduler.allocate('gpu') is not None
    scheduler.close()


def test_cpu_thread_budget_under_thread_contention():
    scheduler = ResourceScheduler(start_monitor=False)
    maximum = [0]
    lock = threading.Lock()
    def work(number):
        with scheduler.allocate('cpu', block=True, timeout=3) as lease:
            assert lease.cpu_threads == 4
            with lock:
                maximum[0] = max(maximum[0], scheduler.metrics()['cpu/threads_in_use'])
            time.sleep(.01)
    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as pool:
        list(pool.map(work, range(80)))
    assert maximum[0] == 32
    assert scheduler.metrics()['cpu/threads_in_use'] == 0


def test_gpu_unknown_process_blocks_and_known_process_allows():
    observations = gpu_sample()
    scheduler = ResourceScheduler(sampler=lambda: observations, start_monitor=False)
    scheduler.sample()
    lease = scheduler.allocate('gpu')
    assert lease.gpu_id == 1
    observations[1]['process_pids'] = [999]
    scheduler.sample()
    assert scheduler.allocate('gpu').gpu_id == 2
    scheduler.register_pid(lease, 999)
    scheduler.sample()
    assert scheduler.allocate('gpu').gpu_id == 3
    scheduler.close()


def test_gpu_adaptation_metrics_and_sampling_error():
    observations = gpu_sample()
    scheduler = ResourceScheduler(sampler=lambda: observations, start_monitor=False)
    scheduler.sample()
    lease = scheduler.allocate('gpu')
    scheduler.record_completion(lease, 10, 5)
    scheduler.sample()
    scheduler.sample()
    assert scheduler.metrics()['gpu/1/concurrent_job_limit'] == 4
    for value in observations.values():
        value['memory_used_mb'] = 990
        value['utilization_percent'] = 100
    for _ in range(3):
        scheduler.sample()
    assert scheduler.metrics()['gpu/1/concurrent_job_limit'] < 4
    assert scheduler.metrics()['gpu/1/mean_memory_used_percent'] == 99
    def failure():
        raise OSError('sampling unavailable')
    scheduler._sampler = failure
    scheduler.sample()
    assert scheduler.allocate('gpu') is None
    assert scheduler.metrics()['scheduler/sample_error'] == 'sampling unavailable'
    assert scheduler.allocate('cpu') is not None


def test_commit_recovers_after_rename_before_registry_update(tmp_path, monkeypatch):
    import backend.app.benchmark_resources as module
    store = store_at(tmp_path / 'payload')
    reservation = store.reserve('weights/a', 100, 'g')
    with reservation.open_writer() as handle:
        handle.write(b'x' * 80)
    actual_replace = module.os.replace
    def crash_replace(source, destination):
        actual_replace(source, destination)
        if str(destination).endswith('weights/a'):
            raise RuntimeError('simulated interruption after rename')
    monkeypatch.setattr(module.os, 'replace', crash_replace)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        reservation.commit()
    monkeypatch.setattr(module.os, 'replace', actual_replace)
    restored = store_at(store.root)
    with pytest.raises(RuntimeError, match='recover'):
        restored.reservation(reservation.token).abort()
    artifact = restored.reservation(reservation.token).commit()
    assert artifact['bytes'] == 80
    assert restored.snapshot()['active_reservations'] == 0
    assert restored.snapshot()['actual_bytes'] == 80


def test_high_watermark_blocked_admission_and_no_active_deletion(tmp_path):
    store = store_at(tmp_path / 'payload')
    store.save('a', lambda f: f.write(b'x' * 850), 850, 'g')
    assert store.cleanup()['capacity_blocked'] is True
    with pytest.raises(CapacityBlocked, match='high watermark'):
        store.reserve('b', 1, 'h')
    assert (store.root / 'a').exists()
    store.release_group('g')
    reservation = store.reserve('b', 1, 'h')
    reservation.abort()


def test_joblib_and_torch_serialization_handles(tmp_path):
    import joblib
    import numpy as np
    import torch
    store = WeightStore(tmp_path / 'payload', limit_bytes=100_000, high=80_000,
                        low=60_000, retained_limit=10_000)
    array = np.arange(20)
    store.save('transforms/a.joblib', lambda f: joblib.dump(array, f), 1, 'g')
    assert np.array_equal(joblib.load(store.root / 'transforms/a.joblib'), array)
    value = {'weight': torch.arange(20)}
    store.save('checkpoints/a.pt', lambda f: torch.save(value, f), 1, 'g')
    result = torch.load(store.root / 'checkpoints/a.pt', weights_only=True)
    assert torch.equal(result['weight'], value['weight'])
    assert store.snapshot()['actual_bytes'] <= store.limit_bytes


def test_gpu_does_not_increase_without_fresh_throughput_evidence():
    scheduler = ResourceScheduler(sampler=gpu_sample, start_monitor=False)
    scheduler.sample()
    lease = scheduler.allocate('gpu')
    scheduler.record_completion(lease, 10, 5)
    for _ in range(3):
        scheduler.sample()
    assert scheduler.metrics()['gpu/1/concurrent_job_limit'] == 4
    for _ in range(6):
        scheduler.sample()
    assert scheduler.metrics()['gpu/1/concurrent_job_limit'] == 4
    scheduler.record_completion(lease, 10, 5)
    scheduler.sample()
    assert scheduler.metrics()['gpu/1/concurrent_job_limit'] == 8


def test_explicit_shared_gpu_allows_existing_processes():
    scheduler = ResourceScheduler(sampler=lambda: gpu_sample(pids=[999]), start_monitor=False, allow_shared_gpu=True)
    scheduler.sample()
    lease = scheduler.allocate("gpu")
    assert lease.gpu_id == 1
    lease.release()
    scheduler.close()


def test_shared_gpu_external_utilization_does_not_reduce_own_slots():
    scheduler = ResourceScheduler(sampler=lambda: gpu_sample(util=99, pids=[999]), start_monitor=False, allow_shared_gpu=True)
    scheduler.sample()
    assert scheduler.metrics()["gpu/1/concurrent_job_limit"] == 2
    scheduler.close()

"""Private benchmark storage and CPU/GPU admission controls.

All artifact writers must use ``open_writer``/``save``. The payload root includes
weights, checkpoints, transforms, cache and temporary files; its sibling control
directory contains only the lock, durable reservations and append-only audit log.
No cleanup operation follows symlinks or removes unregistered files.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid


class CapacityBlocked(RuntimeError):
    """Capacity is held by active/pinned artifacts or outstanding reservations."""


class WeightStore:
    def __init__(self, root: Path, limit_bytes=50_000_000_000,
                 high=40_000_000_000, low=35_000_000_000,
                 retained_limit=5_000_000_000):
        self.root = Path(root).resolve()
        if not 0 <= low <= high <= limit_bytes or retained_limit > limit_bytes:
            raise ValueError('invalid storage limits')
        self.limit_bytes, self.high, self.low = int(limit_bytes), int(high), int(low)
        self.retained_limit = int(retained_limit)
        self.root.mkdir(parents=True, exist_ok=True)
        self.control = self.root.parent / ('.' + self.root.name + '.weightstore')
        self.control.mkdir(exist_ok=True)
        self._state_path = self.control / 'state.json'
        self._lock_path = self.control / 'lock'
        self._events_path = self.control / 'events.jsonl'
        with self._locked() as state:
            if state['root'] != str(self.root):
                raise ValueError('storage identity mismatch')

    @contextmanager
    def _locked(self):
        with self._lock_path.open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = (json.loads(self._state_path.read_text()) if self._state_path.exists()
                     else {'root': str(self.root), 'reservations': {}, 'artifacts': {},
                           'last_cleanup': 0.0})
            try:
                yield state
            finally:
                self._persist(state)
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _persist(self, state):
        temporary = self.control / 'state.next.json'
        with temporary.open('w') as handle:
            json.dump(state, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self._state_path)
        directory = os.open(self.control, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def reservation(self, token):
        """Resume an explicitly selected durable reservation after a crash."""
        with self._locked() as state:
            if token not in state['reservations']:
                raise KeyError(token)
        return Reservation(self, token)

    def _path(self, relative):
        supplied = Path(relative)
        path = (self.root / supplied).resolve()
        if supplied.is_absolute() or path == self.root or not path.is_relative_to(self.root):
            raise ValueError('artifact path must be inside private store root')
        # Reject symlink paths even when they happen to resolve inside this root.
        for parent in [self.root / supplied, *(self.root / supplied).parents]:
            if parent == self.root:
                break
            if parent.is_symlink():
                raise ValueError('symlink artifact paths are not supported')
        return path

    def _physical(self):
        # lstat counts links themselves and never follows directories via symlinks.
        total = 0
        for directory, dirs, files in os.walk(self.root, followlinks=False):
            for name in files:
                total += (Path(directory) / name).lstat().st_size
            for name in dirs:
                path = Path(directory) / name
                if path.is_symlink():
                    total += path.lstat().st_size
        return total

    def _usage(self, state):
        actual = self._physical()
        held = sum(max(0, r['bytes'] - self._size(r['temporary']))
                   for r in state['reservations'].values())
        return actual, held

    def _size(self, relative):
        path = self._path(relative)
        return path.stat().st_size if path.exists() else 0

    def _event(self, event, **fields):
        with self._events_path.open('a') as handle:
            handle.write(json.dumps({'time': time.time(), 'event': event, **fields},
                                    sort_keys=True) + '\n')
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _sha(path):
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()

    def _remove(self, state, relative, reason):
        artifact = state['artifacts'][relative]
        path = self._path(relative)
        size = path.stat().st_size if path.exists() else 0
        sha = self._sha(path) if path.exists() else artifact['sha256']
        if path.exists():
            path.unlink()
        artifact.update(available=False, required=False, deleted_at=time.time())
        self._event('artifact_removed', path=relative, reason=reason,
                    bytes=size, sha256=sha, removed_count=1)
        return size

    def _cleanup(self, state, force=False):
        actual, held = self._usage(state)
        if actual + held < self.high and not force:
            return {'removed_count': 0, 'removed_bytes': 0, 'capacity_blocked': False}
        count = removed = 0
        candidates = sorted(state['artifacts'].items(), key=lambda item: item[1]['created'])
        for relative, artifact in candidates:
            if actual + held <= self.low:
                break
            if (artifact['available'] and not artifact['active'] and
                    not artifact['pinned'] and not artifact['retained']):
                amount = self._remove(state, relative, 'high_watermark')
                removed += amount
                actual -= amount
                count += 1
        state['last_cleanup'] = time.time()
        result = {'removed_count': count, 'removed_bytes': removed,
                  'capacity_blocked': actual + held > self.low,
                  'actual_bytes': actual, 'reserved_bytes': held}
        self._event('cleanup', **result)
        return result

    def reserve(self, relative_path, expected_bytes, group_id, *, model_id=None,
                order=None, pinned=False):
        path = self._path(relative_path)
        if expected_bytes < 0:
            raise ValueError('expected bytes must be nonnegative')
        relative = str(path.relative_to(self.root))
        token = uuid.uuid4().hex
        temporary = '.tmp/' + token
        with self._locked() as state:
            if path.exists() or any(r['path'] == relative for r in state['reservations'].values()):
                raise FileExistsError(relative)
            cleaned = self._cleanup(state)
            actual, held = self._usage(state)
            if cleaned['capacity_blocked']:
                self._event('capacity_blocked', path=relative, requested_bytes=expected_bytes,
                            actual_bytes=actual, reserved_bytes=held, reason='high_watermark')
                raise CapacityBlocked('high watermark held by active/pinned artifacts')
            if actual + held + expected_bytes > self.limit_bytes:
                self._cleanup(state, force=True)
                actual, held = self._usage(state)
            if actual + held + expected_bytes > self.limit_bytes:
                self._event('capacity_blocked', path=relative, requested_bytes=expected_bytes,
                            actual_bytes=actual, reserved_bytes=held)
                raise CapacityBlocked('private weight store capacity is held')
            state['reservations'][token] = {
                'path': relative, 'temporary': temporary, 'bytes': int(expected_bytes),
                'group_id': str(group_id), 'model_id': model_id, 'order': order,
                'pinned': bool(pinned), 'created': time.time()}
            self._event('reserved', reservation=token, path=relative, bytes=expected_bytes)
        return Reservation(self, token)

    @contextmanager
    def open_writer(self, relative_path, expected_bytes, group_id, **metadata):
        reservation = self.reserve(relative_path, expected_bytes, group_id, **metadata)
        try:
            with reservation.open_writer() as handle:
                yield handle
            reservation.commit()
        except BaseException:
            reservation.abort()
            raise

    def save(self, relative_path, writer, expected_bytes, group_id, **metadata):
        with self.open_writer(relative_path, expected_bytes, group_id, **metadata) as handle:
            writer(handle)
        with self._locked() as state:
            return dict(state['artifacts'][str(Path(relative_path))])

    def register(self, relative_path, group_id, *, pinned=False, model_id=None, order=None):
        """Register existing private files; external writes must already have been bounded."""
        path = self._path(relative_path)
        with self._locked() as state:
            actual, held = self._usage(state)
            if actual + held > self.limit_bytes:
                raise CapacityBlocked('existing files exceed private storage capacity')
            relative = str(path.relative_to(self.root))
            artifact = dict(path=relative, bytes=path.stat().st_size, sha256=self._sha(path),
                            group_id=str(group_id), active=True, pinned=bool(pinned),
                            retained=False, available=True, required=True,
                            model_id=model_id, order=order, created=time.time())
            state['artifacts'][relative] = artifact
            self._event('registered', path=relative, bytes=artifact['bytes'],
                        sha256=artifact['sha256'])
            return dict(artifact)

    def retain_representative(self, relative_path, model_id, order):
        """Keep the first successful artifact in caller-supplied development order."""
        relative = str(self._path(relative_path).relative_to(self.root))
        with self._locked() as state:
            candidate = state['artifacts'][relative]
            if not candidate['available']:
                raise FileNotFoundError(relative)
            previous = [(p, a) for p, a in state['artifacts'].items()
                        if a['retained'] and a['model_id'] == model_id]
            if previous and previous[0][1]['order'] <= order:
                return previous[0][0] == relative
            retained = sum(a['bytes'] for a in state['artifacts'].values() if a['retained'])
            previous_bytes = sum(a['bytes'] for _, a in previous)
            if retained - previous_bytes + candidate['bytes'] > self.retained_limit:
                self._event('representative_capacity_blocked', path=relative, model_id=model_id)
                return False
            for _, artifact in previous:
                artifact['retained'] = False
            candidate.update(retained=True, model_id=model_id, order=order)
            self._event('representative_retained', path=relative, model_id=model_id, order=order,
                        bytes=candidate['bytes'], sha256=candidate['sha256'])
            return True

    def set_pinned(self, relative_path, pinned=True):
        with self._locked() as state:
            state['artifacts'][str(self._path(relative_path).relative_to(self.root))]['pinned'] = bool(pinned)

    def release_group(self, group_id):
        """Caller invokes only after paired predictions and explanations are complete."""
        count = removed = 0
        with self._locked() as state:
            pending = [r for r in state['reservations'].values() if r['group_id'] == str(group_id)]
            if pending:
                raise RuntimeError('group has unfinished reservations; commit or abort first')
            for relative, artifact in state['artifacts'].items():
                if artifact['group_id'] != str(group_id) or not artifact['available']:
                    continue
                artifact['active'] = False
                if not artifact['pinned'] and not artifact['retained']:
                    removed += self._remove(state, relative, 'paired_group_complete')
                    count += 1
            self._cleanup(state)
            return {'removed_count': count, 'removed_bytes': removed}

    def cleanup(self, force=False):
        with self._locked() as state:
            return self._cleanup(state, force)

    def snapshot(self):
        with self._locked() as state:
            actual, held = self._usage(state)
            return {'actual_bytes': actual, 'reserved_bytes': held,
                    'accounted_bytes': actual + held, 'limit_bytes': self.limit_bytes,
                    'active_reservations': len(state['reservations']),
                    'retained_bytes': sum(a['bytes'] for a in state['artifacts'].values() if a['retained']),
                    'artifacts': [dict(a) for a in state['artifacts'].values()],
                    'reservations': {k: dict(v) for k, v in state['reservations'].items()}}

    def finalized_manifest(self):
        with self._locked() as state:
            if state['reservations'] or any(a['active'] for a in state['artifacts'].values() if a['available']):
                raise RuntimeError('release all groups before finalizing artifact manifest')
            return {'artifacts': [dict(a, required=bool(a['available'])) for a in state['artifacts'].values()]}


class Reservation:
    def __init__(self, store, token):
        self.store, self.token = store, token

    @contextmanager
    def open_writer(self):
        with self.store._locked() as state:
            record = state['reservations'][self.token]
            path = self.store._path(record['temporary'])
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open('x+b', buffering=0)
        bounded = _BoundedFile(self, handle)
        try:
            yield bounded
            bounded.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()

    def commit(self):
        with self.store._locked() as state:
            record = state['reservations'][self.token]
            temporary = self.store._path(record['temporary'])
            path = self.store._path(record['path'])
            if path.exists():
                # The durable committing phase closes rename -> ledger crash window.
                if (temporary.exists() or record.get('phase') != 'committing' or
                        path.stat().st_size != record['committed_bytes'] or
                        self.store._sha(path) != record['sha256']):
                    raise FileExistsError(record['path'])
                size, sha = record['committed_bytes'], record['sha256']
            else:
                size = temporary.stat().st_size
                sha = self.store._sha(temporary)
                record.update(phase='committing', committed_bytes=size, sha256=sha)
                self.store._persist(state)
                path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temporary, path)
                directory = os.open(path.parent, os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            artifact = dict(path=record['path'], bytes=size, sha256=sha,
                            group_id=record['group_id'], active=True, pinned=record['pinned'],
                            retained=False, available=True, required=True,
                            model_id=record['model_id'], order=record['order'], created=time.time())
            state['artifacts'][record['path']] = artifact
            del state['reservations'][self.token]
            self.store._event('committed', reservation=self.token, path=record['path'],
                              bytes=size, sha256=sha)
            return dict(artifact)

    def abort(self):
        with self.store._locked() as state:
            record = state['reservations'].get(self.token)
            if record is None:
                return
            path = self.store._path(record['temporary'])
            if record.get('phase') == 'committing' and not path.exists():
                raise RuntimeError('rename completed; recover with reservation.commit()')
            exists = path.exists()
            size = path.stat().st_size if exists else 0
            sha = self.store._sha(path) if exists else None
            if exists:
                path.unlink()
            del state['reservations'][self.token]
            self.store._event('reservation_aborted', reservation=self.token,
                              bytes=size, sha256=sha, removed_count=int(exists))


class _BoundedFile(io.BufferedIOBase):
    def __init__(self, reservation, handle):
        self.reservation, self._handle = reservation, handle

    def writable(self): return True
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self._handle.tell()
    def seek(self, offset, whence=0): return self._handle.seek(offset, whence)
    def read(self, size=-1): return self._handle.read(size)
    def readinto(self, buffer): return self._handle.readinto(buffer)
    def flush(self):
        if not self._handle.closed:
            self._handle.flush()

    def _mutate(self, final_size, operation):
        store = self.reservation.store
        with store._locked() as state:
            record = state['reservations'][self.reservation.token]
            actual, held = store._usage(state)
            old_size = store._size(record['temporary'])
            wanted = max(old_size, final_size)
            extra = max(0, wanted - record['bytes'])
            if actual + held + extra > store.limit_bytes:
                store._cleanup(state, force=True)
                actual, held = store._usage(state)
            if actual + held + extra > store.limit_bytes:
                store._event('write_capacity_blocked', reservation=self.reservation.token,
                             requested_bytes=wanted, actual_bytes=actual, reserved_bytes=held)
                raise CapacityBlocked('write exceeds private weight store capacity')
            if extra:
                record['bytes'] = wanted
                store._event('reservation_extended', reservation=self.reservation.token, bytes=wanted)
            return operation()

    def write(self, data):
        return self._mutate(self.tell() + memoryview(data).nbytes, lambda: self._handle.write(data))

    def truncate(self, size=None):
        size = self.tell() if size is None else size
        return self._mutate(size, lambda: self._handle.truncate(size))


@dataclass
class ResourceLease:
    scheduler: 'ResourceScheduler'
    token: str
    kind: str
    cpu_threads: int
    gpu_id: int | None

    def release(self): self.scheduler.release(self)
    def __enter__(self): return self
    def __exit__(self, *args): self.release()


class ResourceScheduler:
    """Admission only; never changes training batch size/epochs or starts processes.

    GPU processes must be registered with ``register_pid`` as soon as they start.
    Unknown processes withhold a GPU unless explicit shared-GPU execution is enabled.
    """
    def __init__(self, gpu_ids=(1, 2, 3, 5), cpu_tokens=32, sample_interval=5.0,
                 sampler=None, start_monitor=True, allow_shared_gpu=False):
        if cpu_tokens < 4 or sample_interval < 5:
            raise ValueError('CPU tokens >= 4 and sample interval >= 5 seconds required')
        self.gpu_ids = tuple(gpu_ids)
        self.allow_shared_gpu = allow_shared_gpu
        self.cpu_tokens, self.sample_interval = cpu_tokens, sample_interval
        self._sampler = sampler or self._nvidia_metrics
        self._condition = threading.Condition()
        self._leases = {}
        self._pids = {}
        self._slots = {gpu: 2 for gpu in self.gpu_ids}
        self._samples = {gpu: deque(maxlen=3) for gpu in self.gpu_ids}
        self._completed = {gpu: deque() for gpu in self.gpu_ids}
        self._previous_throughput = {gpu: None for gpu in self.gpu_ids}
        self._adjusted_completion = {gpu: 0.0 for gpu in self.gpu_ids}
        self._observed = {}
        self._stop = threading.Event()
        self._thread = None
        self._last_sample = 0.0
        self._sample_error = None
        if start_monitor:
            self._thread = threading.Thread(target=self._monitor, daemon=True,
                                            name='benchmark-resource-monitor')
            self._thread.start()

    @staticmethod
    def _nvidia_metrics():
        fields = 'index,uuid,utilization.gpu,memory.used,memory.total'
        command = ['nvidia-smi', '--query-gpu=' + fields, '--format=csv,noheader,nounits']
        output = subprocess.run(command, capture_output=True, text=True, timeout=4, check=True).stdout
        result, identifiers = {}, {}
        for row in output.strip().splitlines():
            index, identifier, util, used, total = [value.strip() for value in row.split(',')]
            index = int(index)
            identifiers[identifier] = index
            result[index] = {'utilization_percent': float(util), 'memory_used_mb': float(used),
                             'memory_total_mb': float(total), 'process_pids': []}
        output = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                 '--format=csv,noheader,nounits'], capture_output=True,
                                text=True, timeout=4, check=True).stdout
        for row in output.strip().splitlines():
            identifier, pid = [value.strip() for value in row.split(',')]
            if identifier in identifiers:
                result[identifiers[identifier]]['process_pids'].append(int(pid))
        return result

    def _monitor(self):
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.sample_interval)

    def sample(self):
        try:
            observations = self._sampler()
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            with self._condition:
                self._sample_error = str(error)
                self._observed = {}
                self._condition.notify_all()
            return self.metrics()
        now = time.monotonic()
        with self._condition:
            observations = {gpu: dict(value, process_pids=list(value['process_pids']))
                            for gpu, value in observations.items()}
            self._observed = observations
            self._sample_error = None
            self._last_sample = now
            for gpu in self.gpu_ids:
                observation = observations.get(gpu)
                if observation is None:
                    continue
                samples = self._samples[gpu]
                samples.append(dict(observation))
                util = sum(s['utilization_percent'] for s in samples) / len(samples)
                memory = sum(s['memory_used_mb'] / s['memory_total_mb'] for s in samples) / len(samples)
                completed = self._completed[gpu]
                while completed and now - completed[0][0] > 60:
                    completed.popleft()
                throughput = self._throughput(completed)
                previous = self._previous_throughput[gpu]
                current = self._slots[gpu]
                fresh_completion = completed and completed[-1][0] > self._adjusted_completion[gpu]
                if memory >= .9 or (not self.allow_shared_gpu and util >= 95) or (previous and throughput < previous * .85):
                    self._slots[gpu] = max(1, current // 2)
                elif (fresh_completion and len(samples) == 3 and memory < .7 and (self.allow_shared_gpu or util < 70) and throughput > 0
                      and (previous is None or throughput >= previous * .95)):
                    self._slots[gpu] = min(8, current * 2)
                    self._adjusted_completion[gpu] = completed[-1][0]
                self._previous_throughput[gpu] = throughput
            self._condition.notify_all()
        return self.metrics()

    def allocate(self, kind, block=False, timeout=None):
        if kind not in ('cpu', 'gpu'):
            raise ValueError('kind must be cpu or gpu')
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                threads = 4 if kind == 'cpu' else 1
                cpu_used = sum(lease.cpu_threads for lease in self._leases.values())
                gpu = None
                ready = cpu_used + threads <= self.cpu_tokens
                if kind == 'gpu' and ready:
                    available = []
                    for candidate in self.gpu_ids:
                        observation = self._observed.get(candidate)
                        if observation is None:
                            continue
                        own_pids = {pid for token, pid in self._pids.items()
                                    if token in self._leases and self._leases[token].gpu_id == candidate}
                        if not self.allow_shared_gpu and set(observation['process_pids']) - own_pids:
                            continue
                        used = sum(lease.gpu_id == candidate for lease in self._leases.values())
                        if used < self._slots[candidate]:
                            available.append((used / self._slots[candidate], candidate))
                    ready = bool(available)
                    if ready:
                        gpu = min(available)[1]
                if ready:
                    token = uuid.uuid4().hex
                    lease = ResourceLease(self, token, kind, threads, gpu)
                    self._leases[token] = lease
                    return lease
                if not block:
                    return None
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def register_pid(self, lease, pid):
        with self._condition:
            if lease.token not in self._leases:
                raise ValueError('lease already released')
            self._pids[lease.token] = int(pid)

    def release(self, lease):
        with self._condition:
            self._leases.pop(lease.token, None)
            self._pids.pop(lease.token, None)
            self._condition.notify_all()

    def record_completion(self, lease, work_units, seconds):
        if work_units < 0 or seconds <= 0:
            raise ValueError('invalid completion measurement')
        if lease.gpu_id is not None:
            with self._condition:
                self._completed[lease.gpu_id].append((time.monotonic(), work_units, seconds))

    def _throughput(self, completed):
        # Concurrent completions share wall time, rather than summing job durations.
        if not completed:
            return 0.0
        elapsed = max(completed[-1][0] - completed[0][0], max(c[2] for c in completed))
        return sum(c[1] for c in completed) / max(self.sample_interval, elapsed)

    def metrics(self):
        with self._condition:
            metrics = {'cpu/threads_in_use': sum(l.cpu_threads for l in self._leases.values()),
                       'cpu/thread_limit': self.cpu_tokens, 'scheduler/active_jobs': len(self._leases),
                       'scheduler/sample_error': self._sample_error}
            for gpu in self.gpu_ids:
                prefix = f'gpu/{gpu}/'
                samples = self._samples[gpu]
                observation = self._observed.get(gpu)
                completed = self._completed[gpu]
                metrics[prefix + 'concurrent_job_limit'] = self._slots[gpu]
                metrics[prefix + 'active_jobs'] = sum(l.gpu_id == gpu for l in self._leases.values())
                metrics[prefix + 'available'] = observation is not None
                metrics[prefix + 'completed_work_units_per_second'] = self._throughput(completed)
                if samples:
                    metrics[prefix + 'mean_utilization_percent'] = sum(s['utilization_percent'] for s in samples) / len(samples)
                    metrics[prefix + 'mean_memory_used_percent'] = 100 * sum(s['memory_used_mb'] / s['memory_total_mb'] for s in samples) / len(samples)
                if observation:
                    metrics[prefix + 'memory_used_mb'] = observation['memory_used_mb']
                    metrics[prefix + 'memory_total_mb'] = observation['memory_total_mb']
            return metrics

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=9)

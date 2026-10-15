"""Durable, credential-free online tracking journal shared with node2.

A task owns one SwanLab Run. The collector alone talks to SwanLab; producers
must obtain a fresh verified cloud reference before fitting or requesting an LLM.
Raw events survive an outage. No duration aggregation is treated as wall time.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

HEARTBEAT_SECONDS = 15
API_FRESH_SECONDS = 30
FORBIDDEN_FIELDS = frozenset(('token', 'api_key', 'authorization', 'headers', 'httpheaders',
    'private_key', 'password', 'access_token', 'refresh_token'))


class OnlineUnavailable(RuntimeError):
    pass


class OnlineVerificationFailed(RuntimeError):
    pass


def json_value(value):
    """Validate pure JSON and reject credential-bearing fields before persistence."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is list:
        return [json_value(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        if any(key.lower().replace('-', '_') in FORBIDDEN_FIELDS for key in value):
            raise ValueError('tracking accepts safe measurements, not authentication or HTTP headers')
        return {key: json_value(item) for key, item in value.items()}
    raise ValueError('tracking events require finite basic JSON values')


def json_equal(left, right):
    return json.dumps(left, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(right, sort_keys=True, separators=(',', ':'), allow_nan=False)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def cloud_required(root):
    policy = read_json(Path(root) / 'approval' / 'recording-override.json', {})
    return not (policy.get('user_approved') is True and policy.get('swanlab_required') is False)



@contextmanager
def file_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == 'nt':
        from .runs.locking import run_lock
        with run_lock(path.parent / 'tracking-lock-domain', path.name):
            yield
    else:
        import fcntl
        with path.open('a+b') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield


def task_key(task_id):
    if type(task_id) is not str or not task_id:
        raise ValueError('task_id must be a nonempty string')
    return hashlib.sha256(task_id.encode()).hexdigest()


class OnlineJournal:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.directory = self.root / 'tracking'
        self.tasks = self.directory / 'tasks'
        self.tasks.mkdir(parents=True, exist_ok=True)

    def task_dir(self, task_id):
        return self.tasks / task_key(task_id)

    def _event(self, directory, task_id, sequence, kind, data):
        event_id = uuid.uuid4().hex
        event = dict(schema_version='benchmark-online-event-v1', event_id=event_id,
            task_id=task_id, sequence=sequence, kind=kind, data=data, created_at=time.time())
        atomic_json(directory / 'events' / f'{sequence:08d}-{event_id}.json', event)
        return event_id

    def start_task(self, task_id, config):
        config = json_value(config)
        if type(config) is not dict: raise ValueError('task config must be a JSON object')
        directory = self.task_dir(task_id)
        with file_lock(directory / 'producer.lock'):
            previous = read_json(directory / 'task.json')
            if previous is not None:
                if previous['task_id'] != task_id or not json_equal(previous['config'], config):
                    raise ValueError('tracking task configuration is already frozen')
                return previous['start_event_id']
            orphan = list((directory / 'events').glob('00000001-*.json'))
            if orphan:
                event = read_json(orphan[0])
                if event['kind'] != 'task.started' or not json_equal(event['data'], config):
                    raise ValueError('incomplete tracking start differs from requested config')
                event_id = event['event_id']
            else:
                event_id = self._event(directory, task_id, 1, 'task.started', config)
            atomic_json(directory / 'task.json', dict(task_id=task_id, config=config,
                start_event_id=event_id, next_sequence=2, started_at=time.time(), finished=False))
            return event_id

    def emit(self, task_id, kind, data):
        data = json_value(data)
        if type(data) is not dict or type(kind) is not str or not kind or kind in ('task.started', 'task.finished'):
            raise ValueError('emit requires an event kind and JSON object')
        directory = self.task_dir(task_id)
        with file_lock(directory / 'producer.lock'):
            task = read_json(directory / 'task.json')
            if task is None or task['finished']:
                raise ValueError('tracking task is missing or already finished')
            if any(read_json(path)['kind'] == 'task.finished' for path in (directory / 'events').glob('*.json')):
                raise ValueError('tracking task has a durable finish event')
            # The event file is the authoritative sequence log after a crash.
            sequence = max([int(path.name.split('-')[0]) for path in (directory / 'events').glob('*.json')], default=0) + 1
            event_id = self._event(directory, task_id, sequence, kind, data)
            task['next_sequence'] = sequence + 1
            atomic_json(directory / 'task.json', task)
            return event_id

    def finish_task(self, task_id, summary):
        summary = json_value(summary)
        if type(summary) is not dict: raise ValueError('summary must be a JSON object')
        directory = self.task_dir(task_id)
        with file_lock(directory / 'producer.lock'):
            task = read_json(directory / 'task.json')
            if task is None: raise ValueError('tracking task has not started')
            if task['finished']:
                if not json_equal(task['summary'], summary): raise ValueError('final tracking summary is frozen')
                return task['finish_event_id']
            orphan = [read_json(path) for path in (directory / 'events').glob('*.json') if read_json(path)['kind'] == 'task.finished']
            if orphan:
                if not json_equal(orphan[0]['data'], summary): raise ValueError('durable finish summary differs')
                event_id = orphan[0]['event_id']
            else:
                sequence = max(int(path.name.split('-')[0]) for path in (directory / 'events').glob('*.json')) + 1
                event_id = self._event(directory, task_id, sequence, 'task.finished', summary)
            task.update(finished=True, summary=summary, finish_event_id=event_id)
            atomic_json(directory / 'task.json', task)
            return event_id

    def ensure_online(self, task_id, timeout=30):
        if not cloud_required(self.root):
            return dict(mode='local_only', cloud_verified=False, user_waived=True)
        deadline = time.monotonic() + timeout
        directory = self.task_dir(task_id)
        while True:
            now = time.time()
            heartbeat = read_json(self.directory / 'heartbeat.json', {})
            status = read_json(directory / 'status.json', {})
            task = read_json(directory / 'task.json', {})
            if (task and not task.get('finished') and heartbeat.get('online') is True
                    and 0 <= now - heartbeat.get('updated_at', 0) <= HEARTBEAT_SECONDS
                    and status.get('online') is True and status.get('cloud_state', '').upper() == 'RUNNING'
                    and 0 <= now - status.get('api_verified_at', 0) <= API_FRESH_SECONDS
                    and (directory / 'acks' / (task['start_event_id'] + '.json')).exists()):
                return status['cloud_ref']
            if time.monotonic() >= deadline:
                raise OnlineUnavailable('SwanLab collector is offline, stale, or this task has no verified cloud Run')
            time.sleep(min(.2, max(0, deadline - time.monotonic())))

    def wait_verified(self, task_id, timeout=30):
        if not cloud_required(self.root):
            task = read_json(self.task_dir(task_id) / 'task.json', {})
            if not task.get('finished'):
                raise OnlineVerificationFailed('local task journal is not finished')
            return dict(verified=True, scope='local_only', cloud_verified=False, user_waived=True,
                finish_event_id=task['finish_event_id'], summary=task['summary'], cloud_ref=None)
        deadline = time.monotonic() + timeout
        while True:
            verified = read_json(self.task_dir(task_id) / 'verified.json')
            task = read_json(self.task_dir(task_id) / 'task.json', {})
            if verified and task.get('finished') and verified.get('finish_event_id') == task['finish_event_id']:
                return verified
            if time.monotonic() >= deadline:
                raise OnlineVerificationFailed('Cloud FINISHED state or final numeric summary is not verified; local outbox is retained')
            time.sleep(min(.2, max(0, deadline - time.monotonic())))


def numeric_metrics(data, prefix=''):
    result = {}
    items = data.items() if type(data) is dict else enumerate(data)
    for key, value in items:
        key = str(key).replace('~', '~0').replace('/', '~1')
        name = prefix + key
        if type(value) in (int, float, bool): result[name] = int(value) if type(value) is bool else value
        elif type(value) in (dict, list): result.update(numeric_metrics(value, name + '/'))
    return result


def event_metrics(event):
    data = event['data']
    model = str(data.get('training_model_id', data.get('model_id', 'unspecified'))).replace('/', '_')
    budget = str(data.get('budget', data.get('max_trials', 'unspecified'))).replace('/', '_')
    stage = event['kind'].replace('.', '/')
    prefix = ('summary/' if event['kind'] == 'task.finished' else f'model/{model}/budget/{budget}/{stage}/')
    metrics = numeric_metrics(data, prefix)
    metrics['tracking/event_sequence'] = event['sequence']
    if event['kind'] == 'budget.completed':
        for field in ('validation_balanced_accuracy', 'test_balanced_accuracy', 'test_accuracy', 'test_macro_f1'):
            if field in data:
                metrics[f'model/{model}/budget/{budget}/' + field] = data[field]
    if event['kind'] == 'lm.request':
        measurement = data.get('last_request_measurement', data.get('measurement', data))
        usage = data.get('usage', measurement.get('usage', measurement))
        for field, alias in (('input_tokens', 'prompt_tokens'), ('output_tokens', 'completion_tokens'),
                             ('total_tokens', 'total_tokens'), ('cached_tokens', 'cached_tokens')):
            value = usage.get(field, usage.get(alias))
            metrics[prefix + field + '_known_tokens_sum'] = value if type(value) is int and value >= 0 else 0
            metrics[prefix + field + '_known_requests'] = int(type(value) is int and value >= 0)
            metrics[prefix + field + '_missing_requests'] = int(not (type(value) is int and value >= 0))
        status = data.get('token_status', usage.get('status'))
        if status not in ('known', 'partial', 'unknown'):
            known = [metrics[prefix + field + '_known_requests'] for field in ('input_tokens', 'output_tokens', 'total_tokens')]
            status = 'known' if all(known) else 'partial' if any(known) else 'unknown'
        for value in ('known', 'partial', 'unknown'):
            metrics[prefix + 'usage_' + value + '_requests'] = int(status == value)
        metrics[prefix + 'request_count'] = 1
    return metrics


def summary_matches(cloud_summary, expected):
    return all(key in cloud_summary and type(cloud_summary[key].get('value')) in (int, float)
        and math.isclose(cloud_summary[key]['value'], value, rel_tol=1e-9, abs_tol=1e-9)
        for key, value in expected.items())


class SwanLabBackend:
    """One instance per task process, with one SDK Run instance."""
    def __init__(self, directory, task, project='AutoAI-benchmark-v3', workspace='ryfrcd693'):
        self.directory, self.task = Path(directory), task
        self.project, self.workspace = project, workspace
        self.run = None
        self.binding_path = self.directory / 'cloud-binding.json'

    def open(self):
        import swanlab
        binding = read_json(self.binding_path)
        if binding is None:
            run_id = uuid.uuid4().hex[:21]
            binding = dict(state='creating', run_id=run_id, path=f'{self.workspace}/{self.project}/{run_id}')
            atomic_json(self.binding_path, binding)
            resume = 'allow'
        else:
            # Any ambiguous prior init can only resume a positively found Run.
            cloud = swanlab.Api().run(binding['path'])
            if not cloud.state: raise OnlineUnavailable('Prior cloud initialization outcome is unknown')
            resume = 'must'
        candidate = swanlab.init(project=self.project, workspace=self.workspace, public=False,
            name=self.task['task_id'], mode='online', id=binding['run_id'], resume=resume,
            log_dir=str(self.directory / 'swanlog'), config=self.task['config'])
        if candidate.mode != 'online' or candidate.id != binding['run_id']:
            raise OnlineUnavailable('SwanLab did not return the requested online Run')
        self.run = candidate
        binding.update(state='bound', url=self.run.url)
        atomic_json(self.binding_path, binding)
        return binding

    def query(self):
        from swanlab import Api
        binding = read_json(self.binding_path)
        cloud = Api().run(binding['path'])
        return dict(state=cloud.state, summary=cloud.summary(),
                    cloud_ref={key: binding[key] for key in ('run_id', 'path', 'url')})

    def log(self, metrics, sequence):
        self.run.log(metrics, step=sequence)

    def finish(self):
        self.run.finish(async_log_timeout=30)


class TaskCollector:
    """Serial cloud acknowledgment for one durable task, testable with a backend."""
    def __init__(self, directory, backend):
        self.directory, self.backend = Path(directory), backend
        self.finished = False
        self.last_cloud = None
        self.last_verified = 0.0

    def offline(self, exc):
        atomic_json(self.directory / 'status.json', dict(online=False, updated_at=time.time(),
            error_class=type(exc).__name__, outbox_retained=True))

    def verify(self):
        cloud = self.backend.query()
        if not cloud['state']: raise OnlineUnavailable('Cloud Run has no readable state')
        atomic_json(self.directory / 'status.json', dict(online=True, updated_at=time.time(),
            api_verified_at=time.time(), cloud_ref=cloud['cloud_ref'], cloud_state=cloud['state']))
        self.last_cloud, self.last_verified = cloud, time.monotonic()
        return cloud

    def process_once(self):
        for path in sorted((self.directory / 'events').glob('*.json')):
            event = read_json(path)
            acknowledgment = self.directory / 'acks' / (event['event_id'] + '.json')
            if acknowledgment.exists(): continue
            metrics = event_metrics(event)
            prior_events = [read_json(prior_path) for prior_path in sorted((self.directory / 'events').glob('*.json'))
                            if int(prior_path.name.split('-')[0]) <= event['sequence']]
            if event['kind'] == 'lm.request':
                counters = {key: 0 for key in metrics if key.endswith(('_known_requests', '_missing_requests', '_unknown_requests', '_partial_requests', '_known_tokens_sum', '/request_count'))}
                global_counters = {}; seen = set()
                for prior in prior_events:
                    if prior['kind'] != 'lm.request': continue
                    measurement = prior['data'].get('last_request_measurement', prior['data'].get('measurement', prior['data']))
                    identity = (measurement.get('model_id'), measurement.get('request_started_at_utc', prior['event_id']), measurement.get('phase'), prior['data'].get('retry_index'))
                    if identity in seen: continue
                    seen.add(identity)
                    prior_metrics = event_metrics(prior)
                    for key in counters: counters[key] += prior_metrics.get(key, 0)
                    for key, value in prior_metrics.items():
                        if key.endswith(('_known_requests', '_missing_requests', '_unknown_requests', '_partial_requests', '_known_tokens_sum', '/request_count')):
                            global_key = 'accounting/lm/' + key.rsplit('/', 1)[-1]
                            global_counters[global_key] = global_counters.get(global_key, 0) + value
                metrics.update(counters); metrics.update(global_counters)
            if event['kind'] == 'fit.completed':
                totals = dict(actual_fits=0, started_fits=0, epochs=0, fit_duration_sum_seconds=0)
                selected = dict(totals); seen = set()
                for prior in prior_events:
                    if prior['kind'] != 'fit.completed': continue
                    data = prior['data']; identity = data.get('run_id', prior['event_id'])
                    if identity in seen: continue
                    seen.add(identity)
                    same = all(data.get(key) == event['data'].get(key) for key in ('model_id', 'budget'))
                    for field in totals:
                        value = data.get('wall_seconds' if field == 'fit_duration_sum_seconds' else field, 0)
                        if type(value) not in (int, float): continue
                        totals[field] += value
                        if same: selected[field] += value
                prefix = 'model/' + str(event['data'].get('model_id', 'unspecified')).replace('/', '_') + '/budget/' + str(event['data'].get('budget', 'unspecified')).replace('/', '_') + '/fit/completed/'
                metrics.update({prefix + key + '_cumulative': value for key, value in selected.items()})
                metrics.update({'accounting/fit/' + key: value for key, value in totals.items()})
            if event['kind'] == 'task.finished':
                task = read_json(self.directory / 'task.json')
                metrics['timing/task_wall_seconds'] = event['created_at'] - task['started_at']
            pending = self.directory / 'pending-upload.json'
            previous = read_json(pending)
            if previous is None or previous['event_id'] != event['event_id']:
                atomic_json(pending, dict(event_id=event['event_id'], metrics=metrics, sequence=event['sequence']))
                self.backend.log(metrics, event['sequence'])
                if event['kind'] == 'task.finished': self.backend.finish()
            else:
                # Query before retransmitting the same event/step after a crash.
                metrics = previous['metrics']
                cloud = self.verify()
                if not summary_matches(cloud['summary'], metrics):
                    self.backend.log(metrics, event['sequence'])
                if event['kind'] == 'task.finished' and cloud['state'].upper() != 'FINISHED':
                    self.backend.finish()
            cloud = self.verify()
            if not summary_matches(cloud['summary'], metrics):
                raise OnlineUnavailable('Uploaded event is not yet present in cloud scalar summary')
            if event['kind'] == 'task.finished':
                if cloud['state'].upper() != 'FINISHED':
                    raise OnlineUnavailable('Cloud Run has not reached FINISHED')
                atomic_json(self.directory / 'verified.json', dict(verified=True,
                    finish_event_id=event['event_id'], verified_at=time.time(), cloud_ref=cloud['cloud_ref'],
                    summary=event['data'], verified_numeric_summary=metrics, cloud_summary=cloud['summary']))
                self.finished = True
            atomic_json(acknowledgment, dict(event_id=event['event_id'], verified_at=time.time(),
                cloud_ref=cloud['cloud_ref'], metrics=metrics))
            pending.unlink(missing_ok=True)
        return self.verify() if time.monotonic() - self.last_verified >= 25 else self.last_cloud

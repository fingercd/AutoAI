"""Default durable timing for backend training, including failures and cache reuse."""
from __future__ import annotations

from contextvars import ContextVar
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
import inspect
import os
import time

ACTIVE_ACCOUNT = ContextVar('backend_experiment_account', default=None)


def search_budget(config):
    plan = config.get('execution_search_plan')
    return plan['max_trials'] if plan else config.get('max_trials')


def start_cuda_clock():
    import torch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    return start, end


class ExperimentAccount:
    def __init__(self, directory: Path, run_id: str, config: dict):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.config = config
        self.started = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.spans = []
        self.parents = []
        self.fits_started = self.fits_completed = self.epochs = self.batches = 0
        self.status = 'running'
        self.tracking_status = 'pending'
        self.tracking_error_type = None
        self.default_weight_ref = None
        self.cached_seconds = 0.0
        self.cached_final_fit_count = 0
        self.cached_final_fit_source_seconds = 0.0
        self.flush()

    def synchronize(self):
        if self.config.get('execution_device', 'cpu') == 'cuda':
            import torch
            from .execution_runtime import synchronize_device
            synchronize_device(torch.device('cuda:0'))

    def flush(self):
        from .search_accounting import _atomic_json
        _atomic_json(self.directory / 'experiment_timeline.json', {
            'schema_version': 'experiment-timeline-v1', 'run_id': self.run_id,
            'clock': 'process_monotonic', 'spans': self.spans})
        _atomic_json(self.directory / 'experiment_summary.json', {
            'schema_version': 'experiment-summary-v1', 'run_id': self.run_id, 'status': self.status,
            'measurement_scope': 'backend_training_run', 'started_at': self.started_at,
            'wall_seconds': time.monotonic()-self.started, 'model_fits_started': self.fits_started,
            'model_fits_completed': self.fits_completed, 'executed_epochs': self.epochs,
            'training_batches': self.batches, 'llm_requests_in_backend_process': 0,
            'default_weight_ref': self.default_weight_ref,
            'cached_seconds': self.cached_seconds,
            'cached_final_fit_count': self.cached_final_fit_count,
            'cached_final_fit_source_seconds': self.cached_final_fit_source_seconds,
            'tracking_status': self.tracking_status,
            'tracking_error_type': self.tracking_error_type,
            'llm_input_tokens_in_backend_process': 0, 'llm_output_tokens_in_backend_process': 0,
            'task_lm_cost_source': 'separate durable LM journal; backend zero is not a task-level zero'})

    @contextmanager
    def span(self, stage: str):
        started = time.monotonic()
        row = {'span_id': f'span-{len(self.spans)}', 'parent_span_id': self.parents[-1] if self.parents else None,
            'stage': stage, 'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
            'duration_seconds': None, 'gpu_elapsed_seconds': None,
            'gpu_timing_scope': 'CUDA event interval, includes device idle gaps and is not kernel active time'
                if self.config.get('execution_device') == 'cuda' else 'not_applicable'}
        self.spans.append(row)
        self.parents.append(row['span_id'])
        self.flush()
        body_failed = False
        cuda_clock = None
        try:
            self.synchronize()
            if self.config.get('execution_device') == 'cuda':
                cuda_clock = start_cuda_clock()
            yield row
        except BaseException as exc:
            body_failed = True
            row['status'] = 'failed'
            row['error_type'] = type(exc).__name__
            raise
        else:
            row['status'] = 'succeeded'
        finally:
            sync_error = None
            try:
                if cuda_clock is not None:
                    cuda_clock[1].record()
                self.synchronize()
                if cuda_clock is not None:
                    row['gpu_elapsed_seconds'] = cuda_clock[0].elapsed_time(cuda_clock[1]) / 1000.0
            except Exception as exc:
                row['status'] = 'failed'
                row['synchronization_error_type'] = type(exc).__name__
                sync_error = exc
            finally:
                row['duration_seconds'] = time.monotonic()-started
                row['ended_at'] = datetime.now(timezone.utc).isoformat()
                self.parents.pop()
                self.flush()
            if sync_error is not None and not body_failed:
                raise sync_error


def account():
    return ACTIVE_ACCOUNT.get()


def measured_model_fit(function, *args, **kwargs):
    current = account()
    if current is None:
        return function(*args, **kwargs)
    current.fits_started += 1
    with current.span('candidate_or_final_model_fit'):
        result = function(*args, **kwargs)
    current.fits_completed += 1
    current.flush()
    return result


def measured_deep_fit(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        current = account()
        if current is None:
            return function(*args, **kwargs)
        current.fits_started += 1
        fact = kwargs.get('trial_fact')
        try:
            with current.span('candidate_fit'):
                model, history = function(*args, **kwargs)
            current.fits_completed += 1
            if fact is None:
                current.epochs += len(history)
            return model, history
        finally:
            if fact is not None:
                current.epochs += fact['actual_epochs'] or 0
                current.batches += fact['training_batches'] or 0
            current.flush()
    return wrapped


def timed(stage):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            current = account()
            if current is None:
                return function(*args, **kwargs)
            with current.span(stage):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def record_training(function):
    signature = inspect.signature(function)
    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        values = bound.arguments
        config = dict(values.get('config_data') or {})
        run_id = values.get('run_id') or f'backend-{os.getpid()}-{time.time_ns()}'
        directory = values.get('output_dir')
        if directory is None:
            from .paths import RUNS_DIR
            directory = RUNS_DIR / run_id
        values['run_id'] = run_id
        values['output_dir'] = directory
        current = ExperimentAccount(Path(directory) / 'telemetry', run_id, config)
        token = ACTIVE_ACCOUNT.set(current)
        from .benchmark_cache import fitted_cache_scope
        try:
            context = config.get('benchmark_context')
            from .benchmark_tracking import OnlineJournal
            tracking_root = Path(context['suite_root']) if context else Path(os.environ.get(
                'AUTOAI_BENCHMARK_ROOT', Path(__file__).resolve().parents[2] / 'work' / 'benchmark-20261010'))
            task_id = context.get('tracking_task_id', context['task_id']) if context else run_id
            journal = OnlineJournal(tracking_root)
            if context is None:
                journal.start_task(task_id, {key: config[key] for key in (
                    'model_type','seed','normalization','class_balance','execution_device',
                    'cpu_threads','epochs','batch_size','max_trials') if key in config})
            journal.ensure_online(task_id, timeout=30)
            with fitted_cache_scope(context, {**config, '_cache_run_id': run_id}) as cache, current.span('total_training'):
                result = function(*bound.args, **bound.kwargs)
                if cache is not None and search_budget(config) == 1:
                    current.default_weight_ref = cache.latest_artifact
            current.status = 'succeeded'
            return result
        except BaseException:
            current.status = 'failed'
            raise
        finally:
            current.flush()
            try:
                context = config.get('benchmark_context')
                from .benchmark_tracking import OnlineJournal
                tracking_root = Path(context['suite_root']) if context else Path(os.environ.get(
                    'AUTOAI_BENCHMARK_ROOT', Path(__file__).resolve().parents[2] / 'work' / 'benchmark-20261010'))
                OnlineJournal(tracking_root).emit(context.get('tracking_task_id', context['task_id']) if context else run_id, 'fit.completed', {
                    'run_id': run_id, 'status': current.status, 'model_id': config.get('model_type','cnn1d'),
                    'budget': search_budget(config), 'round_id': context['round_id'] if context else None,
                    'actual_fits': current.fits_completed, 'started_fits': current.fits_started,
                    'epochs': current.epochs, 'wall_seconds': time.monotonic()-current.started})
                current.tracking_status = 'queued'
            except BaseException as exc:
                current.tracking_status = 'failed'
                current.tracking_error_type = type(exc).__name__
                raise
            finally:
                try:
                    current.flush()
                finally:
                    ACTIVE_ACCOUNT.reset(token)
    return wrapped


def finish_publication(run_directory: Path, config: dict, started: float, status='succeeded'):
    from .search_accounting import _atomic_json
    import json
    directory = Path(run_directory) / 'telemetry'
    summary_path = directory / 'experiment_summary.json'
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text())
    timeline = json.loads((directory / 'experiment_timeline.json').read_text())
    elapsed = time.monotonic() - started
    timeline['spans'].append({'span_id': f"span-{len(timeline['spans'])}",
        'parent_span_id': None, 'stage': 'artifact_publication', 'status': status,
        'duration_seconds': elapsed})
    summary.update(publication_seconds=elapsed, total_wall_seconds=summary['wall_seconds']+elapsed,
                   publication_status=status)
    _atomic_json(summary_path, summary)
    _atomic_json(directory / 'experiment_timeline.json', timeline)
    if not config.get('benchmark_context'):
        from .benchmark_tracking import OnlineJournal
        root = Path(os.environ.get('AUTOAI_BENCHMARK_ROOT', Path(__file__).resolve().parents[2]/'work'/'benchmark-20261010'))
        journal = OnlineJournal(root)
        try:
            journal.finish_task(summary['run_id'], summary)
            journal.wait_verified(summary['run_id'], timeout=120)
        except BaseException as exc:
            summary.update(tracking_status='failed', tracking_error_type=type(exc).__name__)
            _atomic_json(summary_path, summary)
            raise
        else:
            summary.update(tracking_status='verified', tracking_error_type=None)
            _atomic_json(summary_path, summary)

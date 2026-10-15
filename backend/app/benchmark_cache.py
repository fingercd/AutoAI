"""Verified private fitted-object reuse for the two approved budgets."""
from __future__ import annotations

from contextvars import ContextVar
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from functools import wraps
from pathlib import Path
import inspect
import io
import json
import pickle
import time

from .evaluation_plan import digest
from .benchmark_protocol import validate_context_reference

ACTIVE_CACHE = ContextVar('benchmark_fitted_cache', default=None)


class FittedCache:
    def __init__(self, context: dict, raw_config: dict):
        from .benchmark_resources import WeightStore
        validate_context_reference(context)
        self.context = context
        self.raw_config = raw_config
        self.store = WeightStore(Path(context['suite_root']) / 'weights')
        self.group_id = context['bundle_id']
        self.index = Path(context['suite_root']) / 'cache-index' / digest(self.group_id)
        self.index.mkdir(parents=True, exist_ok=True)
        self.latest_artifact = None
        self.final_cache_hit = False
        self.normalizer = None

    def key(self, kind: str, config) -> str:
        if hasattr(config, 'config'):
            config = config.config
        params = dict(config.__dict__)
        # Search budget belongs to the evaluator, not the fitted learner.
        params.pop('search_mode', None)
        params.pop('max_trials', None)
        return digest({'kind': kind, 'bundle': self.group_id, 'config': params,
            'source_digest': self.context['source_digest'],
            'dataset_sha256': self.raw_config.get('benchmark_dataset_sha256'),
            'evaluation_plan_digest': self.raw_config['evaluation_plan_digest']})

    def load(self, kind: str, config):
        started = time.monotonic()
        key = self.key(kind, config)
        path = self.index / f'{key}.json'
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        import hashlib
        # Validate every learned object before decoding the pickle. Torch's
        # pickle decoder may itself load tensor storage.
        for asset in [record] + ([record['checkpoint']] if 'checkpoint' in record else []):
            file = (self.store.root / asset['relative_path']).resolve()
            file.relative_to(self.store.root.resolve())
            if not file.is_file():
                raise ValueError('needed fitted checkpoint was removed before paired completion')
            with file.open('rb') as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != asset['sha256']:
                    raise ValueError('fitted checkpoint changed')
        file = self.store.root / record['relative_path']
        with file.open('rb') as handle:
            value = pickle.load(handle)
        if value['key'] != key:
            raise ValueError('fitted checkpoint binding differs')
        self.latest_artifact = record
        from .experiment_accounting import account
        if account() is not None:
            account().cached_seconds += time.monotonic() - started
        return value['payload']

    def save(self, kind: str, config, payload):
        key = self.key(kind, config)
        value = deepcopy(payload)
        if self.normalizer is not None:
            value.setdefault('normalizer', deepcopy(self.normalizer))
        model = value.get('model')
        if hasattr(model, 'cpu'):
            model.cpu()
            import torch
            state = model.state_dict()
            if any(not torch.isfinite(tensor).all().item() for tensor in state.values()):
                raise ValueError('cannot cache nonfinite learned parameters')
            relative_checkpoint = f'checkpoints/{key}.pt'
            size = sum(tensor.numel() * tensor.element_size() for tensor in state.values()) * 2 + 1024**2
            # Torch emits many writes; serialize on CPU before one quota-checked payload write.
            with io.BytesIO() as buffer:
                torch.save(state, buffer)
                checkpoint_bytes = buffer.getvalue()
            checkpoint = self.store.save(relative_checkpoint, lambda handle: handle.write(checkpoint_bytes),
                expected_bytes=size, group_id=self.group_id)
            value['checkpoint'] = {**checkpoint, 'relative_path': relative_checkpoint}
        encoded = pickle.dumps({'key': key, 'payload': value}, protocol=pickle.HIGHEST_PROTOCOL)
        relative = f'cache/{key}.pkl'
        artifact = self.store.save(relative, lambda handle: handle.write(encoded),
            expected_bytes=len(encoded), group_id=self.group_id)
        record = {**artifact, 'relative_path': relative, 'key': key, 'kind': kind,
                  'source_run_id': self.raw_config.get('_cache_run_id')}
        if 'checkpoint' in value:
            record['checkpoint'] = value['checkpoint']
        path = self.index / f'{key}.json'
        from .search_accounting import _atomic_json
        _atomic_json(path, record)
        self.latest_artifact = record
        return record['sha256']


def active_cache():
    return ACTIVE_CACHE.get()


@contextmanager
def fitted_cache_scope(context, raw_config):
    cache = FittedCache(context, raw_config) if context else None
    token = ACTIVE_CACHE.set(cache)
    try:
        yield cache
    finally:
        ACTIVE_CACHE.reset(token)


def cache_deep_fit(function):
    signature = inspect.signature(function)
    @wraps(function)
    def wrapped(*args, **kwargs):
        cache = active_cache()
        if cache is None:
            return function(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        config = values['config']
        payload = cache.load('deep_candidate', config)
        fact = values.get('trial_fact')
        if payload is not None:
            model = payload['model']
            from .execution_runtime import load_deep_checkpoint
            import hashlib
            checkpoint = payload['checkpoint']
            path = cache.store.root / checkpoint['relative_path']
            with path.open('rb') as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != checkpoint['sha256']:
                    raise ValueError('learned tensor checkpoint changed')
            from .experiment_accounting import account
            from .search_accounting import _atomic_json
            current = account()
            if current is None:
                load_deep_checkpoint(model, path, execution_device=config.execution_device)
            else:
                started = time.monotonic()
                with current.span('checkpoint_reload'):
                    load_deep_checkpoint(model, path, execution_device=config.execution_device)
                current.cached_seconds += time.monotonic() - started
            if current is not None:
                _atomic_json(current.directory / 'checkpoint_reload.json', {
                    'status': 'succeeded', 'source_checkpoint': checkpoint,
                    'requested_device': config.execution_device,
                    'loaded_device': str(next(model.parameters()).device),
                    'dtype': str(next(model.parameters()).dtype),
                    'source_fit_seconds': payload['fit_seconds'], 'additional_fits': 0})
            if fact is not None:
                fact.update(cached=True, source_fit_seconds=payload['fit_seconds'],
                    source_epochs=payload['actual_epochs'], actual_epochs=payload['actual_epochs'],
                    training_batches=payload['training_batches'], executed_epochs=0,
                    executed_training_batches=0)
            return model, payload['history']
        started = time.monotonic()
        model, history = function(*args, **kwargs)
        import math
        if not history or not math.isfinite(float(history[-1]['best_valid_loss'])):
            raise ValueError('completed deep checkpoint has no finite validation evidence')
        payload = {'model': model, 'history': history, 'fit_seconds': time.monotonic()-started,
                   'actual_epochs': len(history), 'training_batches': fact['training_batches'] if fact else None}
        cache.save('deep_candidate', config, payload)
        if fact is not None:
            fact.update(cached=False, executed_epochs=fact['actual_epochs'],
                executed_training_batches=fact['training_batches'])
        return model, history
    return wrapped


def cache_final_fit(function):
    signature = inspect.signature(function)
    @wraps(function)
    def wrapped(*args, **kwargs):
        cache = active_cache()
        if cache is None:
            return function(*args, **kwargs)
        values = signature.bind(*args, **kwargs).arguments
        config = values['selected_config']
        payload = cache.load('traditional_final', config)
        cache.final_cache_hit = payload is not None
        if payload is not None:
            from .experiment_accounting import account
            if account() is not None:
                account().cached_final_fit_count += 1
                account().cached_final_fit_source_seconds += payload['fit_seconds']
            return payload['model'], payload['normalizer'], payload['indices']
        started = time.monotonic()
        model, normalizer, indices = function(*args, **kwargs)
        cache.save('traditional_final', config, {'model': model, 'normalizer': normalizer,
            'indices': indices, 'fit_seconds': time.monotonic()-started})
        return model, normalizer, indices
    return wrapped

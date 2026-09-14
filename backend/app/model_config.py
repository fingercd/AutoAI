"""Finite, versioned controls over existing training algorithms.

Search grids and network profiles remain owned by the trainer. This module
only describes their resolution policy and the inputs operators may freeze.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
from typing import Any

from .classification_policy import DEEP_TRAINING_DEFAULTS
from .model_catalog import MODELS_BY_ID, MODEL_DECLARATIONS, ARCHITECTURE_VERSION, model_availability


@dataclass(frozen=True)
class TraditionalTrainingDefaults:
    random_forest_n_estimators: int = 200
    random_forest_search_iterations: int = 10
    xgboost_gamma: float = 0.0


TRADITIONAL_TRAINING_DEFAULTS = TraditionalTrainingDefaults()
CONFIG_POLICY_VERSION = 'agent-model-config-v1'
CATALOG_VERSION = 'agent-model-catalog-v1'


def semantic_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _parameter(name, kind, default, *, minimum=None, maximum=None, exclusive_minimum=None, exclusive_maximum=None, choices=None):
    return dict(**({'exclusive_maximum': exclusive_maximum} if exclusive_maximum is not None else {}), name=name, value_type=kind, nullable=False, default_status='fixed', default=default,
                minimum=minimum, maximum=maximum, exclusive_minimum=exclusive_minimum, choices=choices,
                role='operator_fixed', applies_to=[], condition_refs=[])


def model_policy(model_id: str) -> dict[str, Any]:
    model = MODELS_BY_ID[model_id]
    parameters = []
    rules = []
    if model.implemented and model.execution_family == 'deep_learning':
        for name, default in asdict(DEEP_TRAINING_DEFAULTS).items():
            if name == 'seed':
                continue
            if type(default) is int:
                parameters.append(_parameter(name, 'integer', default, minimum=1))
            else:
                parameters.append(_parameter(name, 'number', default,
                    minimum=0 if name == 'weight_decay' else None,
                    exclusive_minimum=None if name == 'weight_decay' else 0,
                    exclusive_maximum=1 if name == 'scheduler_factor' else None))
        rules.append('train-only-profile-v2')
    defaults = TRADITIONAL_TRAINING_DEFAULTS
    if model_id == 'random_forest':
        parameters += [_parameter('random_forest_n_estimators', 'integer', defaults.random_forest_n_estimators, minimum=50, maximum=1000),
                       _parameter('random_forest_search_iterations', 'integer', defaults.random_forest_search_iterations, minimum=1, maximum=18)]
        rules.append('rf-oob-balanced-accuracy-accuracy-stable-order-v1')
    elif model_id == 'xgboost':
        parameters.append(_parameter('xgboost_gamma', 'number', defaults.xgboost_gamma, minimum=0))
        rules.append('xgboost-twelve-candidate-validation-search-v2')
    elif model.execution_family == 'traditional_ml':
        rules.append(model_id + '-train-bounded-validation-search-v2')
    for parameter in parameters:
        parameter['applies_to'] = [model_id]
    policy = dict(config_policy_version='agent-model-config-v2' if model.implemented and model.execution_family == 'deep_learning' else CONFIG_POLICY_VERSION,
        fixed_execution_defaults={p['name']: p['default'] for p in parameters},
        parameters=parameters, compatibility_rules=rules)
    policy['config_policy_digest'] = semantic_digest(policy)
    return policy


def resolve_model_params(model_id: str, overrides: dict[str, Any] | None = None,
                         *, policy: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = model_policy(model_id) if policy is None else policy
    if overrides is None:
        overrides = {}
    if type(overrides) is not dict:
        raise ValueError('model_params must be an object')
    specs = {p['name']:p for p in policy['parameters'] if p['role'] == 'operator_fixed'}
    if set(overrides) - set(specs):
        raise ValueError('unknown or inapplicable model parameter')
    resolved = dict(policy['fixed_execution_defaults'])
    for name, value in overrides.items():
        spec = specs[name]
        kind = spec['value_type']
        if value is None or (kind == 'integer' and type(value) is not int) or (
            kind == 'number' and type(value) not in (float, int)) or (
            kind == 'string' and type(value) is not str):
            raise ValueError('invalid parameter type: ' + name)
        if kind in ('integer', 'number'):
            if not math.isfinite(value):
                raise ValueError('parameter must be finite: ' + name)
            for key, invalid in [('minimum', lambda x: value < x), ('maximum', lambda x: value > x),
                                  ('exclusive_minimum', lambda x: value <= x), ('exclusive_maximum', lambda x: value >= x)]:
                if spec.get(key) is not None and invalid(spec[key]):
                    raise ValueError('parameter out of range: ' + name)
        if spec['choices'] is not None and value not in spec['choices']:
            raise ValueError('invalid parameter choice: ' + name)
        resolved[name] = float(value) if kind == 'number' else value
    return resolved


def model_capability_snapshot() -> dict[str, Any]:
    models = []
    semantic = []
    availability = []
    for m in MODEL_DECLARATIONS:
        item = dict(id=m.id, display_name=m.display_name, family=m.family,
                    execution_family=m.execution_family, architecture_version=ARCHITECTURE_VERSION,
                    implemented=m.implemented, **model_policy(m.id))
        semantic.append(dict(item))
        available, reason = model_availability(m.id)
        availability.append(dict(id=m.id, available=available, reason_code=reason))
        item.update(available=available, reason_code=reason, availability_basis='declaration-module-dependency-probe')
        models.append(item)
    return dict(catalog_version=CATALOG_VERSION, catalog_digest=semantic_digest(semantic),
                availability_digest=semantic_digest(availability), models=models)


def compatible_frozen_policy(model_id, frozen, params, *, current=None):
    """Only the known v1 scheduler boundary correction is execution compatible."""
    from copy import deepcopy
    current = model_policy(model_id) if current is None else current
    if frozen['config_policy_digest'] != current['config_policy_digest']:
        if current['config_policy_version'] != 'agent-model-config-v2':
            return False
        legacy = deepcopy(current)
        legacy.pop('config_policy_digest')
        legacy['config_policy_version'] = 'agent-model-config-v1'
        for parameter in legacy['parameters']:
            if parameter['name'] == 'scheduler_factor':
                parameter.pop('exclusive_maximum')
                parameter['maximum'] = 1
        if frozen['config_policy_digest'] != semantic_digest(legacy):
            return False
    try:
        return resolve_model_params(model_id, params) == params
    except ValueError:
        return False

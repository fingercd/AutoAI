"""Minimal submission and integrity-checked execution projections for v2."""
import json
import math
from pathlib import Path
from typing import Literal
from pydantic import Field, ValidationError
from .metadata import SafeEffectiveConfig, dataset_metadata
from ..model_catalog import MODELS_BY_ID, RETIRED_MODEL_ALIASES
from ..model_config import model_policy, resolve_model_params
from ..runs.artifacts import RunArtifactWriter, ManifestCorruptError, ArtifactIntegrityError


class SafeEffectiveConfigV2(SafeEffectiveConfig):
    model_type: str
    normalization: Literal['zscore']
    class_balance: Literal['none']
    feature_selection_enabled: Literal[False]
    config_stage: Literal['submission']
    config_policy_version: str
    config_policy_digest: str = Field(pattern=r'^[a-f0-9]{64}$')
    model_params: dict[str, int | float | str]


def run_metadata_v2(record, *, pending=False, snapshot=None):
    response = dataset_metadata(record.dataset_snapshot.get('sha256') if record else None, version='agent-session-v2')
    response.update(effective_config_status='pending' if pending else 'unavailable', effective_config=None)
    if record is None:
        return response
    model_id = record.config.get('model_type')
    if model_id not in MODELS_BY_ID and model_id not in RETIRED_MODEL_ALIASES:
        return response
    raw = record.config
    policy = next((m for m in snapshot['models'] if m['id'] == raw['model_type']), None) if snapshot else (model_policy(model_id) if model_id in MODELS_BY_ID else None)
    if policy is None:
        return response
    if raw.get('agent_config_policy_digest') != policy['config_policy_digest']:
        return response
    try:
        params = {name:raw[name] for name in policy['fixed_execution_defaults']}
        params = resolve_model_params(raw['model_type'], params, policy=policy)
        config = SafeEffectiveConfigV2.model_validate({
            **{key:raw.get(key) for key in ('model_type','normalization','class_balance','seed','feature_selection_enabled')},
            'evaluation_config': dict(mode=raw.get('split_mode'), train_weight=raw.get('split_train'),
                                     validation_weight=raw.get('split_valid'), heldout_weight=raw.get('split_test')),
            'config_stage':'submission', 'config_policy_version':raw.get('agent_config_policy_version'),
            'config_policy_digest':raw.get('agent_config_policy_digest'), 'model_params':params,
        })
    except (ValidationError, ValueError, KeyError):
        return response
    response.update(effective_config_status='ready', effective_config=config.model_dump(mode='json'))
    return response


_NUMERIC = frozenset(('pls_components','pca_components','logistic_c','svm_c','random_forest_n_estimators',
    'random_forest_max_depth','random_forest_min_samples_leaf','xgboost_n_estimators','xgboost_max_depth',
    'xgboost_learning_rate','xgboost_subsample','xgboost_colsample_bytree','xgboost_reg_lambda',
    'xgboost_min_child_weight','xgboost_gamma','dropout','n_components','heads','layers',
    'embedding_dim','hidden_size','kernel_size','pool_size','num_layers','num_heads'))
_ARRAY = frozenset(('channels','kernel_sizes','pool_sizes','hidden_sizes','dilations'))
_CHOICES = {'execution_device':('cpu','cuda:0'), 'svm_kernel':('linear',), 'svm_gamma':('scale',), 'random_forest_max_features':('sqrt','log2'),
            'dscarnet_input_mode':('sar','car','dual'), 'classification_head':('binary_single_logit','multiclass_logits'),
            'loss_function':('BCEWithLogitsLoss','CrossEntropyLoss')}


def _scalars(raw):
    if type(raw) is not dict:
        return {}
    result = {}
    for key, value in raw.items():
        if key in _NUMERIC and type(value) in (int,float) and math.isfinite(value):
            result[key] = value
        elif key in _ARRAY and type(value) is list and len(value) <= 16 and all(type(n) is int and n > 0 for n in value):
            result[key] = value
        elif key in _CHOICES and value in _CHOICES[key]:
            result[key] = value
        elif key == 'random_forest_oob_score' and type(value) is bool:
            result[key] = value
    return result


def resolved_execution(run_dir: Path, record):
    empty = dict(status='pending' if record.state in ('queued','running') else 'unavailable', parameters=None)
    if record.state != 'succeeded':
        return empty
    from .observation import validation_from_manifest
    if validation_from_manifest(run_dir, run_id=record.run_id)[0] != 'ready':
        return empty
    writer = RunArtifactWriter(run_dir)
    try:
        def read(name):
            return json.loads(writer.resolve_download(name).read_text(encoding='utf-8-sig'))
        model = read('model_metadata.json')
        if model.get('model_type') != record.config['model_type']:
            return empty
        params = _scalars(model)
        profile = model.get('model_profile')
        declaration = MODELS_BY_ID.get(record.config['model_type'])
        if type(profile) is dict and ((declaration and declaration.execution_family == 'deep_learning') or record.config['model_type'] in RETIRED_MODEL_ALIASES):
            params.update(_scalars(profile))
            params.update(_scalars(profile.get('values')))
        folds = read('split.json')
        if type(folds) is not list or len(folds) != 1:
            return empty
        params.update(_scalars(folds[0].get('best_params')))
        return dict(status='ready', parameters=params) if params else empty
    except (FileNotFoundError, ManifestCorruptError, ArtifactIntegrityError, OSError, UnicodeError, ValueError, KeyError, TypeError):
        return empty

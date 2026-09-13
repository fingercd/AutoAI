"""Versioned, path-free projections of authoritative Session and Run metadata."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..runs.contracts import RunRecord
from .contracts import AGENT_METADATA_VERSION, AgentDomainError


_SHA256 = re.compile(r'^[0-9a-f]{64}$')


class SafeEvaluationConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    mode: Literal['stratified_holdout']
    train_weight: int = Field(ge=1, le=9)
    validation_weight: int = Field(ge=1, le=9)
    heldout_weight: int = Field(ge=1, le=9)


class SafeEffectiveConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    model_type: Literal['logistic_regression', 'svm', 'random_forest']
    normalization: Literal['zscore', 'minmax', 'area', 'none']
    class_balance: Literal['none', 'class_weight']
    seed: int = Field(ge=0)
    feature_selection_enabled: bool
    evaluation_config: SafeEvaluationConfig


def dataset_metadata(digest: str | None, *, version: str = 'agent-session-v1') -> dict[str, Any]:
    if digest is not None and not _SHA256.fullmatch(digest):
        raise AgentDomainError(
            'agent_metadata_invalid', '数据指纹元数据不完整，需要核对', status_code=409,
        )
    return {
        'metadata_version': 'agent-metadata-v2' if version == 'agent-session-v2' else AGENT_METADATA_VERSION,
        'dataset_sha256': digest,
        'dataset_fingerprint_status': 'ready' if digest is not None else 'unavailable',
    }


def run_metadata(record: RunRecord | None, *, pending: bool = False, version: str = 'agent-session-v1', snapshot: dict | None = None) -> dict[str, Any]:
    """Read the persisted Run snapshot, never reconstruct an effective config from actions."""
    if version == 'agent-session-v2':
        from .metadata_v2 import run_metadata_v2
        return run_metadata_v2(record, pending=pending, snapshot=snapshot)
    digest = record.dataset_snapshot.get('sha256') if record is not None else None
    if digest is not None and not isinstance(digest, str):
        raise AgentDomainError('agent_metadata_invalid', '数据指纹元数据不完整，需要核对', status_code=409)
    response = dataset_metadata(digest)
    response.update(effective_config_status='pending' if pending else 'unavailable', effective_config=None)
    if record is None:
        return response
    raw = record.config
    # Only explicit scalar fields from the already validated, persisted config cross the boundary.
    projection = {
        key: raw.get(key) for key in (
            'model_type', 'normalization', 'class_balance', 'seed', 'feature_selection_enabled',
        )
    }
    projection['evaluation_config'] = {
        'mode': raw.get('split_mode'),
        'train_weight': raw.get('split_train'),
        'validation_weight': raw.get('split_valid'),
        'heldout_weight': raw.get('split_test'),
    }
    try:
        config = SafeEffectiveConfig.model_validate(projection)
    except ValidationError:
        # Historical Runs can lack an effective config; never infer the missing values.
        return response
    if sum((config.evaluation_config.train_weight, config.evaluation_config.validation_weight,
            config.evaluation_config.heldout_weight)) != 10:
        return response
    response.update(effective_config_status='ready', effective_config=config.model_dump(mode='json'))
    return response

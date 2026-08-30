"""Fail-fast semantic and output guards for Agent-created training Runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..classification_split import stratified_group_holdout_indices
from ..parsers import load_modeling_csv


GUARD_SCHEMA_VERSION = 'agent-guard-v1'
MAX_GUARD_CELLS = 2_000_000
MAX_GUARD_FILE_BYTES = 16 * 1024 * 1024


class AgentGuardRejected(ValueError):
    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__('agent guard rejected')
        self.result = result


def _result(
    *,
    stage: str,
    status: str,
    checks: list[dict[str, str]],
    model_fit_count: int,
) -> dict[str, Any]:
    return {
        'schema_version': GUARD_SCHEMA_VERSION,
        'stage': stage,
        'status': status,
        'retryable': False,
        'checks': checks,
        'model_fit_count': int(model_fit_count),
    }


def _reject(
    stage: str,
    checks: list[dict[str, str]],
    code: str,
    *,
    model_fit_count: int = 0,
) -> None:
    checks.append({'code': code, 'status': 'failed'})
    raise AgentGuardRejected(
        _result(
            stage=stage,
            status='rejected',
            checks=checks,
            model_fit_count=model_fit_count,
        )
    )


def run_preflight_guard(
    data_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Reject invalid data/config before the training callback can fit a model."""
    checks: list[dict[str, str]] = []
    envelope = config.get('agent_execution')
    if not isinstance(envelope, dict) or envelope.get('guard_version') != GUARD_SCHEMA_VERSION:
        _reject('preflight', checks, 'guard_contract')
    if envelope.get('fail_fast_guard') is not True:
        _reject('preflight', checks, 'guard_contract')
    if config.get('model_type') != envelope.get('expected_model_type'):
        _reject('preflight', checks, 'task_semantics')
    checks.append({'code': 'guard_contract', 'status': 'passed'})

    if data_path.stat().st_size > MAX_GUARD_FILE_BYTES:
        _reject('preflight', checks, 'resource_limit')
    try:
        dataset = load_modeling_csv(data_path)
    except Exception:
        _reject('preflight', checks, 'data_contract')
    checks.append({'code': 'data_contract', 'status': 'passed'})

    values = np.asarray(dataset.intensity)
    if values.ndim != 2 or values.size > MAX_GUARD_CELLS:
        _reject('preflight', checks, 'resource_limit')
    if not np.isfinite(values).all():
        _reject('preflight', checks, 'finite_values')
    checks.extend([
        {'code': 'resource_limit', 'status': 'passed'},
        {'code': 'finite_values', 'status': 'passed'},
    ])

    label_names = sorted(set(dataset.labels))
    if len(label_names) < 2:
        _reject('preflight', checks, 'classification_semantics')
    label_to_id = {label: index for index, label in enumerate(label_names)}
    y = np.asarray([label_to_id[label] for label in dataset.labels], dtype=np.int64)
    sample_ids = dataset.frame['Sample_ID'].astype(str).to_numpy()
    try:
        stratified_group_holdout_indices(
            y,
            sample_ids,
            seed=int(config.get('seed', 42)),
            split_train=int(config.get('split_train', 8)),
            split_valid=int(config.get('split_valid', 1)),
            split_test=int(config.get('split_test', 1)),
            label_names=label_names,
        )
    except Exception:
        _reject('preflight', checks, 'split_feasibility')
    checks.extend([
        {'code': 'classification_semantics', 'status': 'passed'},
        {'code': 'split_feasibility', 'status': 'passed'},
    ])

    profile = str(config.get('hpo_profile') or 'standard')
    maximum = {'off': 1, 'tiny': 3, 'standard': 18}.get(profile)
    if maximum is None or maximum > int(envelope.get('max_hpo_candidates', 18)):
        _reject('preflight', checks, 'hpo_budget')
    checks.append({'code': 'hpo_budget', 'status': 'passed'})
    return _result(
        stage='preflight',
        status='passed',
        checks=checks,
        model_fit_count=0,
    )


def run_postflight_guard(
    result: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate the safe output contract before result artifacts are finalized."""
    checks: list[dict[str, str]] = []
    envelope = config.get('agent_execution')
    metadata = result.get('model_metadata') if isinstance(result, dict) else None
    raw_fit_count = (
        metadata.get('model_fit_count')
        if isinstance(metadata, dict)
        else None
    )
    observed_fit_count = (
        max(0, raw_fit_count)
        if isinstance(raw_fit_count, int) and not isinstance(raw_fit_count, bool)
        else 0
    )
    if not isinstance(result, dict) or result.get('status') != 'success':
        _reject(
            'postflight', checks, 'result_contract',
            model_fit_count=observed_fit_count,
        )
    if not isinstance(envelope, dict) or (
        result.get('model_type') != envelope.get('expected_model_type')
    ):
        _reject(
            'postflight', checks, 'task_semantics',
            model_fit_count=observed_fit_count,
        )
    if not isinstance(result.get('metrics'), dict):
        _reject(
            'postflight', checks, 'metrics_contract',
            model_fit_count=observed_fit_count,
        )
    model_fit_count = (
        metadata.get('model_fit_count')
        if isinstance(metadata, dict)
        else None
    )
    if (
        isinstance(model_fit_count, bool)
        or not isinstance(model_fit_count, int)
        or model_fit_count < 1
    ):
        _reject(
            'postflight', checks, 'cost_contract',
            model_fit_count=observed_fit_count,
        )
    checks.extend([
        {'code': 'result_contract', 'status': 'passed'},
        {'code': 'task_semantics', 'status': 'passed'},
        {'code': 'metrics_contract', 'status': 'passed'},
        {'code': 'cost_contract', 'status': 'passed'},
    ])
    return _result(
        stage='postflight',
        status='passed',
        checks=checks,
        model_fit_count=model_fit_count,
    )

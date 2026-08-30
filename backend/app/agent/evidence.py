"""Train-only, path-safe evidence for small-sample Agent decisions."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..classification_split import (
    SPLITTER_VERSION,
    stratified_group_holdout_indices,
)
from ..parsers import load_modeling_csv


EVIDENCE_SCHEMA_VERSION = 'small-sample-evidence-v1'
MAX_EVIDENCE_FILE_BYTES = 16 * 1024 * 1024
MAX_EVIDENCE_OBSERVATIONS = 5_000
MAX_EVIDENCE_CELLS = 2_000_000


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def build_train_evidence_card(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_id_train: np.ndarray,
    *,
    split_fingerprint: str,
    data_format: str,
) -> dict[str, Any]:
    """Compute anonymous evidence from train arrays only."""
    x = np.asarray(x_train)
    y = np.asarray(y_train, dtype=np.int64)
    groups = np.asarray(sample_id_train, dtype=str)
    if x.ndim != 2 or y.ndim != 1 or groups.ndim != 1:
        raise ValueError('evidence arrays have invalid dimensions')
    if len(x) == 0 or len(x) != len(y) or len(x) != len(groups):
        raise ValueError('evidence arrays have inconsistent lengths')
    if not np.isfinite(x).all():
        raise ValueError('evidence predictors must be finite')

    observation_count, predictor_count = x.shape
    if observation_count > MAX_EVIDENCE_OBSERVATIONS:
        raise ValueError('evidence observation limit exceeded')
    if x.size > MAX_EVIDENCE_CELLS:
        raise ValueError('evidence matrix limit exceeded')
    class_values, class_counts = np.unique(y, return_counts=True)
    class_distribution = {
        f'class_{position}': {
            'count': int(count),
            'ratio': round(float(count / observation_count), 6),
        }
        for position, (_, count) in enumerate(zip(class_values, class_counts))
    }
    group_values, group_counts = np.unique(groups, return_counts=True)
    zero_variance_count = int(np.sum(np.ptp(x, axis=0) == 0.0))

    row_counts: dict[bytes, int] = {}
    row_labels: dict[bytes, set[int]] = {}
    for row, label in zip(x, y):
        contiguous = np.ascontiguousarray(row)
        row_digest = hashlib.blake2b(
            contiguous.view(np.uint8),
            digest_size=16,
        ).digest()
        row_counts[row_digest] = row_counts.get(row_digest, 0) + 1
        row_labels.setdefault(row_digest, set()).add(int(label))
    duplicate_observation_count = sum(
        max(count - 1, 0) for count in row_counts.values()
    )
    conflicting_duplicate_groups = sum(
        1
        for digest, count in row_counts.items()
        if count > 1 and len(row_labels[digest]) > 1
    )

    n_over_p = round(float(observation_count / max(1, predictor_count)), 6)
    risks: list[str] = []
    minimum_class_ratio = float(np.min(class_counts) / observation_count)
    if minimum_class_ratio < 0.35:
        risks.append('class_imbalance')
    if n_over_p < 5.0:
        risks.append('high_dimension')
    if observation_count < 100:
        risks.append('very_small_train_partition')
    if zero_variance_count:
        risks.append('zero_variance_predictors')
    if duplicate_observation_count:
        risks.append('duplicate_predictors')
    if conflicting_duplicate_groups:
        risks.append('conflicting_duplicate_predictors')

    statistics = {
        'observation_count': int(observation_count),
        'group_count': int(len(group_values)),
        'predictor_count': int(predictor_count),
        'n_over_p': n_over_p,
        'class_distribution': class_distribution,
        'repeat_measurements': {
            'minimum': int(np.min(group_counts)),
            'maximum': int(np.max(group_counts)),
            'uniform': bool(np.min(group_counts) == np.max(group_counts)),
        },
        'zero_variance_predictor_count': zero_variance_count,
        'duplicate_observation_count': duplicate_observation_count,
        'conflicting_duplicate_group_count': int(conflicting_duplicate_groups),
        'missing_value_count': 0,
        'non_finite_value_count': 0,
        'risk_codes': sorted(risks),
    }
    return {
        'schema_version': EVIDENCE_SCHEMA_VERSION,
        'status': 'ready',
        'scope': 'train_only',
        'splitter_version': SPLITTER_VERSION,
        'split_fingerprint': split_fingerprint,
        'data_format': data_format,
        'statistics': statistics,
        'evidence_digest': _digest({
            'split_fingerprint': split_fingerprint,
            'statistics': statistics,
        }),
    }


def build_dataset_evidence_card(
    path: Path,
    *,
    dataset_digest: str,
    seed: int,
    evaluation_config: dict[str, Any],
) -> dict[str, Any]:
    """Validate and cache a bounded registered dataset evidence build."""
    if path.stat().st_size > MAX_EVIDENCE_FILE_BYTES:
        raise ValueError('evidence file limit exceeded')
    return deepcopy(_build_dataset_evidence_card_cached(
        str(path.resolve()),
        dataset_digest,
        int(seed),
        int(evaluation_config['split_train']),
        int(evaluation_config['split_valid']),
        int(evaluation_config['split_test']),
    ))


@lru_cache(maxsize=32)
def _build_dataset_evidence_card_cached(
    path_text: str,
    dataset_digest: str,
    seed: int,
    split_train: int,
    split_valid: int,
    split_test: int,
) -> dict[str, Any]:
    """Private cache keyed by the verified full-file digest; digest is not returned."""
    path = Path(path_text)
    dataset = load_modeling_csv(path)
    if len(dataset.labels) > MAX_EVIDENCE_OBSERVATIONS:
        raise ValueError('evidence observation limit exceeded')
    if int(np.asarray(dataset.intensity).size) > MAX_EVIDENCE_CELLS:
        raise ValueError('evidence matrix limit exceeded')
    label_names = sorted(set(dataset.labels))
    label_to_id = {label: index for index, label in enumerate(label_names)}
    y = np.asarray(
        [label_to_id[label] for label in dataset.labels],
        dtype=np.int64,
    )
    sample_id = dataset.frame['Sample_ID'].astype(str).to_numpy()
    splits = stratified_group_holdout_indices(
        y,
        sample_id,
        seed=seed,
        split_train=split_train,
        split_valid=split_valid,
        split_test=split_test,
        label_names=label_names,
    )
    train_indices = np.asarray(splits['train'], dtype=np.int64)
    split_fingerprint = _digest({
        'splitter_version': SPLITTER_VERSION,
        'seed': seed,
        'ratios': {
            'train': split_train,
            'valid': split_valid,
            'held_out': split_test,
        },
        'splits': {
            name: sorted(int(index) for index in indices)
            for name, indices in sorted(splits.items())
        },
    })
    return build_train_evidence_card(
        np.asarray(dataset.intensity)[train_indices],
        y[train_indices],
        sample_id[train_indices],
        split_fingerprint=split_fingerprint,
        data_format=dataset.data_format,
    )

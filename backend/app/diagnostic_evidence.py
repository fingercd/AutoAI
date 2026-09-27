"""Private Train/Valid provenance. Never fits, loads models or accepts Test evaluations."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

METRICS = ('macro_f1', 'balanced_accuracy', 'accuracy')
Digest = Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]
Count = Annotated[int, Field(strict=True, ge=0)]
Score = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def producer_digest() -> str:
    root = Path(__file__).parent
    return hashlib.sha256(b''.join((root / name).read_bytes()
        for name in ('diagnostic_evidence.py', 'training.py'))).hexdigest()


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)


class Metrics(Strict):
    macro_f1: Score
    balanced_accuracy: Score
    accuracy: Score


class ClassSupport(Strict):
    class_id: Count
    observation_count: Count
    sample_group_count: Count

    @model_validator(mode='after')
    def coherent(self):
        if self.sample_group_count > self.observation_count:
            raise ValueError('invalid group support')
        return self


class Support(Strict):
    observation_count: Count
    sample_group_count: Count
    class_support: list[ClassSupport]

    @model_validator(mode='after')
    def coherent(self):
        if ([row.class_id for row in self.class_support] != list(range(len(self.class_support)))
                or sum(row.observation_count for row in self.class_support) != self.observation_count
                or sum(row.sample_group_count for row in self.class_support) != self.sample_group_count):
            raise ValueError('invalid support totals')
        return self


class Evaluation(Strict):
    evaluation_snapshot_id: Digest
    partition_role: Literal['train', 'valid']
    indices_digest: Digest
    metric_definition: Literal['classification-observation-v1'] = 'classification-observation-v1'
    aggregation: Literal['direct_fold'] = 'direct_fold'
    metrics: Metrics
    support: Support


class FoldAudit(Strict):
    fold_index: Annotated[int, Field(strict=True, ge=1)]
    partition_digest: Digest
    selected_trial_index: Count | None
    logical_snapshot_id: Digest
    snapshot_kind: Literal['selection_model_instance'] = 'selection_model_instance'
    model_id: str
    fit_scope: Literal['train'] = 'train'
    fit_indices_digest: Digest
    preprocessing_digest: Digest
    selection_criterion: Literal['balanced_accuracy', 'oob_balanced_accuracy', 'best_valid_loss']
    best_epoch: Annotated[int, Field(strict=True, ge=1)] | None
    train: Evaluation
    valid: Evaluation
    selection_reuses_validation: bool

    @model_validator(mode='after')
    def coherent(self):
        if (self.train.partition_role != 'train' or self.valid.partition_role != 'valid'
                or self.train.evaluation_snapshot_id != self.logical_snapshot_id
                or self.valid.evaluation_snapshot_id != self.logical_snapshot_id
                or self.train.indices_digest != self.fit_indices_digest
                or len(self.train.support.class_support) != len(self.valid.support.class_support)
                or self.selection_reuses_validation != (self.selection_criterion != 'oob_balanced_accuracy')):
            raise ValueError('unpaired evaluation provenance')
        return self


class TrainingValidationAudit(Strict):
    schema_version: Literal['training-validation-audit-v1'] = 'training-validation-audit-v1'
    status: Literal['ready', 'unavailable']
    reason_code: Literal['audit_input_unavailable'] | None = None
    producer_source_digest: Digest
    config_digest: Digest
    evaluation_strategy: Literal['stratified_holdout', 'leave_one_sample_id_cv', 'external_test_holdout']
    folds: list[FoldAudit]

    @model_validator(mode='after')
    def coherent(self):
        if self.status == 'ready':
            if not self.folds or self.reason_code is not None:
                raise ValueError('empty audit')
            if [f.fold_index for f in self.folds] != list(range(1, len(self.folds) + 1)):
                raise ValueError('incomplete folds')
        elif self.folds or self.reason_code is None:
            raise ValueError('unavailable audit has observations')
        return self


def collect_fold(*, run_id, fold_index, model_id, train_indices, valid_indices,
                 labels, sample_groups, class_count, train_eval, valid_eval,
                 processing_stage, selected_trial_index, selection_criterion, best_epoch):
    """Called with both evaluations while the selected Train-only object is alive."""
    train_indices, valid_indices = list(map(int, train_indices)), list(map(int, valid_indices))
    if set(train_indices) & set(valid_indices):
        raise ValueError('overlapping Train/Valid partitions')
    identity = digest(dict(run_id=run_id, fold_index=fold_index, trial=selected_trial_index,
                           stage='selected_train_only', best_epoch=best_epoch))

    def evaluation(role, indices, values):
        supports = []
        for class_id in range(class_count):
            rows = [i for i in indices if int(labels[i]) == class_id]
            supports.append(dict(class_id=class_id, observation_count=len(rows),
                sample_group_count=len({str(sample_groups[i]) for i in rows})))
        return Evaluation(evaluation_snapshot_id=identity, partition_role=role,
            indices_digest=digest(sorted(indices)), metrics={k: values[k] for k in METRICS},
            support=dict(observation_count=len(indices),
                sample_group_count=len({str(sample_groups[i]) for i in indices}), class_support=supports))

    return FoldAudit(fold_index=fold_index,
        partition_digest=digest(dict(train=train_indices, valid=valid_indices)),
        selected_trial_index=selected_trial_index, logical_snapshot_id=identity, model_id=model_id,
        fit_indices_digest=digest(sorted(train_indices)), preprocessing_digest=digest(processing_stage),
        selection_criterion=selection_criterion, best_epoch=best_epoch,
        train=evaluation('train', train_indices, train_eval),
        valid=evaluation('valid', valid_indices, valid_eval),
        selection_reuses_validation=selection_criterion != 'oob_balanced_accuracy')


def paired_metrics(documents: dict) -> dict:
    """Consume only the caller's verified artifact snapshot; never open a path.

    A malformed optional audit removes comparability, never candidate eligibility.
    Counts are per fold. No cross-fold pooled independent support is constructed.
    """
    missing = dict(status='unavailable', reason_code='audit_missing', pairs=[], support=[])
    raw = documents.get('training_validation_audit.json')
    if raw is None:
        return missing
    try:
        audit = TrainingValidationAudit.model_validate(raw)
        if audit.status != 'ready':
            return {**missing, 'reason_code': audit.reason_code}
        config = documents['config.json']
        folds = documents['split.json']
        if (audit.config_digest != digest(config) or audit.producer_source_digest != producer_digest()
                or audit.evaluation_strategy != config['evaluation_strategy']
                or len(audit.folds) != len(folds) or len(folds) != config['fold_count']):
            raise ValueError('audit binding mismatch')
        selected = {row['fold_index']: row for row in documents.get('search_summary.json', {}).get('selected', [])}
        pairs, supports = [], []
        for actual, source in zip(audit.folds, folds):
            splits = source['splits']
            if (actual.fold_index != source['fold_index'] or actual.model_id != config['model_type']
                    or actual.partition_digest != digest({k: splits[k] for k in ('train', 'valid')})
                    or actual.preprocessing_digest != digest(source['processing_execution']['stages'][0])
                    or actual.fit_indices_digest != source['processing_execution']['stages'][0]['fit_indices_digest']):
                raise ValueError('fold provenance mismatch')
            criterion = source['selection_metric'] or 'best_valid_loss'
            if (actual.selection_criterion != criterion
                    or len(actual.train.support.class_support) != len(documents['label_map.json'])
                    or (actual.best_epoch is not None and actual.best_epoch > config['epochs'])
                    or (criterion == 'best_valid_loss') != (actual.best_epoch is not None)):
                raise ValueError('selection provenance mismatch')
            if selected:
                if actual.selected_trial_index != selected[actual.fold_index]['trial_index']:
                    raise ValueError('trial mismatch')
            elif actual.selected_trial_index is not None:
                raise ValueError('unexpected trial')
            for role in ('train', 'valid'):
                evaluation = getattr(actual, role)
                if (evaluation.indices_digest != digest(sorted(splits[role]))
                        or evaluation.support.observation_count != len(splits[role])
                        or evaluation.support.sample_group_count != len(source[role + '_sample_ids'])):
                    raise ValueError('partition support mismatch')
                if any(row.observation_count == 0 or row.sample_group_count == 0
                       for row in evaluation.support.class_support):
                    return {**missing, 'reason_code': 'class_support_incomplete'}
                for metric in METRICS:
                    value = source['split_metrics'][role][metric]
                    if type(value) not in (int, float) or not math.isfinite(value) or not math.isclose(
                            getattr(evaluation.metrics, metric), value, rel_tol=0, abs_tol=1e-12):
                        raise ValueError('metric mismatch')
            for metric in METRICS:
                train, valid = getattr(actual.train.metrics, metric), getattr(actual.valid.metrics, metric)
                pairs.append(dict(fold_index=actual.fold_index, metric=metric, train=train, valid=valid,
                    delta=train-valid, aggregation='direct_fold', same_snapshot=True, fit_scope='train'))
            supports.append(dict(fold_index=actual.fold_index, train=actual.train.support.model_dump(),
                                 valid=actual.valid.support.model_dump()))
        for role in ('train', 'valid'):
            for metric in METRICS:
                value = documents['metrics.json'][role][metric]
                expected = sum(getattr(getattr(f, role).metrics, metric) for f in audit.folds) / len(audit.folds)
                if type(value) not in (int, float) or not math.isfinite(value) or not math.isclose(
                        value, expected, rel_tol=0, abs_tol=1e-12):
                    raise ValueError('aggregate mismatch')
        return dict(status='ready', reason_code=None, pairs=pairs, support=supports)
    except (ValueError, KeyError, TypeError, IndexError):
        return {**missing, 'reason_code': 'audit_invalid'}

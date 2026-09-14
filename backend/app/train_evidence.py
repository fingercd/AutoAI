"""Train-only, deterministic evidence. No Dataset, database, network or LLM API."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .evaluation_plan import TrainView, PreparationLimits, digest

STATISTICS_VERSION = 'train-statistics-v1'
RISK_VERSION = 'train-risk-rules-v1'


class ClassCount(BaseModel):
    model_config = ConfigDict(extra='forbid',strict=True)
    class_id: int
    observation_count: int
    sample_group_count: int
    observation_fraction: float
    sample_group_fraction: float


class TrainStatistics(BaseModel):
    model_config = ConfigDict(extra='forbid',strict=True)
    observation_count: int
    sample_group_count: int
    feature_count: int
    classes: list[ClassCount]
    repeated_measurement_group_count: int
    group_size_min: int
    group_size_max: int
    group_size_median: float
    unequal_repeats: bool
    zero_variance_feature_count: int
    duplicate_feature_group_count: int
    redundant_feature_count: int
    duplicate_vector_group_count: int
    duplicate_vector_observation_count: int
    cross_sample_duplicate_vector_group_count: int
    conflicting_vector_group_count: int
    conflicting_vector_observation_count: int


class Risk(BaseModel):
    model_config = ConfigDict(extra='forbid',strict=True)
    code: Literal['small_sample','high_dimension','imbalance','zero_variance','duplicate_features','conflicting_vectors','repeated_measurement']
    severity: Literal['warning','informational']


class TrainEvidence(BaseModel):
    model_config = ConfigDict(extra='forbid',strict=True,frozen=True)
    schema_version: Literal['train-evidence-v1'] = 'train-evidence-v1'
    statistics_version: Literal['train-statistics-v1'] = STATISTICS_VERSION
    risk_version: Literal['train-risk-rules-v1'] = RISK_VERSION
    status: Literal['ready'] = 'ready'
    scope: Literal['train'] = 'train'
    input_representation: Literal['raw-pre-normalization-float32'] = 'raw-pre-normalization-float32'
    dataset_sha256: str
    plan_digest: str
    train_content_digest: str
    statistics_digest: str
    evidence_digest: str
    statistics: TrainStatistics
    risks: list[Risk]


    @model_validator(mode='after')
    def evidence_content_digest(self):
        objective=dict(statistics=self.statistics.model_dump(mode='json'),risks=[r.model_dump() for r in self.risks],
            statistics_version=self.statistics_version,risk_version=self.risk_version)
        body=dict(dataset_sha256=self.dataset_sha256,plan_digest=self.plan_digest,
            train_content_digest=self.train_content_digest,statistics_digest=self.statistics_digest,**objective)
        if self.statistics_digest!=digest(objective) or self.evidence_digest!=digest(body):
            raise ValueError('evidence content mismatch')
        return self


def _equivalence_groups(x, *, columns: bool, limits):
    """Hashes select buckets; exact comparison is always the final authority."""
    buckets=defaultdict(list)
    size=x.shape[1] if columns else x.shape[0]
    for start in range(0,size,limits.block_size):
        limits.check(x.nbytes*2)
        for index in range(start,min(start+limits.block_size,size)):
            vector = x[:,index] if columns else x[index]
            key=hashlib.sha256(vector.tobytes()).digest()
            for group in buckets[key]:
                reference=x[:,group[0]] if columns else x[group[0]]
                if np.array_equal(vector,reference):
                    group.append(index);break
            else:
                buckets[key].append([index])
    return [g for candidates in buckets.values() for g in candidates if len(g)>1]


def compute_train_evidence(train: TrainView, limits: PreparationLimits | None = None) -> TrainEvidence:
    limits=limits or PreparationLimits.configured()
    limits.check(train.x.nbytes*3)
    x=np.array(train.x,dtype=np.float32,copy=True,order='C')
    if x.ndim!=2 or not all(x.shape) or not np.isfinite(x).all():
        raise ValueError('invalid_train_features')
    n,p=x.shape
    if len(train.labels)!=n or len(train.sample_ids)!=n:
        raise ValueError('invalid_train_metadata')
    x[x==0]=0 # -0 and +0 have identical execution semantics.
    groups=Counter(train.sample_ids);group_labels={}
    for group,label in zip(train.sample_ids,train.labels):
        if group in group_labels and group_labels[group]!=label:raise ValueError('mixed_group_labels')
        group_labels[group]=label
    rows=Counter(train.labels);counts=Counter(group_labels.values())
    classes=[ClassCount(class_id=i,observation_count=rows[label],sample_group_count=counts[label],
        observation_fraction=float(rows[label]/n),sample_group_fraction=float(counts[label]/len(groups)))
        for i,label in enumerate(sorted(rows))]
    if len(classes)<2:raise ValueError('train_missing_classes')
    duplicate_columns=_equivalence_groups(x,columns=True,limits=limits)
    duplicate_rows=_equivalence_groups(x,columns=False,limits=limits)
    conflicts=[g for g in duplicate_rows if len({train.labels[i] for i in g})>1]
    stats=TrainStatistics(observation_count=n,sample_group_count=len(groups),feature_count=p,classes=classes,
        repeated_measurement_group_count=sum(v>1 for v in groups.values()),
        group_size_min=min(groups.values()),group_size_max=max(groups.values()),
        group_size_median=float(np.median(list(groups.values()))),unequal_repeats=len(set(groups.values()))>1,
        zero_variance_feature_count=int(np.all(x==x[0],axis=0).sum()),
        duplicate_feature_group_count=len(duplicate_columns),redundant_feature_count=sum(len(g)-1 for g in duplicate_columns),
        duplicate_vector_group_count=len(duplicate_rows),duplicate_vector_observation_count=sum(map(len,duplicate_rows)),
        cross_sample_duplicate_vector_group_count=sum(len({train.sample_ids[i] for i in g})>1 for g in duplicate_rows),
        conflicting_vector_group_count=len(conflicts),conflicting_vector_observation_count=sum(map(len,conflicts)))
    rules=[('small_sample',min(counts.values())<5),('high_dimension',p>len(groups)),
        ('imbalance',max(counts.values())/min(counts.values())>2),
        ('zero_variance',stats.zero_variance_feature_count>0),('duplicate_features',bool(duplicate_columns)),
        ('conflicting_vectors',bool(conflicts))]
    risks=[Risk(code=name,severity='warning') for name,applies in rules if applies]
    if stats.repeated_measurement_group_count:risks.append(Risk(code='repeated_measurement',severity='informational'))
    objective=dict(statistics=stats.model_dump(mode='json'),risks=[r.model_dump() for r in risks],
        statistics_version=STATISTICS_VERSION,risk_version=RISK_VERSION)
    h=hashlib.sha256();h.update(x.tobytes());h.update(digest([train.labels,train.sample_ids,train.axis]).encode())
    body=dict(dataset_sha256=train.dataset_sha256,plan_digest=train.plan_digest,
        train_content_digest=h.hexdigest(),statistics_digest=digest(objective),**objective)
    result=TrainEvidence(**body,evidence_digest=digest(body))
    limits.check()
    if len(result.model_dump_json().encode())>65536:
        raise ValueError('evidence_summary_too_large')
    return result

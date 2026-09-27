"""Deterministic training checks. This module neither fits nor books budget."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .artifacts import ARTIFACT_CATALOG, MANIFEST_SCHEMA_VERSION, RunArtifactWriter

RULES_VERSION = 'training-guard-rules-v1'
# Freeze the implementation sources as well as the public rules name.
RULES_DIGEST = hashlib.sha256(b''.join((Path(__file__).parent.parent / name).read_bytes()
    for name in ('runs/guard.py', 'training.py', 'parsers.py', 'evaluation_plan.py',
                 'search_policy.py', 'model_config.py', 'diagnostic_evidence.py'))).hexdigest()
Stage = Literal['admission', 'pre_fit', 'pre_publish', 'publication', 'observation', 'finalize']
Status = Literal['passed', 'failed', 'pending', 'unavailable', 'not_applicable', 'disabled']
Reason = Literal['guard_dataset_invalid', 'guard_dataset_changed', 'guard_split_invalid',
    'guard_config_mismatch', 'guard_capability_unavailable', 'guard_search_plan_invalid',
    'guard_resource_limit', 'guard_execution_fence_invalid', 'guard_required_artifact_missing',
    'guard_manifest_invalid', 'guard_artifact_integrity_failed', 'guard_metric_invalid',
    'guard_execution_audit_mismatch', 'guard_usage_mismatch', 'guard_usage_unresolved',
    'guard_process_exit_unconfirmed', 'guard_report_missing', 'guard_report_binding_mismatch',
    'guard_check_unavailable']
METRICS = ('accuracy', 'balanced_accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1')


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class GuardPolicy(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    schema_version: Literal['training-guard-policy-v1'] = 'training-guard-policy-v1'
    rules_version: Literal['training-guard-rules-v1'] = RULES_VERSION
    rules_digest: str = Field(default_factory=lambda: RULES_DIGEST, pattern=r'^[a-f0-9]{64}$')
    fail_fast_guard: Literal['on', 'off'] = 'on'


def require_current_policy(value) -> GuardPolicy:
    """Execution admission, separate from historical storage/wire decoding."""
    try:
        policy = GuardPolicy.model_validate(value)
    except ValueError as exc:
        raise GuardError('guard_report_binding_mismatch') from exc
    if policy.rules_digest != RULES_DIGEST:
        raise GuardError('guard_check_unavailable')
    return policy


class GuardCheck(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    code: Literal['inputs', 'outputs', 'usage', 'publication']
    status: Status
    reason_code: Reason | None = None


class GuardBindings(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    run_id: str | None = None
    scope_digest: str = Field(pattern=r'^[a-f0-9]{64}$')
    config_digest: str = Field(pattern=r'^[a-f0-9]{64}$')
    dataset_digest: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')
    policy_digest: str = Field(pattern=r'^[a-f0-9]{64}$')
    fence_digest: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')
    manifest_digest: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')
    usage_digest: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')


class GuardReport(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    schema_version: Literal['training-guard-report-v1'] = 'training-guard-report-v1'
    rules_version: Literal['training-guard-rules-v1'] = RULES_VERSION
    report_id: str = Field(pattern=r'^guard-[a-f0-9]{64}$')
    stage: Stage
    status: Status
    eligibility: Literal['pending', 'eligible', 'ineligible', 'unavailable']
    bindings: GuardBindings
    checks: list[GuardCheck] = Field(min_length=1)
    created_at: str

    @model_validator(mode='after')
    def coherent(self):
        if self.status == 'disabled' and self.stage != 'admission':
            raise ValueError('only optional admission can be disabled')
        if self.status == 'passed' and any(c.status not in ('passed', 'not_applicable') for c in self.checks):
            raise ValueError('incomplete checks cannot pass')
        if self.eligibility == 'eligible' and (self.status != 'passed' or self.stage not in ('publication', 'observation', 'finalize')):
            raise ValueError('not a published candidate')
        body = self.model_dump(exclude={'report_id', 'created_at'})
        if self.report_id != 'guard-' + digest(body):
            raise ValueError('report identity mismatch')
        return self


class GuardError(ValueError):
    def __init__(self, code: Reason, *, stage: Stage = 'pre_fit', report_id: str | None = None):
        # Validate all child-envelope strings before they can reach an API.
        GuardCheck(code='inputs', status='failed', reason_code=code)
        self.code, self.stage, self.report_id = code, stage, report_id
        super().__init__('Training integrity check failed')


class GuardProjection(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    schema_version: Literal['agent-guard-projection-v1']
    report_id: str = Field(pattern=r'^guard-[a-f0-9]{64}$')
    stage: Stage
    status: Status
    eligibility: Literal['pending', 'eligible', 'ineligible', 'unavailable']
    checks: list[GuardCheck]


def report(record, stage: Stage, *, reason: Reason | None = None,
           manifest_digest=None, usage_digest=None, status: Status | None = None):
    status = status or ('failed' if reason else 'passed')
    policy = record.guard_policy or GuardPolicy(fail_fast_guard='off').model_dump()
    bindings = GuardBindings(run_id=record.run_id,
        scope_digest=digest([record.owner_id, record.tenant_id]),
        config_digest=digest(record.config), dataset_digest=record.dataset_snapshot.get('sha256'),
        policy_digest=digest(policy),
        fence_digest=digest(record.claim_token) if record.claim_token else None,
        manifest_digest=manifest_digest, usage_digest=usage_digest)
    body = dict(schema_version='training-guard-report-v1', rules_version=RULES_VERSION,
        stage=stage, status=status,
        eligibility=('unavailable' if status == 'unavailable' else 'ineligible' if reason else 'eligible' if stage in ('publication', 'observation', 'finalize') and status == 'passed' else 'pending'),
        bindings=bindings.model_dump(), checks=[dict(code='inputs' if stage in ('pre_fit','admission') else 'outputs', status=status, reason_code=reason)])
    return GuardReport(**body, report_id='guard-'+digest(body), created_at=datetime.now(timezone.utc).isoformat())


def project_guard(value: GuardReport) -> dict:
    # Neither artifact names, scope, Test facts, budget quantities nor exception prose.
    return dict(schema_version='agent-guard-projection-v1', report_id=value.report_id,
        stage=value.stage, status=value.status, eligibility=value.eligibility,
        checks=[dict(code=c.code, status=c.status,
            reason_code=('guard_artifact_integrity_failed' if c.reason_code and c.reason_code != 'guard_check_unavailable' and value.stage not in ('pre_fit','admission') else c.reason_code)) for c in value.checks])


def load_bound_dataset(path, expected_sha=None):
    from ..evaluation_plan import PreparationLimits
    from ..parsers import load_modeling_csv
    limits = PreparationLimits.configured()
    path = Path(path)
    limits.check(path.stat().st_size)
    with path.open('rb') as handle:
        content = handle.read(limits.max_bytes + 1)
    limits.check(len(content))
    sha = hashlib.sha256(content).hexdigest()
    if expected_sha is not None and sha != expected_sha:
        raise GuardError('guard_dataset_changed')
    data = load_modeling_csv(path, content=content)
    limits.check()
    return data, sha


def check_preflight(*, record, repository, prepare: Callable, stage: Stage = 'pre_fit'):
    try:
        if record is not None and record.guard_policy is not None:
            require_current_policy(record.guard_policy)
        prepared = prepare()
    except Exception as exc:
        from ..contracts import TrainingConfigValidationError
        from ..datasets.repository import DatasetIntegrityError
        from ..model_catalog import ModelNotImplementedForVersion, ModelRetiredError
        from ..evaluation_plan import PreparationResourceExhausted
        reason = (exc.code if isinstance(exc, GuardError) else
            'guard_dataset_changed' if isinstance(exc, DatasetIntegrityError) else
            'guard_capability_unavailable' if isinstance(exc, (ModelNotImplementedForVersion, ModelRetiredError)) else
            'guard_config_mismatch' if isinstance(exc, TrainingConfigValidationError) else
            'guard_resource_limit' if isinstance(exc, PreparationResourceExhausted) else
            'guard_dataset_invalid' if isinstance(exc, ValueError) else 'guard_check_unavailable')
        if record is not None and record.guard_policy is not None:
            value = report(record, stage, reason=reason)
            repository.save_guard_report(value)
            raise GuardError(reason, stage=stage, report_id=value.report_id) from exc
        raise
    if record is not None and record.guard_policy is not None:
        repository.save_guard_report(report(record, stage))
    return prepared


@dataclass(frozen=True)
class ArtifactSnapshot:
    manifest_digest: str
    documents: dict


def inspect_artifacts(run_dir: Path, run_id: str, *, required: set[str], active=lambda: None, manifest_bytes=None) -> ArtifactSnapshot:
    """Read JSON and hash the same bytes; stream private models without loading them."""
    writer = RunArtifactWriter(run_dir)
    deadline = time.monotonic() + 60
    def check():
        active()
        if time.monotonic() > deadline:
            raise GuardError('guard_resource_limit')
    check()
    path = writer._path('manifest.json')
    if path.is_symlink() or path.resolve().parent != writer.run_dir:
        raise GuardError('guard_manifest_invalid')
    if manifest_bytes is None:
        with path.open('rb') as handle:
            raw = handle.read(4 * 1024 * 1024 + 1)
    else:
        raw = manifest_bytes
    if len(raw) > 4 * 1024 * 1024:
        raise GuardError('guard_resource_limit')
    manifest = json.loads(raw)
    if (type(manifest) is not dict or manifest.get('schema_version') != MANIFEST_SCHEMA_VERSION
            or manifest.get('run_id') != run_id or type(manifest.get('artifacts')) is not dict):
        raise GuardError('guard_manifest_invalid')
    entries = manifest['artifacts']
    if not required <= entries.keys():
        raise GuardError('guard_required_artifact_missing')
    documents = {}
    for name, entry in entries.items():
        check()
        if (not isinstance(name, str) or '/' in name or '\\' in name or ':' in name or name in ('.','..','manifest.json')
                or not name or type(entry) is not dict):
            raise GuardError('guard_manifest_invalid')
        if ARTIFACT_CATALOG.get(name, {}).get('volatile'):
            continue
        size, sha = entry.get('size_bytes'), entry.get('sha256')
        if type(size) is not int or size < 0 or not isinstance(sha, str) or not re.fullmatch('[a-f0-9]{64}', sha) or entry.get('volatile'):
            raise GuardError('guard_manifest_invalid')
        target = writer._path(name)
        if target.is_symlink() or target.resolve().parent != writer.run_dir:
            raise GuardError('guard_artifact_integrity_failed')
        hasher = hashlib.sha256()
        count = 0
        chunks = [] if name.endswith('.json') else None
        with target.open('rb') as handle:
            while block := handle.read(1024 * 1024):
                check()
                count += len(block)
                if count > size or (chunks is not None and count > 64 * 1024 * 1024):
                    raise GuardError('guard_resource_limit')
                hasher.update(block)
                if chunks is not None:
                    chunks.append(block)
        if count != size or hasher.hexdigest() != sha:
            raise GuardError('guard_artifact_integrity_failed')
        if chunks is not None:
            documents[name] = json.loads(b''.join(chunks).decode('utf-8-sig'))
    check()
    return ArtifactSnapshot(hashlib.sha256(raw).hexdigest(), documents)


def required_artifacts(config: dict) -> set[str]:
    from ..models import model_family
    deep = model_family(config['model_type']) == 'deep_learning'
    names = {'metrics.json', 'cv_metrics.json', 'fold_metrics.csv', 'predictions.csv', 'cv_predictions.csv',
        'model_metadata.json', 'label_map.json', 'split.json', 'config.json',
        'model.pt' if deep else 'model.pkl'}
    if deep:
        names.add('history.csv')
    if deep or config.get('feature_selection_enabled', True):
        names.update(('sample_feature_importance.json', 'sample_feature_importance.csv'))
    if config.get('execution_search_plan') is not None:
        names.update(('search_plan.json', 'search_trials.json', 'search_trials.csv', 'search_summary.json', 'search_timeline.json'))
    return names


def validation_metrics(payload, selection_metric=None, *, complete=False):
    if type(payload) is not dict or type(payload.get('valid')) is not dict:
        raise GuardError('guard_metric_invalid')
    result = {}
    for key in METRICS:
        value = payload['valid'].get(key)
        if value is None and key not in payload['valid'] and not complete and key != selection_metric:
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise GuardError('guard_metric_invalid')
        result[key] = float(value)
    if not result or (selection_metric and selection_metric not in result):
        raise GuardError('guard_metric_invalid')
    return result


def check_fit_audit(fold, actual, *, finite_search=False, selected_params=None):
    """Validate facts against trainer semantics, never against another projection alone."""
    from ..models import model_family
    from ..training import TrainConfig, _traditional_params
    from ..contracts import TrainingSpec
    from ..model_config import semantic_digest
    from ..processing_policy import PROCESSING_POLICY_VERSION

    def reject():
        raise GuardError('guard_execution_audit_mismatch')

    def indices(value, *, empty=False):
        if (type(value) is not list or (not empty and not value) or
                any(type(i) is not int or i < 0 for i in value) or len(set(value)) != len(value)):
            reject()
        return set(value)

    traditional = model_family(actual['model_type']) == 'traditional_ml'
    external = actual['evaluation_strategy'] == 'external_test_holdout'
    splits = fold.get('splits')
    if type(splits) is not dict or set(splits) != {'train', 'valid', 'test'}:
        reject()
    train, valid = indices(splits['train']), indices(splits['valid'])
    test = indices(splits['test'], empty=external)
    outside = indices(fold.get('external_test_indices'), empty=not external)
    if (train & valid or train & test or valid & test or
            (external and test) or (not external and outside)):
        reject()
    internal = train | valid | test
    if internal != set(range(len(internal))) or outside & internal:
        reject()
    if external and outside != set(range(len(internal), len(internal) + len(outside))):
        reject()
    final = indices(fold.get('final_fit_indices'), empty=not traditional)
    if final != (train | valid if traditional else set()):
        reject()

    best = fold.get('best_params')
    defaults = TrainConfig()
    keys = (set(_traditional_params(defaults, actual['model_type'])) if traditional else
            {'learning_rate', 'weight_decay'} if finite_search else set())
    if type(best) is not dict or set(best) != keys:
        reject()
    # Validate observed values without filling in absent execution evidence.
    TrainingSpec.from_legacy({**actual, **best}).validated(has_external_test=external)
    if any(type(v) is bool or (isinstance(v, float) and not math.isfinite(v)) for v in best.values()):
        reject()
    if selected_params is not None and (not keys <= selected_params.keys() or
            any(best[k] != selected_params[k] for k in keys)):
        reject()

    processing = fold.get('processing_execution', {})
    expected = [('selection_train', train), ('final_train_valid', train | valid)] if traditional else [('train', train)]
    stages = processing.get('stages')
    if (type(stages) is not list or len(stages) != len(expected) or
            any(processing.get(k) != actual[k] for k in ('normalization', 'class_balance'))):
        reject()
    for stage, (name, rows) in zip(stages, expected, strict=True):
        scope = {'area': 'per_sample', 'none': 'identity'}.get(actual['normalization'], name)
        if (stage.get('stage') != name or stage.get('fit_scope') != scope or
                stage.get('policy_version') != PROCESSING_POLICY_VERSION or
                type(stage.get('fit_count')) is not int or stage['fit_count'] != len(rows) or
                stage.get('fit_indices_digest') != semantic_digest(sorted(rows))):
            reject()


def check_output_semantics(snapshot, record, repository):
    from ..training import TrainConfig
    from ..models import ARCHITECTURE_VERSION, model_family
    docs = snapshot.documents
    metrics = docs['metrics.json']
    validation = validation_metrics(metrics, complete=True)
    validation_metrics({'valid': metrics}, complete=True)
    for split in ('train', 'test'):
        validation_metrics({'valid': metrics.get(split)}, complete=True)
    actual = docs['config.json']
    metadata = docs['model_metadata.json']
    config = record.config
    for key, default in TrainConfig().__dict__.items():
        if key.startswith('resolved_'):
            continue
        if actual.get(key) != config.get(key, default):
            raise GuardError('guard_config_mismatch')
    if (actual.get('architecture_version') != ARCHITECTURE_VERSION or
            metadata.get('architecture_version') != ARCHITECTURE_VERSION or
            metadata.get('model_type') != actual.get('model_type') or
            metadata.get('model_family') != model_family(actual['model_type'])):
        raise GuardError('guard_config_mismatch')
    cv = docs['cv_metrics.json']
    folds = docs['split.json']
    if type(folds) is not list or not folds or [f.get('fold_index') for f in folds] != list(range(1, len(folds) + 1)):
        raise GuardError('guard_execution_audit_mismatch')
    if cv.get('metrics') != metrics or cv.get('folds') != folds or cv.get('strategy') != actual.get('evaluation_strategy'):
        raise GuardError('guard_execution_audit_mismatch')
    is_cv = actual.get('evaluation_strategy') == 'leave_one_sample_id_cv'
    aggregation = 'pooled_out_of_fold' if is_cv else 'direct_holdout'
    summary = cv.get('cv_summary', {})
    if (cv.get('fold_count') != len(folds) or summary.get('fold_count') != len(folds)
            or metrics.get('aggregation') != aggregation or metrics['test'].get('aggregation') != aggregation
            or summary.get('primary_test_aggregation') != aggregation
            or any(metrics.get(k) != metrics['test'][k] or metrics['test'][k] != summary.get('pooled_test', {}).get(k) for k in METRICS)):
        raise GuardError('guard_execution_audit_mismatch')
    audit = metadata.get('execution_audit', {})
    if (audit.get('initial_fit_scope') != 'train' or audit.get('final_fit_scope') !=
            ('train+valid' if metadata['model_family'] == 'traditional_ml' else 'train')):
        raise GuardError('guard_execution_audit_mismatch')
    if config.get('evaluation_plan_digest'):
        from .contracts import Principal
        plan = repository.get_evaluation_plan(config['evaluation_plan_digest'],
            principal=Principal(record.owner_id, record.tenant_id))
        if (len(folds) != 1 or folds[0].get('splits') != plan.indices or
                folds[0].get('partition_digest') != plan.partition_digest or
                audit.get('partition_digest') != plan.partition_digest or
                audit.get('evaluation_plan') != plan.safe_reference()):
            raise GuardError('guard_execution_audit_mismatch')
    processing = audit.get('processing_execution', {})
    if any(processing.get(k) != actual.get(k) for k in ('model_type','normalization','class_balance')):
        raise GuardError('guard_execution_audit_mismatch')
    if config.get('execution_processing_digest') and processing.get('execution_digest') != config['execution_processing_digest']:
        raise GuardError('guard_execution_audit_mismatch')
    from ..model_config import semantic_digest
    if semantic_digest({k:v for k,v in processing.items() if k != 'digest'}) != processing.get('digest'):
        raise GuardError('guard_execution_audit_mismatch')
    for fold, summary in zip(folds, processing.get('folds', []), strict=True):
        check_fit_audit(fold, actual, finite_search=config.get('execution_search_plan') is not None)
        if summary.get('fold_index') != fold.get('fold_index'):
            raise GuardError('guard_execution_audit_mismatch')
        stages = fold.get('processing_execution', {}).get('stages', [])
        projected = []
        for stage in stages:
            indices = sorted(fold['splits']['train'])
            if stage['stage'] == 'final_train_valid':
                indices = sorted(set(indices + fold['splits']['valid']))
            if (stage.get('fit_count') != len(indices) or
                    stage.get('fit_indices_digest') != semantic_digest(indices) or
                    stage.get('normalizer_digest') != semantic_digest(stage.get('normalizer')) or
                    stage.get('weighting_digest') != semantic_digest(stage.get('weighting')) or
                    stage.get('normalization') != actual['normalization'] or
                    stage.get('class_balance') != actual['class_balance']):
                raise GuardError('guard_execution_audit_mismatch')
            projected.append({k:stage[k] for k in ('stage','fit_scope','fit_indices_digest','normalizer_digest','weighting_digest')})
        if projected != summary.get('stages'):
            raise GuardError('guard_execution_audit_mismatch')
    search = config.get('execution_search_plan')
    if search is not None:
        summary, trials = docs['search_summary.json'], docs['search_trials.json']
        if (docs['search_plan.json'] != search or summary.get('plan_digest') != search['plan_digest'] or
                summary.get('integrity') != 'complete' or summary.get('completed_trials') != search['effective_trials'] or
                summary.get('candidate_fit_count') != len(trials) or
                len(trials) != search['effective_trials'] or any(t.get('state') != 'succeeded' for t in trials)):
            raise GuardError('guard_execution_audit_mismatch')
        for trial, candidate in zip(trials, search['candidates']):
            if trial.get('params') != candidate['params'] or trial.get('params_digest') != candidate['params_digest']:
                raise GuardError('guard_execution_audit_mismatch')
        for selected, fold in zip(summary['selected'], folds, strict=True):
            index = selected.get('trial_index')
            if type(index) is not int or not 0 <= index < len(trials):
                raise GuardError('guard_execution_audit_mismatch')
            trial = trials[index]
            check_fit_audit(fold, actual, finite_search=True, selected_params=trial['params'])
            score = trial.get('selection_score')
            if (type(score) not in (int, float) or not math.isfinite(score)
                    or selected.get('params') != trial['params'] or selected.get('selection_score') != score
                    or fold.get('selection_score') != score):
                raise GuardError('guard_execution_audit_mismatch')
    return validation


def check_outputs(*, record, repository, run_dir, stage: Stage, active=lambda: None, usage=None, manifest_bytes=None):
    try:
        require_current_policy(record.guard_policy)
        from .contracts import Principal
        expected = report(record, 'pre_fit').bindings
        prior = repository.guard_reports_for_stage(record.run_id, 'pre_fit', principal=Principal(record.owner_id, record.tenant_id))
        if not any(p.status == 'passed' and p.bindings == expected for p in prior):
            raise GuardError('guard_report_missing')
        if record.config.get('execution_budget_task_id') and usage is None:
            raise GuardError('guard_usage_unresolved')
        snapshot = inspect_artifacts(run_dir, record.run_id, required=required_artifacts(record.config), active=active, manifest_bytes=manifest_bytes)
        check_output_semantics(snapshot, record, repository)
        if usage is not None:
            summary = snapshot.documents.get('search_summary.json', {})
            if (usage['model_fits'] != summary.get('total_fit_count') or
                    usage['training_epochs'] != (summary.get('actual_epochs') or 0)):
                raise GuardError('guard_usage_mismatch')
        value = report(record, stage, manifest_digest=snapshot.manifest_digest,
                       usage_digest=digest(usage) if usage is not None else None)
    except Exception as exc:
        reason = exc.code if isinstance(exc, GuardError) else 'guard_check_unavailable'
        value = report(record, stage, reason=reason)
        repository.save_guard_report(value)
        raise GuardError(reason, stage=stage, report_id=value.report_id) from exc
    if stage != 'publication':
        repository.save_guard_report(value)
    return value


@dataclass(frozen=True)
class CandidateAssessment:
    status: str
    metrics: dict
    reason: str | None
    report: GuardReport | None = None
    documents: dict | None = None


def assess_candidate(*, record, repository, run_dir, selection_metric, stage: Stage = 'observation'):
    from .contracts import Principal
    from .guard_store import validate_binding
    if record.state != 'succeeded':
        status = 'pending' if record.state in ('queued','running') else 'unavailable'
        value = None
        if record.guard_policy is not None:
            try:
                reference = (record.error_details or {}).get('report_id')
                if reference:
                    value = repository.get_guard_report(reference, principal=Principal(record.owner_id, record.tenant_id))
                    validate_binding(value, record)
                else:
                    value = report(record, stage, status=status)
                    repository.save_guard_report(value)
            except Exception:
                return CandidateAssessment('unavailable', {}, 'guard_check_unavailable')
        return CandidateAssessment(status, {}, None, value)
    try:
        if record.guard_policy is not None:
            try:
                require_current_policy(record.guard_policy)
            except GuardError as exc:
                # Keep historical publication and lock facts immutable. This release
                # cannot certify results under a different source policy.
                value = report(record, stage, reason=exc.code, status='unavailable')
                repository.save_guard_report(value)
                return CandidateAssessment('unavailable', {}, exc.code, value)
        snapshot = inspect_artifacts(run_dir, record.run_id,
            required=required_artifacts(record.config) if record.guard_policy else
                {k for k,v in ARTIFACT_CATALOG.items() if v.get('required')})
        metrics = validation_metrics(snapshot.documents.get('metrics.json'), selection_metric)
        if record.guard_policy is not None:
            if not record.publication_report_id:
                raise GuardError('guard_report_missing')
            published = repository.get_guard_report(record.publication_report_id,
                principal=Principal(record.owner_id, record.tenant_id))
            validate_binding(published, record, publication=True)
            if published.bindings.manifest_digest != snapshot.manifest_digest:
                raise GuardError('guard_report_binding_mismatch')
            check_output_semantics(snapshot, record, repository)
            if record.config.get('execution_budget_task_id'):
                from ..agent.training_budget import load_training_binding, TrainingBudget
                binding = load_training_binding(repository.database_path.parent/'agent.sqlite3', record,
                    runs_database=repository.database_path)
                if binding is None:
                    raise GuardError('guard_usage_unresolved')
                ledger = TrainingBudget(binding.agent_database, task_id=binding.task_id,
                    policy_digest=binding.policy_digest, reservation_id=binding.reservation_id,
                    run_id=record.run_id, claim_token=record.claim_token, claim_check=lambda: None)
                if digest(ledger.guard_snapshot(settled=True)) != published.bindings.usage_digest:
                    raise GuardError('guard_usage_mismatch')
            value = report(record, stage, manifest_digest=snapshot.manifest_digest,
                           usage_digest=published.bindings.usage_digest)
            repository.save_guard_report(value)
        else:
            value = None
        return CandidateAssessment('ready', metrics, None, value, snapshot.documents)
    except Exception as exc:
        reason = exc.code if isinstance(exc, GuardError) else 'guard_check_unavailable'
        value = None
        if record.guard_policy is not None:
            value = report(record, stage, reason=reason)
            try:
                repository.save_guard_report(value)
            except Exception:
                # A failed audit write can only remove eligibility.
                return CandidateAssessment('unavailable', {}, 'guard_check_unavailable')
        return CandidateAssessment('unavailable', {}, reason, value)

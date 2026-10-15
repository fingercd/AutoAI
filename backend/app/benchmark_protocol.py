"""Approved benchmark rules and backend-owned test partitions."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import numpy as np

from .evaluation_plan import DatasetView, EvaluationPlan, digest, evaluation_config, split_indices, validate_plan

# Keep v3 as the historical default; loading a v4 approval never upgrades v3.
PROTOCOL_ID = 'autoai-classification-benchmark-v3'
FULL_PROTOCOL_ID = 'autoai-classification-benchmark-v4'
OBJECTIVE_PROTOCOL_ID = 'autoai-classification-benchmark-v5'
SUPPORTED_PROTOCOL_IDS = frozenset({PROTOCOL_ID, FULL_PROTOCOL_ID, OBJECTIVE_PROTOCOL_ID})


def runtime_policies(protocol):
    from .benchmark_llm import LEGACY_PROMPT_VERSION, OBJECTIVE_PROMPT_VERSION
    from .probability_evaluation import EVALUATION_VERSION
    if protocol['protocol_id'] == OBJECTIVE_PROTOCOL_ID:
        return dict(decision_prompt_version=OBJECTIVE_PROMPT_VERSION,
                    evaluation_version=EVALUATION_VERSION)
    return dict(decision_prompt_version=LEGACY_PROMPT_VERSION, evaluation_version=None)

SEARCH_POLICY = 'benchmark-three-config-v1'
TABLES = {
    'pls_da': ({'pls_components': n} for n in (1, 3, 5)),
    'pca_lda': ({'pca_components': n} for n in (3, 10, 30)),
    'logistic_regression': ({'logistic_c': n} for n in (0.1, 1.0, 10.0)),
    'svm': ({'svm_kernel': 'linear', 'svm_gamma': 'scale', 'svm_c': n} for n in (0.1, 1.0, 10.0)),
    'random_forest': ({'random_forest_max_depth': n, 'random_forest_min_samples_leaf': 2, 'random_forest_max_features': 'sqrt'} for n in (3, 5, 10)),
    'xgboost': ({'xgboost_n_estimators': 100, 'xgboost_max_depth': n, 'xgboost_min_child_weight': 3} for n in (2, 3, 5)),
}
TABLES = {key: tuple(value) for key, value in TABLES.items()}
DEEP_TABLE = tuple({'learning_rate': n, 'weight_decay': 0.0001} for n in (0.0003, 0.001, 0.003))
UNORDERED_MODELS = frozenset({'pls_da', 'pca_lda', 'logistic_regression', 'svm', 'random_forest', 'xgboost', 'pca_mlp'})


def load_approved_protocol(path: Path) -> dict:
    body = json.loads(Path(path).read_text(encoding='utf-8'))
    expected = body.pop('approval_digest')
    if digest(body) != expected or body['protocol_id'] not in SUPPORTED_PROTOCOL_IDS or not body['governance']['user_approved']:
        raise ValueError('benchmark protocol is not the approved version')
    if body['hpo']['candidate_budgets'] != [1, 3] or body['split_seed'] != 42:
        raise ValueError('approved benchmark rules differ')
    return {**body, 'approval_digest': expected}


def validate_context_reference(context: dict) -> dict:
    required = {'suite_root', 'bundle_id', 'task_id', 'round_id', 'protocol_digest', 'source_digest'}
    if type(context) is not dict or not required <= context.keys():
        raise ValueError('benchmark execution binding is incomplete')
    configured = Path(os.environ.get('AUTOAI_BENCHMARK_ROOT', Path(__file__).resolve().parents[2] / 'work' / 'benchmark-20261010')).resolve()
    root = Path(context['suite_root']).resolve()
    if root != configured:
        raise ValueError('benchmark root is not the server-authorized directory')
    protocol = load_approved_protocol(root / 'approval' / 'protocol.json')
    if context['protocol_digest'] != protocol['approval_digest']:
        raise ValueError('benchmark approval binding differs')
    for key in ('protocol_digest', 'source_digest'):
        value = context[key]
        if type(value) is not str or len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
            raise ValueError('invalid benchmark digest')
    for key in ('bundle_id', 'task_id', 'round_id'):
        if type(context[key]) is not str or not context[key]:
            raise ValueError('invalid benchmark identity')
    frozen_source = root / 'approval' / 'source-freeze.json'
    if frozen_source.exists() and json.loads(frozen_source.read_text())['source_digest'] != context['source_digest']:
        raise ValueError('benchmark source snapshot differs')
    return protocol


def build_benchmark_search_plan(*, evaluation_version=None, **kwargs) -> dict:
    from .search_policy import build_search_plan
    from .model_catalog import MODELS_BY_ID
    if kwargs['max_trials'] not in (1, 3) or kwargs['seed'] != 42:
        raise ValueError('benchmark requires budget 1 or 3 and seed 42')
    plan = build_search_plan(**kwargs)
    model_id = kwargs['model_id']
    table = TABLES.get(model_id, DEEP_TABLE if MODELS_BY_ID[model_id].execution_family == 'deep_learning' else ())
    baseline = dict(kwargs['baseline'])
    keys = set(table[0])
    baseline_choice = {key: baseline[key] for key in keys}
    cap = min(kwargs['feature_count'], kwargs['train_count'] - (model_id == 'pls_da'))
    valid, excluded = [], []
    for row in table:
        component = row.get('pls_components', row.get('pca_components'))
        if component is not None and component > cap:
            excluded.append({'params': dict(row), 'reason': 'train_dimension_cap'})
        else:
            valid.append(dict(row))
    ordered = [baseline_choice] + [row for row in valid if row != baseline_choice]
    candidates = [{'index': i, 'params': {**baseline, **row}, 'params_digest': digest({**baseline, **row})} for i, row in enumerate(ordered)]
    plan.update(policy_version=SEARCH_POLICY, domain=list(table), excluded=excluded, candidates=candidates,
                effective_trials=min(kwargs['max_trials'], len(candidates)), selection_metric='valid_balanced_accuracy')
    if evaluation_version is not None:
        from .probability_evaluation import EVALUATION_VERSION
        if evaluation_version != EVALUATION_VERSION:
            raise ValueError('unsupported benchmark evaluation version')
        plan['evaluation_version'] = evaluation_version
    plan['effective_search'] = plan['effective_trials'] > 1
    plan['plan_digest'] = digest({key: value for key, value in plan.items() if key != 'plan_digest'})
    return plan


def validate_benchmark_search_plan(plan: dict, **bindings) -> None:
    from .search_policy import _digest_source
    if plan['source_digest'] != _digest_source():
        raise ValueError('benchmark training source changed')
    config = bindings['config']
    if any(config.get(key) != value for key, value in plan['baseline'].items()):
        raise ValueError('benchmark baseline differs from Run')
    context = config.get('benchmark_context')
    evaluation_version = (runtime_policies(validate_context_reference(context))['evaluation_version']
                          if context is not None else None)
    rebuilt = build_benchmark_search_plan(evaluation_version=evaluation_version,
        model_id=bindings['model_id'], baseline=plan['baseline'],
        mode=plan['mode'], max_trials=plan['max_trials'], train_count=bindings['train_count'],
        feature_count=bindings['feature_count'], class_count=bindings['class_count'],
        dataset_sha256=bindings['dataset_sha256'], evaluation_plan_digest=bindings['evaluation_plan_digest'],
        normalization=config['normalization'], class_balance=config['class_balance'], seed=config['seed'],
        architecture_version=bindings['architecture_version'], processing_policy_version=bindings['processing_policy_version'])
    if plan != rebuilt:
        raise ValueError('benchmark search plan binding differs')


def make_plan(view: DatasetView, indices: dict[str, list[int]], config: dict) -> EvaluationPlan:
    from .parsers import natural_sort_key
    groups = {name: sorted({view.sample_ids[index] for index in rows}, key=natural_sort_key) for name, rows in indices.items()}
    body = {'schema_version': 'evaluation-plan-v1', 'split_algorithm_version': 'benchmark-held-test-v1',
            'dataset_sha256': view.dataset_sha256, 'axis_digest': digest(view.axis),
            'evaluation_config_digest': digest(evaluation_config(config)), 'seed': 42,
            'indices': indices, 'groups': groups, 'partition_digest': digest({'indices': indices, 'groups': groups})}
    plan = EvaluationPlan(**body, plan_digest=digest(body))
    validate_plan(plan, view, config, 42)
    return plan


def _training_validation(view: DatasetView, available: list[int]) -> dict[str, list[int]]:
    labels = sorted(set(view.labels))
    lookup = {label: index for index, label in enumerate(labels)}
    local = split_indices(np.asarray([lookup[view.labels[i]] for i in available]),
        np.asarray([view.sample_ids[i] for i in available]),
        SimpleNamespace(split_train=8, split_valid=2, split_test=0, seed=42), labels)
    return {name: [available[i] for i in local[name]] for name in ('train', 'valid')}


def build_test_rounds(view: DatasetView, *, official_test_indices: list[int] | None = None) -> list[EvaluationPlan]:
    config = dict(split_mode='stratified_holdout', split_train=8, split_valid=1, split_test=1)
    if official_test_indices is not None:
        tests = set(official_test_indices)
        available = [i for i in range(len(view.x)) if i not in tests]
        return [make_plan(view, {**_training_validation(view, available), 'test': list(official_test_indices)}, config)]
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    grouped = len(set(view.sample_ids)) != len(view.sample_ids)
    failure = None
    for count in (5, 3):
        splitter = (StratifiedGroupKFold if grouped else StratifiedKFold)(n_splits=count, shuffle=True, random_state=42)
        try:
            partitions = splitter.split(np.zeros(len(view.x)), view.labels, view.sample_ids if grouped else None)
            plans = [make_plan(view, {**_training_validation(view, train.tolist()), 'test': test.tolist()}, config)
                     for train, test in partitions]
            tests = [i for plan in plans for i in plan.indices['test']]
            if sorted(tests) != list(range(len(view.x))):
                raise ValueError('benchmark test coverage differs')
            return plans
        except ValueError as exc:
            failure = exc
    raise ValueError('no class-complete approved test partition is feasible') from failure


def prepare_task_rounds(task: dict, root: Path) -> list[dict]:
    import pandas as pd
    from .parsers import load_modeling_csv
    root = Path(root).resolve()
    directory = root / 'prepared' / digest([task['suite'], task['task_id']])[:20]
    directory.mkdir(parents=True, exist_ok=True)
    main = Path(task['main_path'])
    for name in ('main', 'test'):
        path = task.get(name + '_path')
        expected = task.get(name + '_sha256')
        if path and expected:
            with Path(path).open('rb') as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != expected:
                    raise ValueError('audited source dataset changed')
    frame = pd.read_csv(main, dtype={'Label': str, 'Sample_ID': str, 'Name': str})
    official = None
    if task.get('test_path'):
        test = pd.read_csv(task['test_path'], dtype={'Label': str, 'Sample_ID': str, 'Name': str})
        if list(frame.columns) != list(test.columns):
            raise ValueError('official test coordinate columns differ')
        if set(frame['Sample_ID'].astype(str)) & set(test['Sample_ID'].astype(str)):
            raise ValueError('official test shares independent groups')
        official = list(range(len(frame), len(frame) + len(test)))
        frame = pd.concat([frame, test], ignore_index=True)
    frame['Index'] = np.arange(1, len(frame) + 1)
    path = directory / 'combined.csv'
    encoded = frame.to_csv(index=False).encode('utf-8')
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError('frozen combined dataset changed')
    if not path.exists():
        path.write_bytes(encoded)
    sha = hashlib.sha256(encoded).hexdigest()
    dataset = load_modeling_csv(path)
    view = DatasetView.loaded(dataset, sha)
    plans = build_test_rounds(view, official_test_indices=official)
    results = []
    for index, plan in enumerate(plans):
        results.append({'task_id': task['task_id'], 'suite': task['suite'], 'family_id': task['family_id'],
            'role': task['role'], 'axis_kind': task['axis_kind'], 'round_id': f'{index + 1:02d}',
            'data_path': str(path), 'dataset_sha256': sha, 'plan': plan.model_dump(mode='json'),
            'labels': sorted(set(view.labels)), 'round_count': len(plans)})
    return results

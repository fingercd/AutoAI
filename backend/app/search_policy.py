"""Frozen, finite hyperparameter plans for agent recipe revision v4.

The plan is made from Train-only dimensions at Session preparation.  A worker
validates it again before fitting; neither an Agent action nor the worker may
sample a new candidate after the Session has been frozen.
"""
from __future__ import annotations

from itertools import product
from functools import lru_cache
from pathlib import Path
from typing import Any
import hashlib

from .model_catalog import MODELS_BY_ID
from .model_config import semantic_digest


POLICY_VERSION = 'finite-hpo-v1'
SCHEMA_VERSION = 'search-plan-v1'
SEARCH_MODES = ('fixed', 'bounded')
DEEP_DOMAIN = tuple(dict(learning_rate=rate, weight_decay=decay)
                    for rate, decay in product((0.0003, 0.001, 0.003), (0.0, 0.0001)))
DOMAINS: dict[str, tuple[dict[str, Any], ...]] = {
    'pls_da': tuple({'pls_components': n} for n in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15)),
    'pca_lda': tuple({'pca_components': n} for n in (2, 3, 5, 8, 10, 15, 20, 30, 40, 50)),
    'logistic_regression': tuple({'logistic_c': n} for n in (0.1, 1.0, 10.0)),
    'svm': tuple({'svm_kernel': 'linear', 'svm_gamma': 'scale', 'svm_c': n}
                 for n in (0.01, 0.1, 1.0, 10.0, 100.0)),
    'random_forest': tuple(dict(random_forest_max_depth=depth,
                                random_forest_min_samples_leaf=leaf,
                                random_forest_max_features=features)
                           for depth, leaf, features in product((3, 5, 10), (2, 5), ('sqrt', 'log2', 0.1))),
    'xgboost': tuple(dict(xgboost_n_estimators=trees, xgboost_max_depth=depth,
                          xgboost_min_child_weight=child)
                     for trees, depth, child in product((100, 300), (2, 3, 5), (3, 5))),
}

BASELINES: dict[str, dict[str, Any]] = {
    'pls_da': {'pls_components': 3},
    'pca_lda': {'pca_components': 3},
    'logistic_regression': {'logistic_c': 1.0},
    'svm': {'svm_kernel': 'linear', 'svm_gamma': 'scale', 'svm_c': 1.0},
    'random_forest': {'random_forest_max_depth': 3, 'random_forest_min_samples_leaf': 2,
                      'random_forest_max_features': 'sqrt'},
    'xgboost': {'xgboost_n_estimators': 100, 'xgboost_max_depth': 2,
                'xgboost_min_child_weight': 3, 'xgboost_learning_rate': 0.1,
                'xgboost_subsample': 0.8, 'xgboost_colsample_bytree': 0.3,
                'xgboost_reg_lambda': 10.0},
}
for _name, _model in MODELS_BY_ID.items():
    if _model.implemented and _model.execution_family == 'deep_learning':
        BASELINES[_name] = {'learning_rate': 0.001, 'weight_decay': 0.0001}
        DOMAINS[_name] = DEEP_DOMAIN


def validate_search_options(mode: str, max_trials: int | None) -> int:
    if mode not in SEARCH_MODES:
        raise ValueError('search_mode must be fixed or bounded')
    if max_trials is None:
        max_trials = 1 if mode == 'fixed' else 6
    if type(max_trials) is not int or not 1 <= max_trials <= 18:
        raise ValueError('max_trials must be an integer from 1 to 18')
    if mode == 'fixed' and max_trials != 1:
        raise ValueError('fixed search requires max_trials=1')
    return max_trials


def baseline_fields(model_id: str) -> dict[str, Any]:
    return dict(BASELINES[model_id])


def _legal_domain(model_id: str, train_count: int, feature_count: int,
                  class_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if model_id not in DOMAINS:
        raise ValueError('model has no finite search domain')
    if min(train_count, feature_count, class_count) < 1:
        raise ValueError('invalid Train dimensions')
    # Centered PLS has at most n-1 independent observations.  PCA's SVD
    # permits n components, while LDA still requires n > class count.
    cap = min(feature_count, train_count - (1 if model_id == 'pls_da' else 0))
    if model_id == 'pca_lda' and train_count <= class_count:
        cap = 0
    legal, excluded = [], []
    for row in DOMAINS[model_id]:
        component = row.get('pls_components', row.get('pca_components'))
        if component is not None and component > cap:
            excluded.append({'params': row, 'reason': 'train_dimension_cap'})
        else:
            legal.append(dict(row))
    if not legal:
        raise ValueError('no legal Train-only search candidates')
    return legal, excluded


@lru_cache(maxsize=1)
def _digest_source() -> str:
    # Bind the policy and the single authoritative training implementation.
    h = hashlib.sha256()
    for name in ('search_policy.py', 'training.py'):
        h.update((Path(__file__).parent / name).read_bytes())
    return h.hexdigest()


def adjust_baselines_for_train(snapshot: dict[str, Any], *, train_count: int,
                               feature_count: int, class_count: int,
                               explicit_configs: dict[str, Any] | None = None) -> None:
    """Resolve dimension-dependent defaults into the one frozen model_configs source."""
    for model_id in ('pls_da', 'pca_lda'):
        if model_id not in snapshot['model_configs']:
            continue
        key = 'pls_components' if model_id == 'pls_da' else 'pca_components'
        if model_id == 'pca_lda' and (feature_count < 2 or train_count <= class_count):
            # The catalog excludes this model before making a recipe.  A user
            # supplied baseline still has to fail instead of being discarded.
            if model_id in (explicit_configs or {}):
                raise ValueError('explicit PCA-LDA baseline is not trainable on this data')
            continue
        if key in (explicit_configs or {}).get(model_id, {}):
            continue
        legal, _ = _legal_domain(model_id, train_count, feature_count, class_count)
        value = snapshot['model_configs'][model_id][key]
        if value == 3 and not any(row[key] == 3 for row in legal):
            snapshot['model_configs'][model_id][key] = max(row[key] for row in legal if row[key] <= 3)


def build_search_plan(*, model_id: str, baseline: dict[str, Any], mode: str,
                      max_trials: int | None, train_count: int, feature_count: int,
                      class_count: int, dataset_sha256: str, evaluation_plan_digest: str,
                      normalization: str, class_balance: str, seed: int,
                      architecture_version: str, processing_policy_version: str,
                      source_digest: str | None = None) -> dict[str, Any]:
    limit = validate_search_options(mode, max_trials)
    legal, excluded = _legal_domain(model_id, train_count, feature_count, class_count)
    baseline = dict(baseline)
    selected_keys = set(legal[0])
    if not selected_keys <= set(baseline):
        raise ValueError('incomplete baseline')
    baseline_candidate = {key: baseline[key] for key in selected_keys}
    if baseline_candidate not in legal:
        raise ValueError('baseline is outside legal finite domain')
    # The canonical domain order is frozen in the plan.  Candidate zero is the
    # same baseline in both modes, independent of the requested budget.
    ordered = [baseline_candidate] + [row for row in legal if row != baseline_candidate]
    candidates = [dict(index=i, params={**baseline, **row}, params_digest=semantic_digest({**baseline, **row}))
                  for i, row in enumerate(ordered)]
    metric = ('oob_balanced_accuracy_then_accuracy' if model_id == 'random_forest' else
              'valid_balanced_accuracy' if MODELS_BY_ID[model_id].execution_family == 'traditional_ml' else
              'best_valid_loss')
    body = dict(schema_version=SCHEMA_VERSION, policy_version=POLICY_VERSION,
                model_id=model_id, architecture_version=architecture_version,
                dataset_sha256=dataset_sha256, evaluation_plan_digest=evaluation_plan_digest,
                processing=dict(normalization=normalization, class_balance=class_balance,
                                policy_version=processing_policy_version),
                mode=mode, max_trials=limit, effective_trials=min(limit, len(candidates)),
                effective_search=mode == 'bounded' and min(limit, len(candidates)) > 1,
                baseline=baseline, domain=list(DOMAINS[model_id]), excluded=excluded,
                candidates=candidates, seed=seed, selection_metric=metric,
                tie_break='earliest_trial', fit_strategy=('train_then_train_valid_refit'
                    if MODELS_BY_ID[model_id].execution_family == 'traditional_ml' else 'train_best_epoch_no_refit'),
                source_digest=source_digest or _digest_source())
    body['plan_digest'] = semantic_digest(body)
    return body


def validate_search_plan(plan: dict[str, Any], *, model_id: str, config: dict[str, Any],
                         train_count: int, feature_count: int, class_count: int,
                         dataset_sha256: str, evaluation_plan_digest: str,
                         architecture_version: str, processing_policy_version: str) -> None:
    if type(plan) is not dict or set(plan) != {
        'schema_version', 'policy_version', 'model_id', 'architecture_version',
        'dataset_sha256', 'evaluation_plan_digest', 'processing', 'mode', 'max_trials',
        'effective_trials', 'effective_search', 'baseline', 'domain', 'excluded',
        'candidates', 'seed', 'selection_metric', 'tie_break', 'fit_strategy',
        'source_digest', 'plan_digest',
    }:
        raise ValueError('invalid search plan schema')
    if plan['source_digest'] != _digest_source():
        raise ValueError('search implementation changed')
    if plan['seed'] != config.get('seed') or type(plan['seed']) is not int:
        raise ValueError('search seed mismatch')
    if any(config.get(key) != value for key, value in plan['baseline'].items()):
        raise ValueError('search baseline differs from Run config')
    rebuilt = build_search_plan(model_id=model_id, baseline=plan['baseline'],
        mode=plan['mode'], max_trials=plan['max_trials'], train_count=train_count,
        feature_count=feature_count, class_count=class_count,
        dataset_sha256=dataset_sha256, evaluation_plan_digest=evaluation_plan_digest,
        normalization=config.get('normalization'), class_balance=config.get('class_balance'),
        seed=plan['seed'], architecture_version=architecture_version,
        processing_policy_version=processing_policy_version,
        source_digest=plan['source_digest'])
    if plan != rebuilt:
        raise ValueError('search plan binding mismatch')

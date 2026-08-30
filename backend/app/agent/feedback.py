"""Safe train/validation diagnosis and finite replanning hints."""

from __future__ import annotations

import math
from typing import Any


DIAGNOSIS_SCHEMA_VERSION = 'agent-diagnosis-v1'

_CATEGORY_ACTIONS = {
    'data': ('request_human',),
    'split': ('request_human',),
    'feature': ('choose_unused_proposal',),
    'model': ('choose_unused_proposal', 'stop'),
    'hpo': ('choose_unused_proposal', 'stop'),
    'training': ('choose_unused_proposal', 'stop'),
    'environment': ('request_human',),
    'task_semantics': ('request_human',),
    'none': ('finalize',),
}


def _failure_category(error_details: dict[str, Any]) -> str:
    code = str(error_details.get('code') or '')
    if code in {'dataset_changed', 'dataset_unavailable'}:
        return 'data'
    if code == 'agent_guard_rejected':
        guard = error_details.get('guard_result')
        checks = guard.get('checks') if isinstance(guard, dict) else None
        failed_codes = {
            str(item.get('code'))
            for item in (checks or [])
            if isinstance(item, dict) and item.get('status') == 'failed'
        }
        if failed_codes & {'data_contract', 'finite_values', 'resource_limit'}:
            return 'data'
        if 'split_feasibility' in failed_codes:
            return 'split'
        if failed_codes & {'hpo_budget', 'model_fit_budget', 'cost_contract'}:
            return 'hpo'
        if failed_codes & {'task_semantics', 'classification_semantics'}:
            return 'task_semantics'
        return 'training'
    if code == 'worker_contract_mismatch':
        return 'environment'
    if code == 'training_failed':
        return 'training'
    if code == 'invalid_training_data_or_config':
        return 'training'
    return 'training'


def _finite_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    }


def build_diagnosis(
    *,
    run_state: str,
    selection_metric: str,
    training_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    error_details: dict[str, Any] | None = None,
    evidence_card: dict[str, Any] | None = None,
    restricted_actions_available: bool = False,
) -> dict[str, Any]:
    train = _finite_metrics(training_metrics)
    valid = _finite_metrics(validation_metrics)
    symptoms: list[str] = []
    generalization_gap: float | None = None

    if run_state == 'failed':
        category = _failure_category(error_details or {})
        symptoms.append('failed_run')
    elif run_state == 'succeeded':
        train_score = train.get(selection_metric)
        valid_score = valid.get(selection_metric)
        if train_score is None or valid_score is None:
            return {
                'schema_version': DIAGNOSIS_SCHEMA_VERSION,
                'status': 'insufficient_evidence',
                'primary_category': 'none',
                'symptom_codes': [],
                'recoverable': False,
                'allowed_action_ids': ['request_human'],
                'training_metrics': train,
                'validation_metrics': valid,
                'generalization_gap': None,
            }
        generalization_gap = round(train_score - valid_score, 6)
        if generalization_gap >= 0.15:
            symptoms.append('generalization_gap_high')
        if valid_score < 0.5:
            symptoms.append('validation_score_low')
        risk_codes = set(
            (evidence_card or {}).get('statistics', {}).get('risk_codes', [])
        )
        if 'class_imbalance' in risk_codes:
            symptoms.append('class_imbalance_present')
        if 'high_dimension' in risk_codes:
            symptoms.append('high_dimension_present')
        if 'generalization_gap_high' in symptoms:
            category = 'training'
        elif 'validation_score_low' in symptoms:
            category = 'model'
        elif 'high_dimension_present' in symptoms:
            category = 'feature'
        else:
            category = 'none'
    else:
        return {
            'schema_version': DIAGNOSIS_SCHEMA_VERSION,
            'status': 'pending',
            'primary_category': 'none',
            'symptom_codes': [],
            'recoverable': False,
            'allowed_action_ids': [],
            'training_metrics': {},
            'validation_metrics': {},
            'generalization_gap': None,
        }

    allowed = list(_CATEGORY_ACTIONS[category])
    if not restricted_actions_available:
        allowed = [item for item in allowed if item != 'choose_unused_proposal']
        if not allowed:
            allowed = ['request_human']
    return {
        'schema_version': DIAGNOSIS_SCHEMA_VERSION,
        'status': 'ready',
        'primary_category': category,
        'symptom_codes': sorted(set(symptoms)),
        'recoverable': category not in {'none', 'task_semantics'},
        'allowed_action_ids': allowed,
        'training_metrics': train,
        'validation_metrics': valid,
        'generalization_gap': generalization_gap,
    }

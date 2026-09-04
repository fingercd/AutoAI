"""供任意 LLM Orchestrator 注册的有限工具 JSON Schema。"""

from __future__ import annotations

from typing import Any


MODELS = ['logistic_regression', 'svm', 'random_forest']
MODULES = [
    'evidence_card', 'dynamic_preprocessing', 'restricted_strategy_pool', 'bounded_hpo',
    'fail_fast_guard', 'feedback_diagnosis', 'limited_replanning', 'uncertainty_selection',
    'case_memory', 'budget_control', 'constrained_code_evolution',
]


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {'type': 'object', 'properties': properties, 'required': required,
            'additionalProperties': False}


TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    'inspect_ml_capabilities': _schema({}, []),
    'start_ml_session': _schema({
        'dataset_id': {'type': 'string', 'minLength': 1},
        'selection_metric': {'type': 'string', 'enum': ['macro_f1', 'balanced_accuracy']},
        'allowed_models': {'type': 'array', 'items': {'type': 'string', 'enum': MODELS}, 'minItems': 1},
        'max_runs': {'type': 'integer', 'minimum': 1, 'maximum': 10},
        'seed': {'type': 'integer', 'minimum': 0},
        'modules': {'type': 'array', 'items': {'type': 'string', 'enum': MODULES}},
        'context_policy': _schema({
            'source_role': {'type': 'string', 'enum': ['development', 'benchmark', 'domain']},
            'case_write': {'type': 'boolean'},
        }, []),
        'client_request_id': {'type': 'string', 'minLength': 1, 'maxLength': 128},
    }, ['dataset_id', 'selection_metric', 'allowed_models', 'max_runs']),
    'inspect_ml_session': _schema({'session_id': {'type': 'string', 'minLength': 1}}, ['session_id']),
    'submit_ml_experiment': _schema({
        'session_id': {'type': 'string', 'minLength': 1},
        'model_type': {'type': 'string', 'enum': MODELS},
        'normalization': {'type': 'string', 'enum': ['zscore', 'minmax', 'area', 'none']},
        'class_balance': {'type': 'string', 'enum': ['none', 'class_weight']},
        'parent_run_id': {'type': 'string', 'minLength': 1},
        'rationale': {'type': 'string', 'maxLength': 2000},
        'client_request_id': {'type': 'string', 'minLength': 1, 'maxLength': 128},
    }, ['session_id', 'model_type']),
    'observe_ml_experiment': _schema({
        'session_id': {'type': 'string', 'minLength': 1}, 'run_id': {'type': 'string', 'minLength': 1},
    }, ['session_id', 'run_id']),
    'finalize_ml_session': _schema({
        'session_id': {'type': 'string', 'minLength': 1},
        'selected_run_id': {'type': 'string', 'minLength': 1},
    }, ['session_id', 'selected_run_id']),
}

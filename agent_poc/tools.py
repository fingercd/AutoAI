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


def validate_tool_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate the finite JSON Schema subset used by these six tools.

    No dynamic getattr target or arbitrary JSON schema implementation is exposed.
    Error text intentionally omits untrusted tool names and arguments.
    """
    from agent_poc.clients.autoai_client import AgentContractError, validate_identifier

    def check(schema: dict[str, Any], value: Any) -> None:
        kind = schema['type']
        if kind == 'object':
            if type(value) is not dict or set(value) - set(schema['properties']):
                raise ValueError
            if set(schema.get('required', [])) - set(value):
                raise ValueError
            for key, item in value.items():
                check(schema['properties'][key], item)
        elif kind == 'array':
            if type(value) is not list or len(value) < schema.get('minItems', 0):
                raise ValueError
            if len(value) != len(set(value)):
                raise ValueError
            for item in value:
                check(schema['items'], item)
        elif kind == 'string':
            if type(value) is not str or not schema.get('minLength', 0) <= len(value) <= schema.get('maxLength', 10000):
                raise ValueError
        elif kind == 'integer':
            if type(value) is not int or not schema.get('minimum', 0) <= value <= schema.get('maximum', 2**63 - 1):
                raise ValueError
        elif kind == 'boolean':
            if type(value) is not bool:
                raise ValueError
        else:
            raise ValueError
        if 'enum' in schema and value not in schema['enum']:
            raise ValueError

    try:
        if name not in TOOL_SCHEMAS:
            raise ValueError
        check(TOOL_SCHEMAS[name], arguments)
        for key in ('session_id', 'run_id', 'selected_run_id', 'parent_run_id',
                    'dataset_id', 'client_request_id'):
            if key in arguments:
                validate_identifier(arguments[key])
    except Exception:
        raise AgentContractError('工具名称或参数不符合有限契约') from None
    return dict(arguments)


class ToolDispatcher:
    """One dispatcher for deterministic nodes and validated LLM proposals."""
    def __init__(self, client) -> None:
        self.client = client
        self._calls = {
            'inspect_ml_capabilities': client.inspect_ml_capabilities,
            'start_ml_session': client.start_ml_session,
            'inspect_ml_session': client.inspect_ml_session,
            'submit_ml_experiment': client.submit_ml_experiment,
            'observe_ml_experiment': client.observe_ml_experiment,
            'finalize_ml_session': client.finalize_ml_session,
        }

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        validated = validate_tool_arguments(name, arguments)
        return self._calls[name](**validated)

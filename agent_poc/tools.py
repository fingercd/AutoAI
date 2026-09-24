"""供任意 LLM Orchestrator 注册的有限工具 JSON Schema。"""

from __future__ import annotations

from typing import Any
import math


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


def validate_tool_arguments(name: str, arguments: dict[str, Any], schemas: dict | None = None) -> dict[str, Any]:
    """Validate the finite JSON Schema subset used by these six tools.

    No dynamic getattr target or arbitrary JSON schema implementation is exposed.
    Error text intentionally omits untrusted tool names and arguments.
    """
    from agent_poc.clients.autoai_client import AgentContractError, validate_identifier

    schemas = TOOL_SCHEMAS if schemas is None else schemas

    def check(schema: dict[str, Any], value: Any) -> None:
        kind = schema['type']
        if isinstance(kind, list):
            if value is None and 'null' in kind:
                return
            for candidate in kind:
                if candidate == 'null':
                    continue
                try:
                    check({**schema, 'type': candidate}, value)
                    return
                except ValueError:
                    pass
            raise ValueError
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
        elif kind == 'number':
            if type(value) not in (int,float) or not math.isfinite(value):
                raise ValueError
            if 'minimum' in schema and value < schema['minimum']:
                raise ValueError
            if 'maximum' in schema and value > schema['maximum']:
                raise ValueError
            if 'exclusiveMaximum' in schema and value >= schema['exclusiveMaximum']:
                raise ValueError
            if 'exclusiveMinimum' in schema and value <= schema['exclusiveMinimum']:
                raise ValueError
        elif kind == 'boolean':
            if type(value) is not bool:
                raise ValueError
        else:
            raise ValueError
        if 'enum' in schema and value not in schema['enum']:
            raise ValueError

    try:
        if name not in schemas:
            raise ValueError
        check(schemas[name], arguments)
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
        validated = validate_tool_arguments(name, arguments, self.client.tool_schemas if getattr(self.client, 'api_version', 'v1') == 'v2' else None)
        return self._calls[name](**validated)


def build_tool_schemas(capabilities: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return independent schemas from a validated v2 server capability snapshot."""
    from copy import deepcopy
    from agent_poc.clients.execution_contracts import HealthResponse
    health = HealthResponse.model_validate(capabilities)
    models = [m for m in health.models if m.available]
    ids = [m.id for m in models]
    schemas = deepcopy(TOOL_SCHEMAS)
    start = schemas['start_ml_session']['properties']
    submit = schemas['submit_ml_experiment']['properties']
    start['allowed_models']['items']['enum'] = ids
    start['max_runs'] = {'type':'integer','minimum':1,'maximum':1}
    def parameter_schema(p):
        value={'type':['integer','number','string'] if p.value_type=='scalar' else p.value_type}
        for source,target in [('minimum','minimum'),('maximum','maximum'),('exclusive_minimum','exclusiveMinimum'),('exclusive_maximum','exclusiveMaximum'),('choices','enum')]:
            item=getattr(p,source)
            if item is not None:
                value[target]=item
        if p.nullable:
            value['type']=(value['type'] if isinstance(value['type'],list) else [value['type']])+['null']
        return value
    per_model={m.id:{p.name:parameter_schema(p) for p in m.parameters
                     if p.role in ('operator_fixed','search_baseline')} for m in models}
    start['model_configs']=_schema({name:_schema(params,[]) for name,params in per_model.items()},[])
    submit['model_type']['enum']=ids
    submit['normalization']['enum']=['zscore']
    submit['class_balance']['enum']=['none']
    submit['model_params']=_schema({key:value for params in per_model.values() for key,value in params.items()},[])
    return schemas

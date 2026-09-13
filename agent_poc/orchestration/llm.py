"""Explicit JSON action / native tool adapter; no SDK, tracing or hidden retries.

The caller checkpoints each attempted call before invoking ``propose`` and owns
all retry/repair limits. Malformed output is never retained on an exception.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Literal

from agent_poc.clients.autoai_client import Transport, validate_base_url, validate_identifier
from agent_poc.clients.http_transport import ResponseTooLarge, bounded_request
from agent_poc.tools import TOOL_SCHEMAS, validate_tool_arguments
from .projection import CONTEXT_VERSION, validate_context
from .state import validate_safe_text

PROMPT_VERSION = 'agent-decision-step1-v2'
LLM_CONFIG_VERSION = 'agent-llm-http-v1'
_REPAIR_CODES = frozenset({
    'llm_context_invalid', 'llm_output_invalid', 'llm_output_too_large',
    'llm_timeout', 'llm_unavailable', 'llm_http_error',
})


@dataclass(frozen=True, repr=False)
class LLMConfig:
    base_url: str
    model: str
    protocol: Literal['json_action', 'native_tools'] = 'json_action'
    timeout: float = 30.0
    max_tokens: int = 1024
    max_response_bytes: int = 32768
    temperature: float = 0.0
    top_p: float = 1.0
    prompt_version: str = PROMPT_VERSION

    def __post_init__(self):
        object.__setattr__(self, 'base_url', validate_base_url(self.base_url))
        if (not isinstance(self.model, str) or
                not re.fullmatch(r'[A-Za-z0-9_./:\\-]{1,512}', self.model) or
                '://' in self.model or
                self.model.lower().startswith('file:')):
            raise ValueError('LLM served model ID 无效')
        if self.prompt_version not in (PROMPT_VERSION,'agent-decision-step2-v1'):
            raise ValueError('Unknown Prompt version')
        if self.protocol not in ('json_action', 'native_tools'):
            raise ValueError('必须明确选择受支持的 LLM 协议')
        for value, lower, upper in ((self.timeout, 0.1, 600), (self.temperature, 0, 2), (self.top_p, 0.01, 1)):
            if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError('LLM 生成配置超出允许范围')
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 8192:
            raise ValueError('LLM 输出 token 上限无效')
        if type(self.max_response_bytes) is not int or not 1024 <= self.max_response_bytes <= 1048576:
            raise ValueError('LLM 响应长度上限无效')

    def __repr__(self):
        return f'LLMConfig(base_url=<configured>, protocol={self.protocol!r})'

    def public_config(self) -> dict[str, Any]:
        # A provider's served ID may be an absolute model path. Its raw value
        # belongs only in process memory and the provider's HTTP model field.
        return {'version': LLM_CONFIG_VERSION,
                'model_id_sha256': hashlib.sha256(self.model.encode('utf-8')).hexdigest(),
                'protocol': self.protocol,
                'timeout': self.timeout, 'max_tokens': self.max_tokens,
                'max_response_bytes': self.max_response_bytes,
                'temperature': self.temperature, 'top_p': self.top_p,
                'prompt_version': self.prompt_version, 'context_version': ('agent-context-step2-v1' if self.prompt_version == 'agent-decision-step2-v1' else CONTEXT_VERSION)}

    def fingerprint(self) -> str:
        # Endpoint binding is checked without putting a URL into durable State.
        public = self.public_config()
        public.pop('model_id_sha256')
        return hashlib.sha256(json.dumps({'endpoint': self.base_url, 'model': self.model, **public},
                                         sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    status: Literal['known', 'partial', 'unknown'] = 'unknown'


@dataclass(frozen=True)
class Proposal:
    tool_name: str
    arguments: dict[str, Any]
    rationale: str
    tool_call_id: str | None = None
    response_id: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)


class LLMError(RuntimeError):
    def __init__(self, code: str, *, usage: TokenUsage | None = None):
        super().__init__(code)
        self.code = code
        self.usage = usage or TokenUsage()


def _strict_json(text: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON field')
            result[key] = value
        return result
    def nonfinite(_value):
        raise ValueError('nonfinite JSON number')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)


def _usage(payload: dict[str, Any]) -> TokenUsage:
    value = payload.get('usage')
    if value is None:
        return TokenUsage()
    if type(value) is not dict:
        raise ValueError('invalid usage')
    counts = [value.get(key) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')]
    if any(number is not None and (type(number) is not int or number < 0) for number in counts):
        raise ValueError('invalid usage')
    if all(number is not None for number in counts) and counts[0] + counts[1] != counts[2]:
        raise ValueError('inconsistent usage')
    status = 'known' if all(number is not None for number in counts) else (
        'partial' if any(number is not None for number in counts) else 'unknown')
    return TokenUsage(*counts, status=status)


def _rationale(value: Any) -> str:
    if type(value) is not str or not 1 <= len(value.strip()) <= 1000:
        raise ValueError('invalid rationale')
    if (any(ord(c) < 32 for c in value) or '/' in value or '\\' in value or
            re.search(r'(?i)\b(bearer|password|api[_ -]?key|authorization|traceback|test)\b|prediction|artifact', value)):
        raise ValueError('unsafe rationale')
    return validate_safe_text(value.strip())


class _BoundedHttpxTransport:
    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes

    def request(self, method, url, *, headers, json, timeout):
        try:
            return bounded_request(method, url, headers=headers, json=json, timeout=timeout,
                                   max_response_bytes=self.max_bytes)
        except ResponseTooLarge:
            raise LLMError('llm_output_too_large') from None


class LLMAdapter:
    def __init__(self, config: LLMConfig, *, token: str | None = None,
                 transport: Transport | None = None):
        self.config, self._token = config, token
        self._transport = transport or _BoundedHttpxTransport(config.max_response_bytes)

    def __repr__(self):
        return f'LLMAdapter(protocol={self.config.protocol!r}, token=<redacted>)'

    def propose(self, phase: str, context: dict[str, Any], *,
                timeout_seconds: float | None = None, repair_code: str | None = None) -> Proposal:
        if repair_code is not None and (type(repair_code) is not str or repair_code not in _REPAIR_CODES):
            raise LLMError('llm_context_invalid') from None
        if timeout_seconds is not None and (
                type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)):
            raise LLMError('llm_context_invalid') from None
        request_timeout = min(self.config.timeout, timeout_seconds) if timeout_seconds is not None else self.config.timeout
        if request_timeout <= 0:
            raise LLMError('llm_timeout') from None
        try:
            projected = validate_context(phase, context)
        except Exception:
            raise LLMError('llm_context_invalid') from None
        if projected['context_version'] != self.config.public_config()['context_version']:
            raise LLMError('llm_context_invalid')
        step2 = projected['context_version'] == 'agent-context-step2-v1'
        tool = projected['allowed_actions'][0]
        schema = json.loads(json.dumps(TOOL_SCHEMAS[tool]))
        if tool == 'submit_ml_experiment':
            schema['properties'].pop('parent_run_id', None)
            schema['properties']['model_type']['enum'] = projected['capabilities']['models']
        for key, value in projected['bindings'].items():
            schema['properties'][key]['enum'] = [value]
            if key not in schema['required']:
                schema['required'].append(key)
        if self.config.protocol == 'json_action':
            schema['properties'].pop('rationale', None)
        system = (
            'You select one permitted action for a single classification experiment. '
            'Use only the supplied candidates and validation evidence. Preserve every binding exactly. '
            'Give a brief decision rationale, without private reasoning or external references. '
        )
        if step2:
            system += 'Model parameters are fixed by the operator and shown in fixed_model_params. Select only the model; the runner binds its fixed parameters. '
        request: dict[str, Any] = {
            'model': self.config.model, 'temperature': self.config.temperature,
            'top_p': self.config.top_p, 'max_tokens': self.config.max_tokens, 'stream': False,
            'messages': [],
        }
        if self.config.protocol == 'json_action':
            system += (
                'Respond with exactly one JSON object with only tool_name, arguments and rationale. '
                'tool_name must be the single allowed action. arguments must follow its schema; '
                'include every supplied binding. rationale is one short nonempty sentence. '
                'Put rationale only at the top level, never inside arguments. '
                'No markdown, list or parallel actions. The permitted tool schema is: '
                + json.dumps(schema, ensure_ascii=False, separators=(',', ':'))
            )
        else:
            system += 'Return exactly one native function call and a short rationale as message content.'
            request['tools'] = [{'type': 'function', 'function': {
                'name': tool, 'description': 'The only permitted action at this phase.',
                'parameters': schema}}]
            request['tool_choice'] = {'type': 'function', 'function': {'name': tool}}
            request['parallel_tool_calls'] = False
        if repair_code is not None:
            system += (
                ' The previous attempt did not yield an accepted proposal. Recheck the response '
                'format, permitted tool, and every frozen binding. '
                + ('Return exactly one JSON object containing only tool_name, arguments and rationale; '
                   'do not add markdown, extra fields, duplicate keys, or multiple actions.'
                   if self.config.protocol == 'json_action' else
                   'Return exactly one native function call for the permitted tool, with exact '
                   'bound arguments and a brief rationale as message content; do not return parallel calls.')
            )
        request['messages'] = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': json.dumps(projected, ensure_ascii=False, separators=(',', ':'))},
        ]
        headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        if self._token:
            headers['Authorization'] = f'Bearer {self._token}'
        try:
            response = self._transport.request('POST', self.config.base_url + '/chat/completions',
                                               headers=headers, json=request, timeout=request_timeout)
        except LLMError:
            raise LLMError('llm_output_too_large') from None
        except Exception as exc:
            code = 'llm_timeout' if isinstance(exc, TimeoutError) or 'timeout' in type(exc).__name__.lower() else 'llm_unavailable'
            raise LLMError(code) from None
        usage = TokenUsage()
        try:
            if type(response.status_code) is not int or not 200 <= response.status_code < 300:
                raise LLMError('llm_http_error')
            raw_content = getattr(response, 'content', None)
            if isinstance(raw_content, bytes):
                if len(raw_content) > self.config.max_response_bytes:
                    raise LLMError('llm_output_too_large')
                payload = _strict_json(raw_content.decode('utf-8'))
            else:
                payload = response.json()
                if len(json.dumps(payload, allow_nan=False).encode()) > self.config.max_response_bytes:
                    raise LLMError('llm_output_too_large')
            if type(payload) is not dict:
                raise ValueError
            usage = _usage(payload)
            if payload.get('model', self.config.model) != self.config.model:
                raise ValueError
            choices = payload['choices']
            if type(choices) is not list or len(choices) != 1:
                raise ValueError
            choice = choices[0]
            if choice.get('index', 0) != 0 or choice.get('finish_reason') not in ('stop', 'tool_calls'):
                raise ValueError
            message = choice['message']
            if message.get('role') != 'assistant' or message.get('refusal') or message.get('function_call'):
                raise ValueError
            call_id = None
            if self.config.protocol == 'json_action':
                if message.get('tool_calls'):
                    raise ValueError
                action = _strict_json(message['content'])
                if type(action) is not dict or set(action) != {'tool_name', 'arguments', 'rationale'}:
                    raise ValueError
                name, arguments, rationale = action['tool_name'], action['arguments'], _rationale(action['rationale'])
            else:
                calls = message.get('tool_calls')
                if type(calls) is not list or len(calls) != 1:
                    raise ValueError
                call = calls[0]
                if set(call) != {'id', 'type', 'function'} or call['type'] != 'function':
                    raise ValueError
                call_id = validate_identifier(call['id'])
                function = call['function']
                if set(function) != {'name', 'arguments'}:
                    raise ValueError
                name, arguments = function['name'], _strict_json(function['arguments'])
                rationale = _rationale(message.get('content') or arguments.get('rationale'))
            if name != tool:
                raise ValueError
            arguments = validate_tool_arguments(name, arguments, {tool:schema} if step2 else None)
            for key, expected in projected['bindings'].items():
                if arguments.get(key) != expected:
                    raise ValueError
            if phase == 'submit':
                if arguments['model_type'] not in projected['capabilities']['models'] or 'parent_run_id' in arguments:
                    raise ValueError
                if 'rationale' in arguments and _rationale(arguments['rationale']) != rationale:
                    raise ValueError
                if step2:
                    arguments['model_params'] = dict(projected['capabilities']['fixed_model_params'][arguments['model_type']])
                arguments['rationale'] = rationale
            response_id = payload.get('id')
            if response_id is not None:
                response_id = validate_identifier(response_id)
            return Proposal(name, arguments, rationale, call_id, response_id, usage)
        except LLMError as exc:
            raise LLMError(exc.code, usage=usage) from None
        except Exception:
            raise LLMError('llm_output_invalid', usage=usage) from None

"""Benchmark semantic wire protocol with trusted native-action binding.

Only the benchmark driver opts into this adapter. The ordinary LLM adapter still
validates the frozen context and the translated native action. Actual wire input
and output are retained separately in ``last_semantic_trace`` for local auditing.
"""
from __future__ import annotations

import base64
import copy
from datetime import datetime, timezone
import hashlib
import json
import re
import time

from agent_poc.orchestration.llm import (
    LLMAdapter, LLMError, _BoundedHttpxTransport, _rationale, _strict_json,
)
from .model_catalog import SUPPORTED_MODEL_TYPES

SEMANTIC_VERSION = "benchmark-semantic-llm-v1"
LEGACY_PROMPT_VERSION = "benchmark-decision-v1"
OBJECTIVE_PROMPT_VERSION = "benchmark-decision-objective-v2"
OBJECTIVE_INSTRUCTION = (
    " Select the legal model expected to achieve the highest balanced accuracy on unseen samples."
    " Candidate display order does not indicate priority."
    " Compare candidates using the supplied task evidence."
)
_CONTEXTS = frozenset({"agent-context-search-v1", "agent-context-budget-v1"})


def _bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _semantic_action(text, trace):
    """Repair only serialization, retaining every supplied decision value."""
    normalizations = []
    try:
        action = _strict_json(text)
    except ValueError:
        quoted = escaped = False
        stack, output = [], []
        for character in text:
            if quoted:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    quoted = False
                elif character in "\n\r\t":
                    output.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[character])
                    normalizations.append("escape_string_whitespace")
                    continue
            elif character == '"':
                quoted = True
            elif character in "{[":
                stack.append("}" if character == "{" else "]")
            elif character in "}]":
                if not stack or stack.pop() != character:
                    raise ValueError("unmatched JSON delimiter")
            output.append(character)
        if quoted or escaped or stack not in ([], ["}"]):
            raise ValueError("incomplete semantic fields")
        if stack:
            output.append("}")
            normalizations.append("append_final_object_brace")
        action = _strict_json("".join(output))
    if type(action) is dict and type(action.get("rationale")) is str:
        rationale = action["rationale"].strip()
        if rationale != action["rationale"]:
            action["rationale"] = rationale
            normalizations.append("trim_rationale_edge_whitespace")
    if normalizations:
        trace["semantic_format_normalization"] = dict(
            operations=sorted(set(normalizations)), original_content_sha256=_sha(text.encode()),
            normalized_action_sha256=_sha(_bytes(action)), normalized_action=copy.deepcopy(action))
    return action


class _TranslatedResponse:
    def __init__(self, status_code, content):
        self.status_code, self.content = status_code, content

    def json(self):
        return _strict_json(self.content.decode("utf-8"))


class _SemanticTransport:
    def __init__(self, owner, provider):
        self.owner, self.provider = owner, provider

    def request(self, method, url, *, headers, json, timeout):
        owner = self.owner
        trace = owner.last_semantic_trace
        trace["dispatched"] = True
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        try:
            response = self.provider.request(method, url, headers=headers, json=json, timeout=timeout)
        finally:
            trace.update(provider_request_started_at_utc=started_at,
                         provider_response_ended_at_utc=datetime.now(timezone.utc).isoformat(),
                         provider_request_duration_seconds=time.monotonic() - started)
        raw = getattr(response, "content", None)
        if isinstance(raw, bytes):
            trace["raw_response_encoding"] = "exact-http-body-base64"
        else:
            raw = _bytes(response.json())
            trace["raw_response_encoding"] = "reserialized-json-base64"
        trace.update(raw_response_sha256=_sha(raw), raw_response_bytes=len(raw),
                     http_status=response.status_code)
        # Preserve the ordinary byte bound; do not retain an oversized body.
        if len(raw) > owner.config.max_response_bytes:
            trace["raw_response_retained"] = False
            return _TranslatedResponse(response.status_code, raw)
        trace.update(raw_response_retained=True,
                     raw_response_body_base64=base64.b64encode(raw).decode("ascii"))
        if type(response.status_code) is not int or not 200 <= response.status_code < 300:
            return _TranslatedResponse(response.status_code, raw)
        parse_started = time.monotonic()
        payload = None
        try:
            payload = _strict_json(raw.decode("utf-8"))
            translated = owner._translate(payload)
            trace["translation_status"] = "semantic_validated_then_trusted_binding"
            trace["translated_native_response"] = translated
            return _TranslatedResponse(response.status_code, _bytes(translated))
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            trace["translation_status"] = "rejected_semantic_output"
            # Never pass rejected semantic arguments through native validation:
            # even a correctly copied native UUID must fail this new protocol.
            rejected = {"choices": []}
            if type(payload) is dict and "usage" in payload:
                rejected["usage"] = payload["usage"]
            return _TranslatedResponse(response.status_code, _bytes(rejected))
        finally:
            trace["semantic_parse_duration_seconds"] = time.monotonic() - parse_started


class BenchmarkLLMAdapter(LLMAdapter):
    """Same propose API/config as LLMAdapter; benchmark-only semantic wire input.

    Contexts must come from the validated Session projection, with fixed
    processing, exactly one recipe per model, and disabled knowledge/budget
    visibility. Client-request/catalog bindings remain the driver's trusted
    submission bindings; they are never accepted from model output.
    """
    def __init__(self, config, *, token=None, transport=None,
                 decision_prompt_version=OBJECTIVE_PROMPT_VERSION):
        if decision_prompt_version not in (LEGACY_PROMPT_VERSION, OBJECTIVE_PROMPT_VERSION):
            raise ValueError("unsupported benchmark decision prompt")
        self.decision_prompt_version = decision_prompt_version
        if config.diagnosis_phase:
            raise ValueError("benchmark semantic adapter does not support diagnosis")
        self.last_semantic_trace = None
        provider = transport or _BoundedHttpxTransport(config.max_response_bytes)
        super().__init__(config, token=token, transport=_SemanticTransport(self, provider))

    def _build_request(self, phase, context, repair_code=None):
        # The existing adapter validates all fields, phase and config version.
        original, projected, schema, tool, knowledge, recipes, step2 = super()._build_request(
            phase, context, repair_code)
        try:
            if projected["context_version"] not in _CONTEXTS:
                raise ValueError("unsupported benchmark context")
            if projected["allowed_actions"] != [tool] or projected.get("budget_card") is not None or projected.get("budget_balance") is not None:
                raise ValueError("budget visibility not supported")
            semantic = {"semantic_version": SEMANTIC_VERSION, "phase": phase,
                        "task": copy.deepcopy(projected["task"]), "allowed_actions": [tool]}
            mapping = {}
            if phase == "submit":
                if (projected.get("decision_mode") != "recipe_id" or projected["processing_mode"] != "fixed"
                        or projected["knowledge"]["status"] != "disabled"):
                    raise ValueError("unsupported benchmark decision configuration")
                ids = set()
                candidates = []
                for recipe in projected["recipes"]:
                    model = recipe["model_id"]
                    recipe_id = recipe["recipe_id"]
                    if (model not in SUPPORTED_MODEL_TYPES or model in mapping or recipe_id in ids
                            or not re.fullmatch(r"recipe_[a-f0-9]{64}", recipe_id)):
                        raise ValueError("ambiguous or unknown frozen recipe")
                    processing = {key: recipe[key] for key in ("normalization", "class_balance")}
                    if projected["fixed_processing"][model] != processing:
                        raise ValueError("processing does not match frozen recipe")
                    mapping[model] = recipe_id
                    ids.add(recipe_id)
                    candidates.append(dict(model_id=model, **processing,
                        model_params=projected["fixed_model_params"][model],
                        **projected["model_profiles"][model]))
                if set(projected["fixed_model_params"]) != set(mapping):
                    raise ValueError("model parameter domain differs from frozen recipes")
                semantic.update(candidates=candidates, search_mode=projected["search_mode"],
                                max_trials=projected["max_trials"])
                for key in ("train_statistics", "train_risks"):
                    if projected.get(key) is not None:
                        semantic[key] = projected[key]
                semantic_schema = dict(type="object", additionalProperties=False,
                    properties={"model_id": {"type": "string", "enum": list(mapping)}}, required=["model_id"])
            elif phase == "finalize":
                semantic.update(validation=projected["validation"], candidate={
                    key: projected["candidate"][key] for key in ("validation_score", "eligibility")})
                semantic_schema = dict(type="object", additionalProperties=False, properties={}, required=[])
            else:
                raise ValueError("unsupported phase")
        except (ValueError, KeyError, TypeError):
            raise LLMError("llm_context_invalid") from None
        system = ("Select the permitted action for this classification benchmark using only supplied evidence. "
                  "Choose one canonical model_id from candidates for submit_ml_experiment. "
                  "For finalize_ml_session return empty arguments {}. "
                  "The backend binds all execution identifiers. Never supply identifiers or execution parameters. "
                  "Give one short nonempty rationale sentence without private reasoning or external references. ")
        request = {key: value for key, value in original.items()
                   if key not in {"messages", "tools", "tool_choice", "parallel_tool_calls"}}
        if self.config.protocol == "json_action":
            system += ("Respond with exactly one compact JSON object on one physical line, containing only tool_name, arguments and rationale. "
                       "Keep all string values on that line and escape every control character. "
                       "No markdown, extra fields, duplicate keys or parallel actions. Permitted tool schema: "
                       + json.dumps({tool: semantic_schema}, separators=(",", ":")))
        else:
            system += "Return exactly one native function call and the rationale as message content."
            request.update(tools=[{"type": "function", "function": {"name": tool,
                "description": "The permitted benchmark action.", "parameters": semantic_schema}}],
                tool_choice={"type": "function", "function": {"name": tool}}, parallel_tool_calls=False)
        if repair_code is not None:
            system += " The previous output was rejected. Return valid compact JSON on one physical line using the exact permitted schema; quoted strings must contain no literal line breaks."
        if self.decision_prompt_version == OBJECTIVE_PROMPT_VERSION:
            # The transport's 8:1:1 fields are not the trusted benchmark partition.
            # Remove that misleading placeholder without adding dataset/architecture descriptions.
            semantic["task"].pop("evaluation_config", None)
            if phase == "submit":
                system += OBJECTIVE_INSTRUCTION
        request["messages"] = [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(semantic, ensure_ascii=False, separators=(",", ":"))}]
        self._semantic_binding = (projected, mapping, tool)
        self.last_semantic_trace = dict(semantic_version=SEMANTIC_VERSION, phase=phase,
            decision_prompt_version=self.decision_prompt_version,
            trusted_context_sha256=_sha(_bytes(projected)), semantic_input_sha256=_sha(_bytes(semantic)),
            actual_request_sha256=_sha(_bytes(request)), actual_request=copy.deepcopy(request), dispatched=False)
        return request, projected, schema, tool, knowledge, recipes, step2

    def _translate(self, payload):
        projected, mapping, expected_tool = self._semantic_binding
        translated = copy.deepcopy(payload)
        choices = translated["choices"]
        if type(choices) is not list or len(choices) != 1:
            raise ValueError("single action required")
        message = choices[0]["message"]
        if self.config.protocol == "json_action":
            if message.get("tool_calls"):
                raise ValueError("unexpected tool calls")
            action = _semantic_action(message["content"], self.last_semantic_trace)
            if type(action) is not dict or set(action) != {"tool_name", "arguments", "rationale"}:
                raise ValueError("invalid action keys")
            name, arguments = action["tool_name"], action["arguments"]
            _rationale(action["rationale"])
        else:
            calls = message["tool_calls"]
            if type(calls) is not list or len(calls) != 1:
                raise ValueError("single call required")
            call = calls[0]
            if set(call) != {"id", "type", "function"} or call["type"] != "function":
                raise ValueError("invalid native call")
            function = call["function"]
            if set(function) != {"name", "arguments"}:
                raise ValueError("invalid function keys")
            name, arguments = function["name"], _strict_json(function["arguments"])
            _rationale(message.get("content"))
        if name != expected_tool or type(arguments) is not dict:
            raise ValueError("unexpected semantic action")
        native = dict(projected["bindings"])
        if projected["phase"] == "submit":
            if set(arguments) != {"model_id"} or type(arguments["model_id"]) is not str or arguments["model_id"] not in mapping:
                raise ValueError("invalid semantic model choice")
            native.update(recipe_id=mapping[arguments["model_id"]], knowledge_refs=[])
        elif arguments:
            raise ValueError("finalization requires empty arguments")
        if self.config.protocol == "json_action":
            action["arguments"] = native
            message["content"] = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
        else:
            function["arguments"] = json.dumps(native, ensure_ascii=False, separators=(",", ":"))
        return translated

    def propose(self, phase, context, **kwargs):
        self.last_semantic_trace = None
        try:
            proposal = super().propose(phase, context, **kwargs)
            self.last_semantic_trace["native_validation_status"] = "accepted"
            return proposal
        except Exception as error:
            if self.last_semantic_trace is not None:
                self.last_semantic_trace["native_validation_status"] = "failed"
                self.last_semantic_trace["error_code"] = getattr(error, "code", "llm_unexpected")
            raise
        finally:
            trace, measurement = self.last_semantic_trace, self.last_request_measurement
            if trace is not None and measurement is not None and "provider_request_duration_seconds" in trace:
                # Client request duration is the actual provider wait; semantic
                # translation belongs to parsing, not provider compute time.
                measurement.update(request_started_at_utc=trace["provider_request_started_at_utc"],
                    response_ended_at_utc=trace["provider_response_ended_at_utc"],
                    request_duration_seconds=trace["provider_request_duration_seconds"])
                measurement["parse_duration_seconds"] = (measurement.get("parse_duration_seconds", 0)
                    + trace.get("semantic_parse_duration_seconds", 0))
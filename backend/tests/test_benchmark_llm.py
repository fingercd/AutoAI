"""Offline semantic-wire tests; no fitting or real provider requests."""
import base64
import copy
import json

import pytest

from agent_poc.orchestration.llm import LLMConfig, LLMError, store_proposal
from agent_poc.orchestration.projection import finalization_context
from backend.app.benchmark_llm import BenchmarkLLMAdapter

SESSION = "27d2d5fd8b644c36845a298ec5e67a45"
RUN = "run-opaque-fixture"
RECIPE = "recipe_" + "a" * 64
TASK = dict(task_type="classification", selection_metric="balanced_accuracy", seed=42,
            evaluation_config=dict(mode="stratified_holdout", train_weight=8, validation_weight=1, heldout_weight=1))


def context():
    return dict(context_version="agent-context-budget-v1", phase="submit", task=TASK,
        bindings=dict(session_id=SESSION), allowed_actions=["submit_ml_experiment"],
        recipes=[dict(recipe_id=RECIPE, model_id="svm", normalization="zscore", class_balance="none")],
        decision_mode="recipe_id", fixed_model_params=dict(svm=dict(svm_c=1.0, svm_kernel="linear")),
        processing_mode="fixed", fixed_processing=dict(svm=dict(normalization="zscore", class_balance="none")),
        model_profiles=dict(svm=dict(architecture_version="docx-classification-v2", model_policy_version="agent-model-config-v3",
                                   search_strategy_version="training-execution-policy-v1", cost_description="At most 1 candidates")),
        search_mode="bounded", max_trials=1,
        knowledge=dict(status="disabled", projection_version=None, matched_count=0, provided_count=0,
                       omitted_count=0, provided_entry_ids=[], entries=[]), train_statistics=None, train_risks=None)


def final_context():
    return finalization_context(task=TASK, session_id=SESSION, run_id=RUN,
        validation=dict(balanced_accuracy=.7), validation_score=.7,
        allowed_actions=["finalize_ml_session"], context_version="agent-context-budget-v1")


def action(finalize=False):
    return dict(tool_name="finalize_ml_session" if finalize else "submit_ml_experiment",
                arguments={} if finalize else dict(model_id="svm"), rationale="Select the permitted candidate.")


def envelope(value=None, protocol="json_action"):
    value = action() if value is None else value
    message = dict(role="assistant", content=json.dumps(value) if protocol == "json_action" else value["rationale"])
    if protocol == "native_tools":
        message["tool_calls"] = [dict(id="call-fixture", type="function",
            function=dict(name=value["tool_name"], arguments=json.dumps(value["arguments"])))]
    return dict(id="response-fixture", model="qwen35_9b", choices=[dict(index=0,
        finish_reason="stop" if protocol == "json_action" else "tool_calls", message=message)],
        usage=dict(prompt_tokens=20, completion_tokens=10, total_tokens=30, prompt_tokens_details=dict(cached_tokens=4)))


class Response:
    def __init__(self, payload, status=200):
        self.status_code = status
        self.content = json.dumps(payload).encode()


class Transport:
    def __init__(self, response):
        self.response, self.calls = response, []

    def request(self, method, url, *, headers, json, timeout):
        self.calls.append(dict(request=copy.deepcopy(json), timeout=timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def adapter(payload=None, protocol="json_action", status=200, **options):
    provider = Transport(Response(envelope(protocol=protocol) if payload is None else payload, status))
    llm = BenchmarkLLMAdapter(LLMConfig("http://localhost/v1", "qwen35_9b", protocol=protocol,
        prompt_version="agent-decision-budget-v1", **options), transport=provider)
    return llm, provider


@pytest.mark.parametrize("protocol", ["json_action", "native_tools"])
def test_semantic_choice_trusted_mapping_audit_and_actual_accounting(protocol):
    llm, provider = adapter(protocol=protocol)
    dispatched = []
    proposal = llm.propose("submit", context(), on_dispatch=lambda: dispatched.append(True) or .75)
    assert proposal.arguments == dict(session_id=SESSION, recipe_id=RECIPE,
        knowledge_refs=[], rationale=action()["rationale"])
    assert store_proposal(proposal, context())["arguments"]["recipe_id"] == RECIPE
    assert proposal.usage.total_tokens == 30 and proposal.usage.cached_tokens == 4
    assert dispatched == [True] and provider.calls[0]["timeout"] == .75
    wire = json.dumps(provider.calls[0]["request"])
    for opaque in (SESSION, RUN, RECIPE, "session_id", "recipe_id", "client_request_id", "catalog_digest", "selected_run_id"):
        assert opaque not in wire
    semantic = json.loads(provider.calls[0]["request"]["messages"][1]["content"])
    assert semantic["candidates"][0]["model_id"] == "svm"
    assert semantic["candidates"][0]["model_params"]["svm_c"] == 1.0
    trace = llm.last_semantic_trace
    raw = json.loads(base64.b64decode(trace["raw_response_body_base64"]))
    assert raw == envelope(protocol=protocol)
    assert SESSION not in json.dumps(raw)
    assert SESSION in json.dumps(trace["translated_native_response"])
    assert trace["translation_status"] == "semantic_validated_then_trusted_binding"
    assert len(trace["semantic_input_sha256"]) == 64
    assert "headers" not in trace and "base_url" not in trace
    assert llm.last_request_measurement["request_duration_seconds"] == trace["provider_request_duration_seconds"]
    assert llm.last_request_measurement["parse_duration_seconds"] >= trace["semantic_parse_duration_seconds"]


@pytest.mark.parametrize("protocol", ["json_action", "native_tools"])
def test_finalization_requires_empty_semantic_arguments_and_binds_run(protocol):
    llm, provider = adapter(envelope(action(True), protocol), protocol)
    proposal = llm.propose("finalize", final_context())
    assert proposal.arguments == dict(session_id=SESSION, selected_run_id=RUN)
    assert proposal.rationale == action(True)["rationale"]
    wire = json.dumps(provider.calls[0]["request"])
    assert SESSION not in wire and RUN not in wire and "selected_run_id" not in wire


@pytest.mark.parametrize("arguments", [dict(model_id="rf"), dict(model_id="unknown"), {},
    dict(model_id=["svm"]), dict(model_id="svm", session_id=SESSION),
    dict(model_id="svm", recipe_id=RECIPE), dict(model_id="svm", rationale="extra"),
    dict(model_id="svm", catalog_digest="x"), dict(session_id=SESSION, recipe_id=RECIPE, knowledge_refs=[])])
@pytest.mark.parametrize("protocol", ["json_action", "native_tools"])
def test_unknown_extra_and_native_identity_outputs_rejected(arguments, protocol):
    output = action(); output["arguments"] = arguments
    llm, provider = adapter(envelope(output, protocol), protocol)
    with pytest.raises(LLMError) as error:
        llm.propose("submit", context())
    assert error.value.code == "llm_output_invalid" and error.value.usage.total_tokens == 30
    assert len(provider.calls) == 1
    assert llm.last_semantic_trace["translation_status"] == "rejected_semantic_output"


@pytest.mark.parametrize("arguments", [dict(session_id=SESSION, selected_run_id=RUN), dict(model_id="svm"), []])
def test_finalization_model_cannot_supply_even_correct_ids(arguments):
    output = action(True); output["arguments"] = arguments
    llm, _ = adapter(envelope(output))
    with pytest.raises(LLMError, match="llm_output_invalid"):
        llm.propose("finalize", final_context())


@pytest.mark.parametrize("content", ["not json", "```json\n{}\n```", '{"tool_name":"submit_ml_experiment","tool_name":"shell"}',
    '{"tool_name":"submit_ml_experiment","arguments":{"model_id":"svm","model_id":"svm"},"rationale":"Select model."}',
    json.dumps({**action(), "extra": 1}), json.dumps({**action(), "tool_name": "shell"})])
def test_malformed_duplicates_extra_keys_and_unknown_tools_rejected(content):
    payload = envelope(); payload["choices"][0]["message"]["content"] = content
    llm, _ = adapter(payload)
    with pytest.raises(LLMError) as error:
        llm.propose("submit", context())
    assert error.value.code == "llm_output_invalid" and error.value.usage.total_tokens == 30


@pytest.mark.parametrize("mutation", ["extra_context", "duplicate_recipe", "unknown_model", "processing", "memory", "budget", "params"])
def test_ambiguous_or_out_of_scope_context_rejected_before_dispatch(mutation):
    value = context()
    if mutation == "extra_context": value["test_metrics"] = {}
    elif mutation == "duplicate_recipe": value["recipes"].append(copy.deepcopy(value["recipes"][0]))
    elif mutation == "unknown_model": value["recipes"][0]["model_id"] = "uuid_model"
    elif mutation == "processing": value["fixed_processing"]["svm"]["normalization"] = "none"
    elif mutation == "memory": value["knowledge"].update(status="ready", projection_version="knowledge-projection-v1")
    elif mutation == "budget": value["allowed_actions"].append("stop_ml_session")
    elif mutation == "params": value["fixed_model_params"]["random_forest"] = {}
    llm, provider = adapter()
    with pytest.raises(LLMError, match="llm_context_invalid"):
        llm.propose("submit", value)
    assert not provider.calls


@pytest.mark.parametrize("status", [400, 503])
def test_provider_http_errors_keep_actual_usage_and_body(status):
    payload = dict(usage=dict(prompt_tokens=9, completion_tokens=1, total_tokens=10), error="fixture failure")
    llm, _ = adapter(payload, status=status)
    with pytest.raises(LLMError) as error:
        llm.propose("submit", context())
    assert error.value.code == "llm_http_error" and error.value.usage.total_tokens == 10
    assert json.loads(base64.b64decode(llm.last_semantic_trace["raw_response_body_base64"])) == payload


@pytest.mark.parametrize("error,code", [(TimeoutError("fixture"), "llm_timeout"), (OSError("fixture"), "llm_unavailable")])
def test_provider_failures_remain_measured(error, code):
    llm, provider = adapter(); provider.response = error
    with pytest.raises(LLMError) as caught:
        llm.propose("submit", context())
    assert caught.value.code == code
    assert llm.last_request_measurement["outcome"] == "failed"
    assert llm.last_semantic_trace["dispatched"] is True
    assert len(provider.calls) == 1


def test_dispatch_callback_failure_is_retained_and_prevents_provider_call():
    llm, provider = adapter()
    def fail(): raise RuntimeError("journal callback failed")
    with pytest.raises(RuntimeError, match="journal callback failed"):
        llm.propose("submit", context(), on_dispatch=fail)
    assert not provider.calls and llm.last_semantic_trace["dispatched"] is False
    assert llm.last_request_measurement is None


def test_hard_output_limit_applies_to_actual_response():
    payload = envelope(); payload["padding"] = "x" * 2000
    llm, _ = adapter(payload, max_response_bytes=1024)
    with pytest.raises(LLMError, match="llm_output_too_large"):
        llm.propose("submit", context())
    assert llm.last_semantic_trace["raw_response_retained"] is False
    assert "raw_response_body_base64" not in llm.last_semantic_trace


def test_original_envelope_validation_still_checks_served_model_and_role():
    for field, value in (("model", "other"), ("role", "user")):
        payload = envelope()
        if field == "model": payload[field] = value
        else: payload["choices"][0]["message"][field] = value
        llm, _ = adapter(payload)
        with pytest.raises(LLMError) as error:
            llm.propose("submit", context())
        assert error.value.code == "llm_output_invalid" and error.value.usage.total_tokens == 30


@pytest.mark.parametrize("content,operations", [
    ('{"tool_name":"submit_ml_experiment","arguments":{"model_id":"svm"},"rationale":"Use the supplied model."',
     {"append_final_object_brace"}),
    ('{"tool_name":"submit_ml_experiment","arguments":{"model_id":"svm"},"rationale":"Use the supplied model.\n"}',
     {"escape_string_whitespace", "trim_rationale_edge_whitespace"}),
    ('{"tool_name":"submit_ml_experiment","arguments":{"model_id":"svm"},"rationale":"Use the supplied model.\n"',
     {"append_final_object_brace", "escape_string_whitespace", "trim_rationale_edge_whitespace"}),
])
def test_bounded_serialization_repair_preserves_real_decision_and_raw_evidence(content, operations):
    payload = envelope(); payload["choices"][0]["message"]["content"] = content
    llm, _ = adapter(payload)
    proposal = llm.propose("submit", context())
    assert proposal.arguments["recipe_id"] == RECIPE
    assert proposal.rationale == "Use the supplied model."
    trace = llm.last_semantic_trace
    assert json.loads(base64.b64decode(trace["raw_response_body_base64"])) == payload
    normalization = trace["semantic_format_normalization"]
    assert set(normalization["operations"]) == operations
    assert normalization["normalized_action"]["arguments"] == {"model_id": "svm"}
    assert normalization["normalized_action"]["rationale"] == proposal.rationale
    assert proposal.usage.total_tokens == 30


@pytest.mark.parametrize("content", [
    '{"tool_name":"submit_ml_experiment","arguments":{"model_id":"svm"',
    '{"tool_name":"submit_ml_experiment","arguments":{"model_id":"svm',
    '{"tool_name":"submit_ml_experiment","arguments":{"model_id":"svm","model_id":"svm"},"rationale":"Use model."',
    '{"tool_name":"submit_ml_experiment","arguments":{"model_id":"svm\n"},"rationale":"Use model."}',
    '{"tool_name":"submit_ml_experiment","arguments":{"model_id":"svm"},"rationale":"Use\nmodel."}',
    '{"tool_name":"submit_ml_experiment","arguments":{"model_id":"unknown"},"rationale":"Use model."',
    '{"tool_name":"submit_ml_experiment","arguments":{"model_id":"svm"},"rationale":"Use model."}{',
])
def test_serialization_repair_cannot_invent_decisions_or_relax_semantic_checks(content):
    payload = envelope(); payload["choices"][0]["message"]["content"] = content
    llm, provider = adapter(payload)
    with pytest.raises(LLMError, match="llm_output_invalid"):
        llm.propose("submit", context())
    assert len(provider.calls) == 1
    assert llm.last_semantic_trace["translation_status"] == "rejected_semantic_output"

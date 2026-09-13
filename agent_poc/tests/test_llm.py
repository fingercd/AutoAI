import copy
import json
import traceback
import pytest

from agent_poc.orchestration.llm import LLMAdapter, LLMConfig, LLMError
from agent_poc.orchestration.projection import selection_context, finalization_context
from agent_poc.tests.test_autoai_client import FakeTransport, Response, EVALUATION

TASK = {'task_type': 'classification', 'selection_metric': 'macro_f1', 'seed': 42,
        'allowed_models': ['logistic_regression', 'svm'], 'evaluation_config': EVALUATION}


def context():
    return selection_context(task=TASK, models=['logistic_regression', 'svm', 'random_forest'],
                             session_id='s', client_request_id='request-1')


def action():
    return {'tool_name': 'submit_ml_experiment', 'arguments': {
        'session_id': 's', 'model_type': 'svm', 'normalization': 'zscore',
        'class_balance': 'none', 'client_request_id': 'request-1'},
        'rationale': 'Choose SVM for this classification experiment.'}


def envelope(content, *, usage=True):
    result = {'id': 'chatcmpl-1', 'model': 'local-model', 'choices': [{
        'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': content}}]}
    if usage:
        result['usage'] = {'prompt_tokens': 20, 'completion_tokens': 10, 'total_tokens': 30}
    return result


def adapter(payload, protocol='json_action'):
    return LLMAdapter(LLMConfig('http://localhost:9000/v1', 'local-model', protocol=protocol),
                      transport=FakeTransport([Response(200, payload)]))


def test_real_model_choice_is_validated_and_not_overridden():
    llm = adapter(envelope(json.dumps(action())))
    proposal = llm.propose('submit', context())
    assert proposal.tool_name == 'submit_ml_experiment'
    assert proposal.arguments['model_type'] == 'svm'
    assert proposal.arguments['rationale'] == action()['rationale']
    assert proposal.usage.total_tokens == 30
    assert proposal.usage.status == 'known'
    assert len(llm._transport.calls) == 1


def test_first_experiment_prompt_exposes_only_stage_valid_arguments():
    llm = adapter(envelope(json.dumps(action())))
    llm.propose('submit', context())
    system = llm._transport.calls[0][3]['messages'][0]['content']
    schema = json.loads(system.split('The permitted tool schema is: ', 1)[1])
    assert 'parent_run_id' not in schema['properties']
    assert 'rationale' not in schema['properties']
    assert schema['properties']['model_type']['enum'] == ['logistic_regression', 'svm']
    for key, value in context()['bindings'].items():
        assert schema['properties'][key]['enum'] == [value]
        assert key in schema['required']
    from agent_poc.tools import TOOL_SCHEMAS
    assert 'parent_run_id' in TOOL_SCHEMAS['submit_ml_experiment']['properties']


@pytest.mark.parametrize('mutate', [
    lambda value: value.update(tool_name='shell'),
    lambda value: value.update(extra='SYNTHETIC_SECRET'),
    lambda value: value['arguments'].update(session_id='other'),
    lambda value: value['arguments'].update(client_request_id='new-key'),
    lambda value: value['arguments'].update(model_type='random_forest'),
    lambda value: value['arguments'].update(normalization='none'),
    lambda value: value['arguments'].update(class_balance='class_weight'),
    lambda value: value['arguments'].update(seed=7),
    lambda value: value['arguments'].update(parent_run_id='other'),
    lambda value: value['arguments'].update(session_id='../private'),
    lambda value: value.update(rationale='Authorization: Bearer SYNTHETIC_SECRET'),
])
def test_invalid_proposals_are_rejected_without_retaining_raw_output(mutate):
    proposal = action()
    mutate(proposal)
    llm = adapter(envelope(json.dumps(proposal)))
    with pytest.raises(LLMError) as info:
        llm.propose('submit', context())
    assert info.value.code == 'llm_output_invalid'
    assert info.value.usage.total_tokens == 30
    assert 'SYNTHETIC_SECRET' not in repr(info.value) + ''.join(traceback.format_exception(info.value))
    assert len(llm._transport.calls) == 1  # Repairs are separately persisted by the runner.


@pytest.mark.parametrize('content', [
    'not JSON SYNTHETIC_SECRET', json.dumps([action(), action()]),
    '{"tool_name":"submit_ml_experiment","tool_name":"shell"}',
    '```json\n{}\n```', '{"bad": NaN}',
])
def test_non_json_multi_action_and_duplicate_fields_fail_closed(content):
    with pytest.raises(LLMError) as info:
        adapter(envelope(content)).propose('submit', context())
    assert info.value.code == 'llm_output_invalid'
    assert info.value.usage.total_tokens == 30


def test_projection_removes_raw_state_paths_predictions_and_result_links():
    sentinel = 'SYNTHETIC_SECRET'
    task = {**TASK, 'raw': sentinel, 'data_path': sentinel,
            'evaluation_config': {**EVALUATION, 'raw': sentinel}}
    projected = selection_context(task=task, models=['svm'], session_id='s', client_request_id='request-1')
    llm = adapter(envelope(json.dumps(action())))
    llm.propose('submit', projected)
    assert sentinel not in json.dumps(llm._transport.calls[0][3])
    final_context = finalization_context(
        task=task, session_id='s', run_id='r', validation={'metrics': {'macro_f1': 0.0,
            'hidden': sentinel}, 'final_result_url': sentinel}, validation_score=0.0,
        allowed_actions=['finalize_ml_session'])
    assert sentinel not in json.dumps(final_context)
    assert final_context['candidate']['validation_score'] == 0.0


def test_unknown_context_fields_fail_before_network():
    llm = adapter(envelope(json.dumps(action())))
    with pytest.raises(LLMError) as info:
        llm.propose('submit', {**context(), 'raw': 'SYNTHETIC_SECRET'})
    assert info.value.code == 'llm_context_invalid'
    assert not llm._transport.calls


def test_finalize_is_an_llm_proposal_bound_to_the_candidate():
    final_context = finalization_context(task=TASK, session_id='s', run_id='r',
        validation={'macro_f1': 0.0}, validation_score=0.0, allowed_actions=['finalize_ml_session'])
    proposed = {'tool_name': 'finalize_ml_session', 'arguments': {'session_id': 's', 'selected_run_id': 'r'},
                'rationale': 'Finalize the sole eligible candidate.'}
    result = adapter(envelope(json.dumps(proposed))).propose('finalize', final_context)
    assert result.arguments == proposed['arguments']
    proposed['arguments']['selected_run_id'] = 'other'
    with pytest.raises(LLMError):
        adapter(envelope(json.dumps(proposed))).propose('finalize', final_context)


def native_envelope():
    proposal = action()
    return {'id': 'chatcmpl-1', 'model': 'local-model', 'choices': [{
        'index': 0, 'finish_reason': 'tool_calls', 'message': {
            'role': 'assistant', 'content': proposal['rationale'], 'tool_calls': [{
                'id': 'call-1', 'type': 'function', 'function': {
                    'name': proposal['tool_name'], 'arguments': json.dumps(proposal['arguments'])}}]}}]}


def test_native_protocol_is_explicit_and_validates_call_association():
    llm = adapter(native_envelope(), 'native_tools')
    result = llm.propose('submit', context())
    assert result.arguments['model_type'] == 'svm'
    assert result.tool_call_id == 'call-1'
    assert result.response_id == 'chatcmpl-1'
    assert result.usage.total_tokens is None and result.usage.status == 'unknown'
    sent = llm._transport.calls[0][3]
    assert sent['parallel_tool_calls'] is False
    assert len(sent['tools']) == 1
    assert sent['tools'][0]['function']['name'] == 'submit_ml_experiment'


@pytest.mark.parametrize('mutate', [
    lambda payload: payload['choices'][0]['message']['tool_calls'].append(
        copy.deepcopy(payload['choices'][0]['message']['tool_calls'][0])),
    lambda payload: payload['choices'].append(copy.deepcopy(payload['choices'][0])),
    lambda payload: payload['choices'][0]['message']['tool_calls'][0].update(id='../secret'),
    lambda payload: payload['choices'][0]['message']['tool_calls'][0]['function'].update(name='shell'),
])
def test_native_multi_call_unknown_tool_and_invalid_ids_fail_closed(mutate):
    payload = native_envelope()
    mutate(payload)
    with pytest.raises(LLMError):
        adapter(payload, 'native_tools').propose('submit', context())


def test_protocol_does_not_silently_switch_to_native():
    with pytest.raises(LLMError):
        adapter(native_envelope(), 'json_action').propose('submit', context())


def test_response_limit_and_unknown_usage_are_explicit():
    llm = LLMAdapter(LLMConfig('http://localhost/v1', 'local-model', max_response_bytes=1024),
                      transport=FakeTransport([Response(200, envelope('x' * 5000))]))
    with pytest.raises(LLMError) as info:
        llm.propose('submit', context())
    assert info.value.code == 'llm_output_too_large'
    result = adapter(envelope(json.dumps(action()), usage=False)).propose('submit', context())
    assert result.usage.status == 'unknown'
    assert result.usage.total_tokens is None


def test_transport_secrets_are_not_in_repr_or_traceback_and_no_hidden_retries():
    sentinel = 'SYNTHETIC_SECRET'
    llm = LLMAdapter(LLMConfig('http://localhost/v1', 'local-model'), token=sentinel,
                      transport=FakeTransport([TimeoutError(sentinel)]))
    with pytest.raises(LLMError) as info:
        llm.propose('submit', context())
    assert info.value.code == 'llm_timeout'
    assert sentinel not in repr(llm) + repr(info.value) + ''.join(traceback.format_exception(info.value))
    assert len(llm._transport.calls) == 1


def test_config_is_frozen_and_endpoint_sensitive_without_exposing_endpoint():
    first = LLMConfig('http://localhost/v1', 'local-model')
    second = LLMConfig('http://other/v1', 'local-model')
    assert first.fingerprint() != second.fingerprint()
    assert 'base_url' not in first.public_config()
    with pytest.raises(AttributeError):
        first.model = 'changed'
    credential_url = 'http://user:SYNTHETIC_SECRET@localhost/v1'
    with pytest.raises(ValueError) as info:
        LLMConfig(credential_url, 'local-model')
    assert 'SYNTHETIC_SECRET' not in ''.join(traceback.format_exception(info.value))


@pytest.mark.parametrize('model', ['http://private', 'file:/private', 'file://private',
                                  'model?credential=value', 'model#private', '', 'model\nprivate'])
def test_served_model_rejects_unsafe_or_malformed_identifiers(model):
    with pytest.raises(ValueError):
        LLMConfig('http://localhost/v1', model)


@pytest.mark.parametrize('prefix', ['/srv/models/', 'C:/models/', 'C:\\models\\'])
def test_path_named_provider_model_is_private_http_metadata(prefix):
    sentinel = 'SYNTHETIC_SECRET_MODEL_PATH'
    served_id = prefix + sentinel
    config = LLMConfig('http://localhost/v1', served_id)
    payload = envelope(json.dumps(action()))
    payload['model'] = served_id
    llm = LLMAdapter(config, transport=FakeTransport([Response(200, payload)]))
    proposal = llm.propose('submit', context())
    assert llm._transport.calls[0][3]['model'] == served_id
    assert sentinel not in json.dumps(llm._transport.calls[0][3]['messages'])
    assert sentinel not in json.dumps(config.public_config()) + repr(config) + repr(llm) + repr(proposal)
    assert len(config.public_config()['model_id_sha256']) == 64


def test_fingerprint_compatible_with_private_model_binding():
    import hashlib
    config = LLMConfig('http://localhost/v1', 'local-model')
    original = {**config.public_config(), 'endpoint': config.base_url, 'model': config.model}
    original.pop('model_id_sha256')
    expected = hashlib.sha256(json.dumps(original, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert config.fingerprint() == expected


def test_finalization_context_rejects_incoherent_bound_candidate_before_network():
    projected = finalization_context(task=TASK, session_id='s', run_id='r',
        validation={'macro_f1': 0.5}, validation_score=0.5, allowed_actions=['finalize_ml_session'])
    projected['candidate']['run_id'] = 'other'
    llm = adapter(envelope(json.dumps(action())))
    with pytest.raises(LLMError) as info:
        llm.propose('finalize', projected)
    assert info.value.code == 'llm_context_invalid'
    assert not llm._transport.calls


def test_finalization_context_refuses_pending_validation():
    with pytest.raises(ValueError):
        finalization_context(task=TASK, session_id='s', run_id='r',
            validation={'status': 'pending', 'metrics': {'macro_f1': 0.5}},
            validation_score=0.5, allowed_actions=['finalize_ml_session'])


def test_remaining_deadline_only_clips_current_request_timeout():
    llm = adapter(envelope(json.dumps(action())))
    fingerprint = llm.config.fingerprint()
    llm.propose('submit', context(), timeout_seconds=0.4)
    assert llm._transport.calls[0][4] == 0.4
    assert llm.config.timeout == 30.0 and llm.config.fingerprint() == fingerprint
    idle = adapter(envelope(json.dumps(action())))
    with pytest.raises(LLMError) as info:
        idle.propose('submit', context(), timeout_seconds=0)
    assert info.value.code == 'llm_timeout'
    assert not idle._transport.calls


def test_real_http_transport_bounds_entire_drip_stream(monkeypatch):
    import asyncio
    import time
    import httpx
    from agent_poc.clients.http_transport import bounded_request

    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(100):
                await asyncio.sleep(0.02)
                yield b' '

    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=SlowBody()))
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: real_client(transport=transport, **kwargs))
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        bounded_request('POST', 'http://localhost/v1/chat/completions', headers={}, json={},
                        timeout=0.15, max_response_bytes=4096)
    assert time.monotonic() - started < 0.8


def test_repair_request_adds_fixed_guidance_without_the_bad_response():
    sentinel = 'SYNTHETIC_SECRET_BAD_RESPONSE'
    transport = FakeTransport([Response(200, envelope(sentinel)),
                               Response(200, envelope(json.dumps(action())))])
    llm = LLMAdapter(LLMConfig('http://localhost/v1', 'local-model'), transport=transport)
    with pytest.raises(LLMError) as info:
        llm.propose('submit', context())
    proposal = llm.propose('submit', context(), repair_code=info.value.code)
    assert proposal.arguments['model_type'] == 'svm'
    second_messages = transport.calls[1][3]['messages']
    assert 'previous attempt did not yield an accepted proposal' in second_messages[0]['content']
    assert 'exactly one JSON object' in second_messages[0]['content']
    assert second_messages[0]['content'] != transport.calls[0][3]['messages'][0]['content']
    assert second_messages[1] == transport.calls[0][3]['messages'][1]
    assert sentinel not in json.dumps(second_messages)
    with pytest.raises(LLMError) as invalid:
        llm.propose('submit', context(), repair_code=sentinel)
    assert invalid.value.code == 'llm_context_invalid'
    assert len(transport.calls) == 2

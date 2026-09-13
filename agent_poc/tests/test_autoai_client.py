import copy
import json
import traceback
import pytest

from agent_poc.clients.autoai_client import (
    AgentConnectionError, AgentContractError, AgentHTTPError, AutoAIClient,
)
from agent_poc.tools import ToolDispatcher, TOOL_SCHEMAS

VERSION = {'contract_version': 'agent-session-v1'}
ACTION = {'model_type': 'logistic_regression', 'normalization': 'zscore',
          'class_balance': 'none', 'parent_run_id': None}
STAMP = '2026-09-12T10:00:00+00:00'
EVALUATION = {'mode': 'stratified_holdout', 'train_weight': 8,
              'validation_weight': 1, 'heldout_weight': 1}
METADATA = {'metadata_version': 'agent-metadata-v1', 'dataset_sha256': 'a' * 64,
            'dataset_fingerprint_status': 'ready'}
EXPERIMENT_METADATA = {**METADATA, 'effective_config_status': 'ready', 'effective_config': {
    **{key: ACTION[key] for key in ('model_type', 'normalization', 'class_balance')},
    'seed': 42, 'feature_selection_enabled': False, 'evaluation_config': EVALUATION}}
LOCKED = {**METADATA, 'dataset_id': 'ds', 'selection_metric': 'macro_f1',
          'allowed_models': ['logistic_regression'], 'max_runs': 1, 'seed': 42,
          'evaluation_config': EVALUATION, 'modules': [],
          'context_policy': {'source_role': 'development', 'case_write': False}}
HEALTH = {**VERSION, 'status': 'ready', 'capabilities': {
    'create_session': True, 'create_experiment': True, 'read_session': True,
    'read_feedback': True, 'finalize_session': True},
    'models': ['logistic_regression', 'svm', 'random_forest'], 'modules': {}}
CREATED = {**VERSION, 'session_id': 's', 'state': 'open', 'remaining_runs': 1,
           'locked_config': LOCKED, 'created_at': STAMP, 'idempotent_replay': False}
SESSION = {**VERSION, 'session_id': 's', 'state': 'open', 'remaining_runs': 1,
           'locked_config': LOCKED, 'created_at': STAMP, 'finalized_at': None,
           'best_run_id': None, 'selected_run_id': None, 'experiments': []}
EXPERIMENT = {**VERSION, **EXPERIMENT_METADATA, 'session_id': 's', 'run_id': 'r', 'attempt': 1,
              'config_hash': 'b' * 16, 'state': 'queued', 'binding_state': 'bound',
              'effective_action': ACTION, 'idempotent_replay': False}
OBSERVATION = {**VERSION, **EXPERIMENT_METADATA, 'observation_version': 'agent-observation-v1',
               'session_id': 's', 'run_id': 'r', 'attempt': 1, 'state': 'queued',
               'effective_action': ACTION, 'selection_metric': 'macro_f1', 'progress': {},
               'validation': {'status': 'pending', 'metrics': {}}, 'validation_score': None,
               'remaining_runs': 0, 'allowed_actions': ['observe_ml_experiment', 'inspect_ml_session'],
               'error': None, 'extensions': {}, 'retry_after_seconds': 2}
FINALIZED = {**VERSION, 'session_id': 's', 'state': 'finalized', 'selected_run_id': 'r',
             'experiments_locked': True, 'final_result_url': '/#/results?run_id=r',
             'finalized_at': STAMP}


class Response:
    def __init__(self, status, body):
        self.status_code, self.body = status, copy.deepcopy(body)
    def json(self):
        return self.body


class FakeTransport:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []
    def request(self, method, url, *, headers, json, timeout):
        self.calls.append((method, url, headers, copy.deepcopy(json), timeout))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_all_six_tools_use_same_dispatcher_and_finite_endpoints():
    transport = FakeTransport([Response(200, payload) for payload in
                               (HEALTH, CREATED, SESSION, EXPERIMENT, OBSERVATION, FINALIZED)])
    dispatcher = ToolDispatcher(AutoAIClient('http://localhost:8000', transport=transport))
    dispatcher.dispatch('inspect_ml_capabilities', {})
    dispatcher.dispatch('start_ml_session', {'dataset_id': 'ds', 'selection_metric': 'macro_f1',
                                             'allowed_models': ['logistic_regression'], 'max_runs': 1})
    dispatcher.dispatch('inspect_ml_session', {'session_id': 's'})
    dispatcher.dispatch('submit_ml_experiment', {'session_id': 's', 'model_type': 'logistic_regression'})
    dispatcher.dispatch('observe_ml_experiment', {'session_id': 's', 'run_id': 'r'})
    dispatcher.dispatch('finalize_ml_session', {'session_id': 's', 'selected_run_id': 'r'})
    assert len(transport.calls) == 6


def test_token_is_absent_from_repr_and_formatted_exception_chain():
    sentinel = 'SYNTHETIC_SECRET_DO_NOT_STORE'
    transport = FakeTransport([ConnectionError(sentinel)])
    client = AutoAIClient('http://localhost', token=sentinel, transport=transport, max_retries=0)
    with pytest.raises(AgentConnectionError) as info:
        client.inspect_ml_capabilities()
    assert sentinel not in repr(client) + repr(info.value) + ''.join(traceback.format_exception(info.value))


def test_post_without_idempotency_key_is_not_retried():
    transport = FakeTransport([ConnectionError('down'), Response(200, EXPERIMENT)])
    with pytest.raises(AgentConnectionError):
        AutoAIClient('http://localhost', transport=transport).submit_ml_experiment('s', model_type='logistic_regression')
    assert len(transport.calls) == 1


def test_post_with_idempotency_key_replays_identical_payload_to_fixed_limit():
    transport = FakeTransport([ConnectionError('down'), Response(200, EXPERIMENT)])
    result = AutoAIClient('http://localhost', transport=transport, max_retries=2).submit_ml_experiment(
        's', model_type='logistic_regression', rationale='Stable reason', client_request_id='stable-1')
    assert result['run_id'] == 'r'
    assert len(transport.calls) == 2
    assert transport.calls[0][3] == transport.calls[1][3]


@pytest.mark.parametrize('payload', [{**HEALTH, 'contract_version': 'future'}, VERSION,
                                      {**HEALTH, 'unexpected': 'SYNTHETIC_SECRET'}])
def test_invalid_health_fails_closed(payload):
    client = AutoAIClient('http://localhost', transport=FakeTransport([Response(200, payload)]))
    with pytest.raises(AgentContractError):
        client.inspect_ml_capabilities()


@pytest.mark.parametrize('status', [409, 422, 503])
def test_domain_error_metadata_is_preserved_but_remote_text_is_not(status):
    sentinel = 'SYNTHETIC_SECRET'
    client = AutoAIClient('http://localhost', transport=FakeTransport([Response(status, {
        'detail': {'code': 'agent_invalid_action', 'message': sentinel, 'retryable': status == 503,
                   'allowed_actions': ['inspect_ml_session']}})]))
    with pytest.raises(AgentHTTPError) as info:
        client.inspect_ml_capabilities()
    assert info.value.code == 'agent_invalid_action'
    assert info.value.allowed_actions == ['inspect_ml_session']
    assert sentinel not in repr(info.value) + ''.join(traceback.format_exception(info.value))


@pytest.mark.parametrize('detail', [{}, {'code': 'bad', 'message': 'secret', 'retryable': 'yes',
                                       'allowed_actions': []},
    {'code': 'bad', 'message': 'secret', 'retryable': False, 'allowed_actions': ['shell']},
    {'code': 'bad', 'message': 'secret', 'retryable': False, 'allowed_actions': [], 'raw': 'secret'}])
def test_invalid_errors_fail_closed(detail):
    client = AutoAIClient('http://localhost', transport=FakeTransport([Response(503, {'detail': detail})]))
    with pytest.raises(AgentContractError):
        client.inspect_ml_capabilities()


@pytest.mark.parametrize('url', ['http://user:SYNTHETIC_SECRET@host', 'http://host/?SYNTHETIC_SECRET',
                                 'http://host/#SYNTHETIC_SECRET', 'http://host/%2fsecret',
                                 'http://host/../secret', 'http://host\\secret'])
def test_base_url_rejects_secret_bearing_or_ambiguous_addresses(url):
    with pytest.raises(ValueError) as info:
        AutoAIClient(url)
    assert 'SYNTHETIC_SECRET' not in repr(info.value) + ''.join(traceback.format_exception(info.value))


@pytest.mark.parametrize('identifier', ['..', '../r', 'r/child', '%2f', '%252f', 'r?key=secret',
                                       'r#secret', 'r\\child', 'r\nsecret'])
def test_ids_fail_before_network(identifier):
    transport = FakeTransport([])
    client = AutoAIClient('http://localhost', transport=transport)
    with pytest.raises(AgentContractError):
        client.observe_ml_experiment('s', identifier)
    with pytest.raises(AgentContractError):
        client.submit_ml_experiment('s', model_type='svm', client_request_id=identifier)
    assert not transport.calls


@pytest.mark.parametrize('change', [
    {'observation_version': 'future'}, {'run_id': 'other'},
    {'validation': {'status': 'pending', 'metrics': {'macro_f1': 0.9}}},
    {'validation': {'status': 'ready', 'metrics': {'macro_f1': float('nan')}}},
    {'validation': {'status': 'pending', 'metrics': {'hidden_metric': 0.5}}},
    {'extensions': {'raw': 'SYNTHETIC_SECRET'}},
    {'effective_config': {**EXPERIMENT_METADATA['effective_config'], 'path': 'SYNTHETIC_SECRET'}},
])
def test_observation_versions_shape_ids_and_finite_metrics_are_checked(change):
    client = AutoAIClient('http://localhost', transport=FakeTransport([Response(200, {**OBSERVATION, **change})]))
    with pytest.raises(AgentContractError):
        client.observe_ml_experiment('s', 'r')


def test_observation_free_text_is_projected_out():
    sentinel = 'SYNTHETIC_SECRET'
    payload = {**OBSERVATION, 'progress': {'message': sentinel, 'stage': sentinel, 'percent': 2.0},
               'error': {'code': sentinel, 'message': sentinel, 'retryable': False}}
    result = AutoAIClient('http://localhost', transport=FakeTransport([Response(200, payload)])).observe_ml_experiment('s', 'r')
    assert sentinel not in json.dumps(result)
    assert result['progress'] == {'percent': 2.0}


def test_reconcile_is_only_controlled_empty_body_and_never_dispatchable():
    transport = FakeTransport([Response(200, {**VERSION, 'session_id': 's', 'status': 'unchanged',
        'summary': {'examined': 0, 'recovered_bound': 0, 'released': 0, 'unchanged': 0, 'manual_review': 0},
        'items': []})])
    client = AutoAIClient('http://localhost', transport=transport)
    client.reconcile_ml_session('s')
    assert transport.calls[0][3] == {}
    assert 'reconcile_ml_session' not in TOOL_SCHEMAS
    with pytest.raises(AgentContractError):
        ToolDispatcher(client).dispatch('reconcile_ml_session', {'session_id': 's'})


def test_successful_zero_score_is_preserved_as_a_valid_measurement():
    payload = {**OBSERVATION, 'state': 'succeeded',
               'validation': {'status': 'ready', 'metrics': {'macro_f1': 0.0}},
               'validation_score': 0.0, 'allowed_actions': ['finalize_ml_session']}
    result = AutoAIClient('http://localhost', transport=FakeTransport([Response(200, payload)])).observe_ml_experiment('s', 'r')
    assert result['validation_score'] == 0.0
    assert result['validation']['status'] == 'ready'


@pytest.mark.parametrize('score', [0.1, None, True, float('inf')])
def test_successful_observation_cannot_misrepresent_selection_score(score):
    payload = {**OBSERVATION, 'state': 'succeeded',
               'validation': {'status': 'ready', 'metrics': {'macro_f1': 0.0}},
               'validation_score': score, 'allowed_actions': ['finalize_ml_session']}
    with pytest.raises(AgentContractError):
        AutoAIClient('http://localhost', transport=FakeTransport([Response(200, payload)])).observe_ml_experiment('s', 'r')


def test_ready_metrics_without_selection_metric_remain_explicitly_unselected():
    payload = {**OBSERVATION, 'state': 'succeeded',
               'validation': {'status': 'ready', 'metrics': {'accuracy': 0.9}},
               'validation_score': None, 'allowed_actions': ['finalize_ml_session']}
    result = AutoAIClient('http://localhost', transport=FakeTransport([Response(200, payload)])).observe_ml_experiment('s', 'r')
    assert result['validation_score'] is None


def test_malicious_capability_model_never_crosses_client_boundary():
    payload = {**HEALTH, 'models': ['logistic_regression', 'C:/private/SYNTHETIC_SECRET']}
    with pytest.raises(AgentContractError) as info:
        AutoAIClient('http://localhost', transport=FakeTransport([Response(200, payload)])).inspect_ml_capabilities()
    assert 'SYNTHETIC_SECRET' not in ''.join(traceback.format_exception(info.value))


def test_backend_module_prose_is_not_returned_as_evidence():
    payload = {**HEALTH, 'modules': {'evidence_card': {'available': False, 'status': 'unavailable',
               'schema_version': None, 'reason': 'SYNTHETIC_SECRET'}}}
    result = AutoAIClient('http://localhost', transport=FakeTransport([Response(200, payload)])).inspect_ml_capabilities()
    assert 'SYNTHETIC_SECRET' not in json.dumps(result)

"""Observable persistence, immutability, replay and safety state guarantees."""

from __future__ import annotations

import copy
import json

import pytest
from pydantic import BaseModel, ValidationError

from agent_poc.orchestration.state import (
    ExperimentRequest, GraphState, PendingOperation, StateModel, apply_patch,
    fingerprint, new_state,
)


@pytest.fixture
def state():
    return new_state(dataset_id='dataset-1', allowed_models=['svm', 'cnn1d'],
                     backend_fingerprint='a' * 64, principal_fingerprint='b' * 64,
                     llm_config_fingerprint='c' * 64, now=1000.0,
                     task_id='task-1', thread_id='thread-1', timeout_seconds=300.0)


def assert_json_primitives(value):
    if isinstance(value, dict):
        assert all(type(key) is str for key in value)
        for item in value.values():
            assert_json_primitives(item)
    elif isinstance(value, list):
        for item in value:
            assert_json_primitives(item)
    else:
        assert type(value) in (str, int, float, bool, type(None))


def test_complete_typed_empty_state_round_trips_json(state):
    expected = {'identity', 'lifecycle', 'task', 'versions', 'module_policy', 'capabilities',
                'evidence', 'recipes', 'knowledge', 'memory', 'decision', 'execution',
                'budget', 'feedback', 'guard', 'diagnosis', 'replanning', 'candidates',
                'history', 'recovery', 'finalization'}
    assert set(state) == expected == set(StateModel.model_fields)
    assert set(GraphState.__annotations__) == expected | {'search_plans'}
    parsed = StateModel.model_validate(state)
    assert all(isinstance(getattr(parsed, key), BaseModel) for key in expected)
    assert all(getattr(parsed, key).model_config['extra'] == 'forbid' for key in expected)
    serialized = json.dumps(state, allow_nan=False)
    assert StateModel.model_validate_json(serialized).model_dump(mode='json') == state
    assert_json_primitives(state)
    for name in ('evidence', 'recipes', 'knowledge', 'memory', 'diagnosis', 'replanning'):
        assert state[name]['status'] == 'disabled'
        assert state[name]['implementation_status'] == 'unavailable'
    assert state['evidence']['statistics']['observation_count'] is None
    assert state['budget']['runs']['actual'] is None
    assert state['budget']['input_tokens']['actual'] is None
    assert state['feedback']['selection_score'] is None
    assert state['finalization']['locked_at'] is None
    assert state['task']['split_fingerprint'] is None
    assert state['task']['split_fingerprint_status'] == 'unavailable'
    assert 'cnn1d' in state['task']['allowed_models']


@pytest.mark.parametrize('block', list(StateModel.model_fields))
def test_each_block_rejects_unknown_fields(state, block):
    state[block]['raw_payload'] = {'private': 'not-allowed'}
    with pytest.raises(ValidationError):
        StateModel.model_validate(state)


@pytest.mark.parametrize('patch', [
    {'task': {'seed': 17}},
    {'task': {'dataset_id': 'another-dataset'}},
    {'task': {'allowed_models': ['svm']}},
    {'versions': {'prompt': 'changed-prompt'}},
    {'versions': {'llm_config_fingerprint': 'd' * 64}},
    {'module_policy': {'ablation_id': 'changed-ablation'}},
    {'budget': {'llm_calls': {'limit': 90}}},
    {'budget': {'deadline_at': 8000.0}},
    {'budget': {'max_operation_attempts': 99}},
    {'lifecycle': {'started_at': 2000.0}},
    {'identity': {'thread_id': 'other-thread'}},
])
def test_frozen_configuration_rejects_changes(state, patch):
    original = copy.deepcopy(state)
    with pytest.raises((ValueError, ValidationError)):
        apply_patch(state, patch)
    assert state == original


def test_startup_summary_detects_checkpoint_configuration_tampering(state):
    state['task']['seed'] = 43
    with pytest.raises(ValidationError, match='fingerprint mismatch'):
        StateModel.model_validate(state)


@pytest.mark.parametrize('block,field,unknown', [
    ('versions', 'state', 'agent-state-v99'),
    ('versions', 'graph', 'agent-graph-v99'),
    ('versions', 'api', 'agent-session-v99'),
    ('versions', 'observation', 'agent-observation-v99'),
    ('versions', 'metadata', 'agent-metadata-v99'),
    ('feedback', 'observation_version', 'agent-observation-v99'),
    ('execution', 'metadata_version', 'agent-metadata-v99'),
])
def test_unknown_protocol_versions_fail_closed(state, block, field, unknown):
    state[block][field] = unknown
    with pytest.raises(ValidationError):
        StateModel.model_validate(state)


def test_backend_dataset_fingerprint_binds_once_without_conflating_split(state):
    bound = apply_patch(state, {'task': {'dataset_fingerprint': 'd' * 64,
                                        'dataset_fingerprint_status': 'ready'}})
    assert bound['task']['split_fingerprint'] is None
    assert bound['task']['evaluation_config_fingerprint'] != bound['task']['dataset_fingerprint']
    assert apply_patch(bound, {'task': {'dataset_fingerprint': 'd' * 64}}) == bound
    with pytest.raises(ValueError, match='frozen'):
        apply_patch(bound, {'task': {'dataset_fingerprint': 'e' * 64}})
    with pytest.raises(ValueError):
        apply_patch(bound, {'execution': {'dataset_fingerprint': 'e' * 64,
                                          'dataset_fingerprint_status': 'ready'}})


def test_partial_patch_preserves_sibling_fields_and_replays_event_once(state):
    event = {'event_id': 'event-1', 'kind': 'session_prepared', 'sequence': 1,
             'occurred_at': 1001.0, 'operation_id': 'operation-1'}
    updated = apply_patch(state, {'lifecycle': {'stage': 'session', 'next_action': 'session'},
                                  'history': {'events': [event]}})
    assert updated['lifecycle']['started_at'] == 1000.0
    replayed = apply_patch(updated, {'history': {'events': [updated['history']['events'][0]]}})
    assert replayed == updated
    assert len(replayed['history']['events']) == 1
    changed = {**updated['history']['events'][0], 'kind': 'different_event'}
    with pytest.raises(ValueError, match='conflicting content'):
        apply_patch(updated, {'history': {'events': [changed]}})
    with pytest.raises(ValueError, match='logical sequence'):
        apply_patch(updated, {'history': {'events': [{**event, 'event_id': 'event-2'}]}})


def test_durable_network_and_repair_limits_do_not_reset_after_serialization(state):
    used = apply_patch(state, {'recovery': {'attempts': [
        {'operation_id': 'choose-1', 'network_attempts': 2, 'repair_attempts': 1} ]},
        'budget': {'llm_calls': {'actual': 2.0, 'remaining': 4.0}},
        'lifecycle': {'status': 'waiting', 'next_action': 'observe'}})
    restored = StateModel.model_validate_json(json.dumps(used)).model_dump(mode='json')
    assert restored['budget']['deadline_at'] == 1300.0
    assert restored['recovery']['attempts'][0]['network_attempts'] == 2
    increased = apply_patch(restored, {'recovery': {'attempts': [
        {'operation_id': 'choose-1', 'repair_attempts': 2}]}})
    assert increased['recovery']['attempts'][0]['network_attempts'] == 2
    with pytest.raises(ValueError, match='cannot decrease'):
        apply_patch(increased, {'recovery': {'attempts': [
            {'operation_id': 'choose-1', 'repair_attempts': 0}]}})
    with pytest.raises(ValueError, match='cannot decrease'):
        apply_patch(restored, {'budget': {'llm_calls': {'actual': 0.0}}})


def test_usage_unknown_is_not_zero_and_settlement_is_idempotent(state):
    started = apply_patch(state, {'budget': {'usage': [
        {'usage_id': 'usage-1', 'operation_id': 'choose-1', 'kind': 'llm',
         'status': 'pending', 'token_status': 'pending'}],
        'input_tokens': {'actual': None, 'unknown_pending': 1}}})
    unknown = apply_patch(started, {'budget': {'usage': [
        {'usage_id': 'usage-1', 'status': 'unknown', 'token_status': 'unknown'}]}})
    assert unknown['budget']['usage'][0]['input_tokens'] is None
    restored = StateModel.model_validate_json(json.dumps(unknown)).model_dump(mode='json')
    settled = apply_patch(restored, {'budget': {'usage': [
        {'usage_id': 'usage-1', 'status': 'confirmed', 'token_status': 'known',
         'input_tokens': 12, 'output_tokens': 5}],
        'input_tokens': {'actual': 12.0, 'unknown_pending': 0, 'measurement_status': 'ready'}}})
    assert len(settled['budget']['usage']) == 1
    assert apply_patch(settled, {'budget': {'usage': settled['budget']['usage']}}) == settled
    with pytest.raises(ValueError, match='cannot be rewritten'):
        apply_patch(settled, {'budget': {'usage': [{'usage_id': 'usage-1', 'input_tokens': 99}]}})


def prepared_experiment():
    content = ExperimentRequest(session_id='session-1', model_type='svm',
                                rationale='Validation evidence supports this candidate.',
                                client_request_id='request-1')
    return PendingOperation(operation_id='operation-1', kind='experiment', request_id='request-1',
                            tool_name='submit_ml_experiment', content=content,
                            content_fingerprint=fingerprint(content.model_dump(mode='json')))


def test_full_request_content_and_id_cannot_change_during_replay(state):
    operation = prepared_experiment().model_dump(mode='json')
    updated = apply_patch(state, {'identity': {'session_id': 'session-1'},
                                  'recovery': {'pending_operation': operation}})
    assert apply_patch(updated, {'recovery': {'pending_operation': {'status': 'unknown',
        'operation_id': 'operation-1'}}})['recovery']['pending_operation']['content'] == operation['content']
    altered = copy.deepcopy(operation)
    altered['content']['rationale'] = 'A different rationale.'
    altered['content_fingerprint'] = fingerprint(altered['content'])
    with pytest.raises(ValueError, match='immutable'):
        apply_patch(updated, {'recovery': {'pending_operation': altered}})
    serialized = json.dumps(updated).lower()
    assert 'split_test' not in serialized
    assert 'prediction' not in serialized
    assert 'train_weight' in serialized


def test_different_operation_replaces_discriminated_request_without_stale_fields(state):
    first = apply_patch(state, {'recovery': {'pending_operation': prepared_experiment().model_dump(mode='json')}})
    content = {'session_id': 'session-1', 'selected_run_id': 'run-1'}
    second = apply_patch(first, {'recovery': {'pending_operation': {
        'operation_id': 'finalize-1', 'kind': 'finalize', 'tool_name': 'finalize_ml_session',
        'content': content, 'content_fingerprint': fingerprint(content)}}})
    assert second['recovery']['pending_operation']['content'] == content


@pytest.mark.parametrize('patch', [
    {'module_policy': {'evidence_card': {'enabled': True}}},
    {'module_policy': {'context_policy': {'case_read': True}}},
    {'evidence': {'status': 'ready'}},
    {'evidence': {'statistics': {'observation_count': 20}}},
    {'memory': {'snapshot_version': 'invented-v1'}},
    {'diagnosis': {'problem_type': 'overfitting'}},
    {'replanning': {'used_count': 1}},
])
def test_unavailable_modules_cannot_be_enabled_or_given_fake_outputs(state, patch):
    with pytest.raises(ValueError):
        apply_patch(state, patch)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), True, '0.9'])
def test_invalid_metric_values_are_rejected(state, value):
    with pytest.raises(ValidationError):
        apply_patch(state, {'feedback': {'validation_metrics': {'macro_f1': value}}})


@pytest.mark.parametrize('text', ['C:\\server\\private.csv', '/srv/data/private.csv',
                                 'Bearer SYNTHETIC_SECRET', 'token=SYNTHETIC_SECRET',
                                 'Traceback: SYNTHETIC_SECRET', 'Test accuracy 0.99',
                                 'prediction values', 'https://private.example/data'])
def test_unsafe_text_never_enters_durable_state(state, text):
    with pytest.raises(ValidationError) as caught:
        apply_patch(state, {'decision': {'rationale': text}})
    assert 'SYNTHETIC_SECRET' not in str(caught.value)
    assert 'SYNTHETIC_SECRET' not in repr(caught.value)


def test_backend_metadata_internal_fields_fail_closed(state):
    with pytest.raises(ValidationError):
        apply_patch(state, {'task': {'evaluation_config': {'split_test': 1}}})
    with pytest.raises(ValidationError):
        apply_patch(state, {'feedback': {'validation_metrics': {'pooled_test': 0.99}}})


def test_candidate_eligibility_uses_capabilities_and_finite_metric_not_quality_threshold(state):
    ready = apply_patch(state, {
        'identity': {'session_id': 'session-1'},
        'execution': {'experiment_id': 'experiment-1', 'run_id': 'run-1', 'run_status': 'succeeded'},
        'capabilities': {'status': 'ready', 'models': [{'model_type': 'svm', 'available': True}],
                         'eligible_models': ['svm']},
        'feedback': {'status': 'ready', 'run_status': 'succeeded', 'validation_status': 'ready',
                     'integrity': 'ready', 'validation_metrics': {'macro_f1': 0.0}, 'selection_score': 0.0},
        'candidates': {'status': 'ready', 'items': [{'candidate_id': 'candidate-1',
            'session_id': 'session-1', 'run_id': 'run-1', 'experiment_id': 'experiment-1',
            'status': 'valid', 'validation_metrics': {'macro_f1': 0.0}, 'selection_score': 0.0}]}})
    assert ready['candidates']['items'][0]['selection_score'] == 0.0
    assert ready['candidates']['items'][0]['uncertainty']['estimate'] is None
    with pytest.raises(ValueError, match='capability'):
        apply_patch(ready, {'capabilities': {'eligible_models': ['svm', 'cnn1d']}})


def test_finalize_requires_matching_backend_confirmation_and_never_invents_lock(state):
    lock = {'selected_run_id': 'run-1', 'rationale': 'The valid candidate was selected.',
            'status': 'confirmed', 'backend_session_state': 'finalized', 'locked_at': 1050.0}
    with pytest.raises(ValueError, match='backend inspection'):
        apply_patch(state, {'finalization': lock})
    confirmed = apply_patch(state, {'finalization': lock, 'recovery': {'last_confirmed_backend_state': {
        'session_state': 'finalized', 'selected_run_id': 'run-1', 'confirmed_at': 1050.0}}})
    assert confirmed['finalization']['status'] == 'confirmed'
    with pytest.raises(ValueError):
        apply_patch(confirmed, {'finalization': {'selected_run_id': 'run-2'}})
    with pytest.raises(ValueError):
        apply_patch(state, {'finalization': {'status': 'unselected', 'locked_at': 1050.0}})


def test_no_candidate_failure_keeps_backend_session_open(state):
    stopped = apply_patch(state, {'lifecycle': {'status': 'failed', 'stage': 'ended',
                                               'next_action': None, 'reason_code': 'no_valid_candidate',
                                               'ended_at': 1100.0},
                                  'finalization': {'status': 'unselected',
                                                   'backend_session_state': 'open',
                                                   'termination_reason': 'no_valid_candidate'}})
    assert stopped['finalization']['selected_run_id'] is None
    assert stopped['finalization']['locked_at'] is None
    assert stopped['finalization']['backend_session_state'] == 'open'

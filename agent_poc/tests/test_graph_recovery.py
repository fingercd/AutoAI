"""Recovery decisions through the validated HTTP client and reopened SQLite runner.

The HTTP service and model in these tests are explicitly simulated; the separate
integration suite covers actual API, worker and subprocess crash behavior.
"""

from copy import deepcopy
from urllib.parse import urlsplit

import httpx
import pytest

from agent_poc.clients.autoai_client import AutoAIClient
from agent_poc.orchestration.runtime import resume_task, start_task
from agent_poc.tests.test_graph import Clock, Model, configuration


class RecoveryTransport:
    def __init__(self, scenario):
        self.scenario = scenario
        self.calls = []
        self.session = None
        self.action = None
        self.binding = None
        self.reconciliations = 0
        self.submissions = []
        self.timestamp = '2026-09-12T00:00:00+00:00'

    def metadata(self, bound):
        return {
            'metadata_version': 'agent-metadata-v1',
            'dataset_sha256': 'a' * 64 if bound else None,
            'dataset_fingerprint_status': 'ready' if bound else 'unavailable',
            'effective_config_status': 'ready' if bound else 'pending',
            'effective_config': {
                **{key: self.action[key] for key in ('model_type', 'normalization', 'class_balance')},
                'seed': 42, 'feature_selection_enabled': False,
                'evaluation_config': {'mode': 'stratified_holdout', 'train_weight': 8,
                                      'validation_weight': 1, 'heldout_weight': 1},
            } if bound else None,
        }

    def session_response(self):
        bound = self.binding == 'bound'
        response = {
            'contract_version': 'agent-session-v1', 'session_id': 'session-a', 'state': 'open',
            'locked_config': {
                **{key: self.session[key] for key in ('dataset_id', 'selection_metric', 'allowed_models',
                   'max_runs', 'seed', 'modules', 'context_policy')},
                'evaluation_config': {'mode': 'stratified_holdout', 'train_weight': 8,
                                      'validation_weight': 1, 'heldout_weight': 1},
                'metadata_version': 'agent-metadata-v1', 'dataset_sha256': 'a' * 64,
                'dataset_fingerprint_status': 'ready',
            },
            'remaining_runs': 1 if self.binding in (None, 'released') else 0,
            'best_run_id': None, 'selected_run_id': None, 'created_at': self.timestamp,
            'finalized_at': None, 'experiments': [],
        }
        if self.binding is not None:
            response['experiments'].append({
                'attempt': 1, 'run_id': 'run-a' if bound else None, 'binding_state': self.binding,
                'parent_run_id': None, 'effective_action': self.action, 'config_hash': 'b' * 16,
                'failure_code': None, 'created_at': self.timestamp,
                'state': 'failed' if bound else 'binding_failed', 'validation_score': None,
                **self.metadata(bound),
            })
        return response

    def request(self, method, url, *, headers, json, timeout):
        path = urlsplit(url).path
        self.calls.append((method, path, deepcopy(json)))
        if path.endswith('/health'):
            return httpx.Response(200, json={
                'contract_version': 'agent-session-v1', 'status': 'ready',
                'capabilities': dict.fromkeys(('create_session', 'create_experiment', 'read_session',
                                              'read_feedback', 'finalize_session'), True),
                'models': ['logistic_regression', 'svm', 'random_forest'], 'modules': {},
            })
        if path == '/api/agent/sessions':
            self.session = deepcopy(json)
            response = self.session_response()
            for key in ('best_run_id', 'selected_run_id', 'finalized_at', 'experiments'):
                response.pop(key)
            response['idempotent_replay'] = False
            return httpx.Response(201, json=response)
        if path.endswith('/experiments'):
            self.submissions.append(deepcopy(json))
            self.action = {key: json[key] for key in ('model_type', 'normalization', 'class_balance')}
            self.action['parent_run_id'] = None
            self.binding = ('released' if self.scenario == 'already_released' else
                            'compensation_required' if self.scenario.startswith('compensation') else 'reserved')
            raise TimeoutError('Synthetic lost POST response')
        if path.endswith('/reconcile'):
            assert method == 'POST' and json == {}, 'Recovery must use an empty request body.'
            self.reconciliations += 1
            if self.scenario == 'storage_once' and self.reconciliations == 1:
                return httpx.Response(503, json={'detail': {
                    'code': 'agent_reconciliation_unavailable', 'message': 'temporarily unavailable',
                    'retryable': True, 'allowed_actions': ['inspect_ml_session'],
                }})
            if self.scenario in ('fresh_release', 'always_fresh') and (
                    self.reconciliations == 1 or self.scenario == 'always_fresh'):
                code = 'reservation_fresh'
            elif self.scenario == 'compensation_running':
                code = 'run_still_active'
            elif self.scenario in ('mapped_run', 'compensation_terminal', 'storage_once'):
                self.binding = 'bound'
                code = 'recovered_terminal_binding' if self.scenario == 'compensation_terminal' else 'recovered_binding'
            else:
                self.binding = 'released'
                code = 'stale_without_run'
            manual = self.scenario == 'compensation_running'
            return httpx.Response(200, json={
                'contract_version': 'agent-session-v1', 'session_id': 'session-a',
                'status': ('manual_review_required' if manual else 'reconciled'
                           if self.binding in ('bound', 'released') else 'unchanged'),
                'summary': {'examined': 1, 'recovered_bound': int(self.binding == 'bound'),
                            'released': int(self.binding == 'released'),
                            'unchanged': int(self.binding in ('reserved', 'compensation_required')),
                            'manual_review': int(manual)},
                'items': [{'attempt': 1, 'state': self.binding, 'resolution_code': code,
                           'requires_manual_review': manual}],
            })
        if path.endswith('/feedback'):
            return httpx.Response(200, json={
                'contract_version': 'agent-session-v1', 'observation_version': 'agent-observation-v1',
                'session_id': 'session-a', 'run_id': 'run-a', 'attempt': 1, 'state': 'failed',
                'effective_action': self.action, **self.metadata(True), 'selection_metric': 'macro_f1',
                'progress': {}, 'validation': {'status': 'failed', 'metrics': {}},
                'validation_score': None, 'remaining_runs': 0,
                'allowed_actions': ['inspect_ml_session'],
                'error': {'code': 'agent_run_failed', 'message': 'Synthetic failed Run', 'retryable': False},
                'extensions': {},
            })
        if method == 'GET' and path == '/api/agent/sessions/session-a':
            return httpx.Response(200, json=self.session_response())
        raise AssertionError('unexpected request')


def _start_and_reopen(tmp_path, scenario, **limits):
    transport, model, clock = RecoveryTransport(scenario), Model(), Clock()
    client = AutoAIClient('http://backend.local', transport=transport, max_retries=0)
    first = start_task(configuration(), dataset_id='ds-a', allowed_models=['svm'],
        storage=tmp_path, thread_id='recovery-thread', client=client, llm=model,
        clock=clock, sleep=clock.sleep, wait=False, **limits)
    assert first['lifecycle']['next_action'] == 'inspect_session', first['lifecycle']
    assert first['execution']['submission_content'] is not None
    clock.sleep(2)
    reopened = AutoAIClient('http://backend.local', transport=transport, max_retries=0)
    final = resume_task(configuration(), storage=tmp_path, thread_id='recovery-thread',
        client=reopened, llm=model, clock=clock, sleep=clock.sleep, wait=True)
    assert len(transport.submissions) == 1
    assert len(model.calls) == 1
    assert first['execution']['submission_content'] == final['execution']['submission_content']
    assert first['identity']['session_id'] == final['identity']['session_id']
    calls = list(transport.calls)
    terminal = resume_task(configuration(), storage=tmp_path, thread_id='recovery-thread',
        client=reopened, llm=model, clock=clock, sleep=clock.sleep, wait=True)
    assert terminal == final and transport.calls == calls
    return final, transport


@pytest.mark.parametrize('scenario', ['already_released', 'stale_no_mapping', 'fresh_release'])
def test_released_attempt_never_submits_a_new_scientific_experiment(tmp_path, scenario):
    state, transport = _start_and_reopen(tmp_path, scenario)
    assert state['lifecycle']['reason_code'] == 'agent_request_released', state['lifecycle']
    assert state['finalization']['status'] == 'unselected'
    assert state['finalization']['selected_run_id'] is None
    assert state['finalization']['backend_session_state'] == 'open'
    assert transport.reconciliations == (0 if scenario == 'already_released' else
                                        2 if scenario == 'fresh_release' else 1)


@pytest.mark.parametrize('scenario', ['mapped_run', 'compensation_terminal', 'storage_once'])
def test_reconcile_recovers_original_run_and_observes_failed_no_candidate(tmp_path, scenario):
    state, transport = _start_and_reopen(tmp_path, scenario)
    assert state['execution']['run_id'] == 'run-a', state['lifecycle']
    assert state['lifecycle']['reason_code'] == 'run_failed'
    assert state['finalization']['status'] == 'unselected'
    assert state['recovery']['reconciliation']['outcome'] == 'bound'
    assert transport.reconciliations == (2 if scenario == 'storage_once' else 1)


def test_compensation_running_requires_attention_and_does_not_finalize(tmp_path):
    state, transport = _start_and_reopen(tmp_path, 'compensation_running')
    assert state['lifecycle']['status'] == 'needs_attention', state['lifecycle']
    assert state['lifecycle']['reason_code'] == 'run_still_active'
    assert state['recovery']['needs_human_review']
    assert state['budget']['runs']['remaining'] == 0
    assert transport.reconciliations == 1


def test_fresh_reservation_waits_with_durable_finite_attempt_limit(tmp_path):
    state, transport = _start_and_reopen(tmp_path, 'always_fresh', max_operation_attempts=2)
    assert state['lifecycle']['reason_code'] == 'operation_attempt_limit', state['lifecycle']
    assert state['recovery']['needs_human_review']
    assert transport.reconciliations == 2
    assert state['recovery']['pending_operation'] is not None

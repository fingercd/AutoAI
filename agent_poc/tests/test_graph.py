"""Behavioral node tests; real HTTP/worker integration lives alongside these."""
import copy
from pathlib import Path

import pytest

from agent_poc.orchestration.llm import LLMConfig, LLMError, Proposal, TokenUsage
from agent_poc.orchestration.runtime import RuntimeConfig, start_task, resume_task
from agent_poc.clients.autoai_client import AgentConnectionError


class Clock:
    value = 1000.0
    def __call__(self): return self.value
    def sleep(self, seconds): self.value += seconds


class Model:
    def __init__(self, failures=0): self.calls, self.failures = [], failures
    def propose(self, phase, context):
        self.calls.append((phase, copy.deepcopy(context)))
        if len(self.calls) <= self.failures:
            raise LLMError('llm_output_invalid')
        args = dict(context['bindings'])
        if phase == 'submit':
            args.update(model_type='svm', rationale='Choose the permitted margin classifier.')
        return Proposal('submit_ml_experiment' if phase == 'submit' else 'finalize_ml_session',
                        args, 'Choose the permitted margin classifier.' if phase == 'submit'
                        else 'The completed candidate has valid evidence.',
                        usage=TokenUsage(10, 5, 15, 'known'))


class Backend:
    max_retries = 0
    def __init__(self, terminal='succeeded', valid=True, score=0.0):
        self.calls = []
        self.session = None
        self.experiment = None
        self.terminal, self.valid, self.score = terminal, valid, score
        self.finalized = False
        self.fail_on = None
        self.pending_observations = 0
    def mark(self, name):
        self.calls.append(name)
        if self.fail_on == name:
            raise AgentConnectionError('unavailable')
    def inspect_ml_capabilities(self):
        self.mark('capabilities')
        return {'models':['logistic_regression','svm','random_forest'], 'modules':{},
                'capabilities':{'create_experiment':True}}
    def start_ml_session(self, **body):
        self.mark('session')
        self.session = self.session or body
        return self.session_response()
    def session_response(self):
        body = self.session
        locked = {key:body[key] for key in ('dataset_id','selection_metric','allowed_models','max_runs','seed','modules','context_policy')}
        locked.update(evaluation_config={'mode':'stratified_holdout','train_weight':8,'validation_weight':1,'heldout_weight':1},
                      metadata_version='agent-metadata-v1', dataset_sha256='a'*64, dataset_fingerprint_status='ready')
        return {'session_id':'session-a', 'locked_config':locked, 'remaining_runs':0 if self.experiment else 1,
                'state':'finalized' if self.finalized else 'open',
                'selected_run_id':'run-a' if self.finalized else None,
                'finalized_at':'2026-09-12T00:00:00+00:00' if self.finalized else None,
                'experiments':[{**self.execution_response(), 'binding_state':'bound'}] if self.experiment else []}
    def inspect_ml_session(self, session_id):
        self.mark('inspect')
        return self.session_response()
    def submit_ml_experiment(self, session_id, **body):
        self.mark('submit')
        if self.experiment:
            assert self.experiment == body
        self.experiment = body
        return self.execution_response('queued')
    def execution_response(self, status=None):
        action = {key:self.experiment[key] for key in ('model_type','normalization','class_balance')}
        return {'run_id':'run-a','state':status or self.terminal, 'effective_action':action,
                'effective_config':{**action,'seed':42,'feature_selection_enabled':False,
                    'evaluation_config':{'mode':'stratified_holdout','train_weight':8,'validation_weight':1,'heldout_weight':1}},
                'effective_config_status':'ready','dataset_sha256':'a'*64,'dataset_fingerprint_status':'ready'}
    def observe_ml_experiment(self, session_id, run_id):
        self.mark('observe')
        waiting = self.pending_observations > 0
        self.pending_observations -= 1
        status = 'running' if waiting else self.terminal
        valid = self.valid and status == 'succeeded'
        return {**self.execution_response(status), 'selection_metric':'macro_f1', 'remaining_runs':0,
                'progress':{}, 'validation':{'status':'ready' if valid else 'pending' if waiting else 'unavailable',
                    'metrics':{'macro_f1':self.score} if valid and self.score is not None else {}},
                'validation_score':self.score if valid else None, 'error':None,
                'allowed_actions':['finalize_ml_session'] if valid else ['observe_ml_experiment'],
                'retry_after_seconds':2 if waiting else None}
    def finalize_ml_session(self, session_id, selected_run_id):
        self.mark('finalize')
        self.finalized = True
        return {'ignored_human_link':'not forwarded'}


def configuration():
    return RuntimeConfig('http://backend.local', 'local', LLMConfig('http://model.local/v1', 'local-model'))


def run(tmp_path, *, backend=None, model=None, clock=None, **kwargs):
    backend, model, clock = backend or Backend(), model or Model(), clock or Clock()
    state = start_task(configuration(), dataset_id='ds-a', allowed_models=['svm','logistic_regression'],
        storage=tmp_path, thread_id='thread-a', client=backend, llm=model,
        clock=clock, sleep=clock.sleep, wait=True, **kwargs)
    return state, backend, model, clock


def test_success_low_score_keeps_real_model_decision_and_six_tools(tmp_path):
    state, backend, model, _ = run(tmp_path)
    assert state['lifecycle']['status'] == 'completed', state['lifecycle']
    assert state['execution']['effective_action']['model_type'] == 'svm'
    assert state['candidates']['items'][0]['selection_score'] == 0.0
    assert state['finalization']['status'] == 'confirmed'
    assert set(backend.calls) == {'capabilities','session','inspect','submit','observe','finalize'}
    assert len(model.calls) == 2
    assert state['budget']['llm_calls']['actual'] == 2
    assert state['budget']['input_tokens']['actual'] == 20
    assert state['budget']['runs']['remaining'] == 0
    assert len({e['event_id'] for e in state['history']['events']}) == len(state['history']['events'])


@pytest.mark.parametrize('terminal,valid,score,expected',[
    ('failed',False,None,'failed'), ('cancelled',False,None,'cancelled'),
    ('succeeded',False,None,'failed'), ('succeeded',True,None,'failed'),
])
def test_no_candidate_is_terminal_without_finalize(tmp_path,terminal,valid,score,expected):
    state, backend, model, clock = run(tmp_path,backend=Backend(terminal,valid,score))
    assert state['lifecycle']['status'] == expected, state['lifecycle']
    assert state['finalization']['selected_run_id'] is None
    assert state['finalization']['status'] == 'unselected'
    before = list(backend.calls)
    resumed = resume_task(configuration(), storage=tmp_path, thread_id='thread-a',client=backend,llm=model,clock=clock)
    assert resumed == state and backend.calls == before


def test_format_repairs_have_finite_durable_limit(tmp_path):
    state, backend, model, _ = run(tmp_path,model=Model(failures=100),max_repair_attempts=1)
    assert state['lifecycle']['reason_code'] == 'llm_repair_exhausted'
    assert len(model.calls) == 2
    assert backend.experiment is None


def test_network_failure_stops_at_operation_limit(tmp_path):
    backend = Backend(); backend.fail_on = 'session'
    state, backend, model, _ = run(tmp_path, backend=backend,max_operation_attempts=2)
    assert state['lifecycle']['reason_code'] == 'operation_attempt_limit', state['lifecycle']
    assert backend.calls.count('session') == 2
    assert len(model.calls) == 0


def test_deadline_preserves_running_fact_and_never_cancels(tmp_path):
    backend = Backend(); backend.pending_observations = 100
    state, backend, model, _ = run(tmp_path,backend=backend,timeout_seconds=3)
    assert state['lifecycle']['status'] == 'timed_out'
    assert state['execution']['run_status'] == 'running'
    assert state['recovery']['needs_human_review']
    assert not backend.finalized and len(model.calls) == 1


def test_maximum_public_task_identifier_has_valid_derived_operations(tmp_path):
    state, _, _, _ = run(tmp_path, task_id='a'*128)
    assert state['lifecycle']['status'] == 'completed'
    assert state['budget']['cached_tokens']['actual'] is None
    assert state['budget']['cached_tokens']['unknown_pending'] == 2

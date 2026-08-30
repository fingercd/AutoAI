from __future__ import annotations

import pytest

from agent_poc.orchestrator import AgentTimeout, run_agent
from agent_poc.schemas import FinalizeDecision, RequestHumanDecision, RunExperimentDecision
from agent_poc.state import AgentConfig


class FakeAutoAI:
    def __init__(self, feedback_states):
        self.feedback_states = list(feedback_states)
        self.feedback_calls = 0
        self.llm_poll_boundary = None
        self.experiments = []
        self.finalized = None

    def health(self):
        return {'status': 'ok', 'worker': {'available': True, 'compatible': True}}

    def create_session(self, payload):
        self.session_payload = payload
        return {'session_id': 'session-1', 'remaining_runs': payload['max_runs']}

    def get_session(self, _session_id):
        return {
            'session_id': 'session-1',
            'remaining_runs': max(0, self.session_payload['max_runs'] - len(self.experiments)),
            'experiments': list(self.experiments),
        }

    def create_experiment(self, _session_id, payload):
        run_id = f'run-{len(self.experiments) + 1}'
        self.experiments.append({
            'run_id': run_id,
            'state': 'queued',
            'effective_action': payload,
            'validation_score': None,
        })
        return {'run_id': run_id, 'state': 'queued', 'effective_action': payload}

    def feedback(self, _session_id, run_id):
        self.feedback_calls += 1
        state = self.feedback_states.pop(0)
        if state == 'succeeded':
            self.experiments[0]['state'] = 'succeeded'
            self.experiments[0]['validation_score'] = 0.8
            return {
                'run_id': run_id,
                'state': 'succeeded',
                'validation': {'status': 'ready', 'metrics': {'macro_f1': 0.8}},
            }
        return {
            'run_id': run_id,
            'state': state,
            'validation': {'status': 'pending', 'metrics': {}},
        }

    def finalize(self, _session_id, selected_run_id):
        self.finalized = selected_run_id
        return {'status': 'finalized', 'selected_run_id': selected_run_id}


class FakeLLM:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.call_count = 0

    def decide(self, _messages):
        self.call_count += 1
        return next(self.decisions)


def _cfg(**overrides):
    values = {
        'autoai_base_url': 'http://autoai',
        'dataset_id': 'ds-1',
        'max_runs': 1,
        'poll_interval_seconds': 0,
        'run_timeout_seconds': 30,
    }
    values.update(overrides)
    return AgentConfig(**values)


def _run_decision():
    return RunExperimentDecision(
        decision='RUN_EXPERIMENT',
        model_type='logistic_regression',
        rationale='baseline',
    )


def test_training_poll_does_not_call_llm_and_finalize_is_real():
    autoai = FakeAutoAI(['queued', 'running', 'succeeded'])
    llm = FakeLLM([
        _run_decision(),
        FinalizeDecision(decision='FINALIZE', selected_run_id='run-1', rationale='validation ready'),
    ])
    result = run_agent(_cfg(), llm, autoai, sleep_fn=lambda _seconds: None)
    assert result['status'] == 'finalized'
    assert autoai.finalized == 'run-1'
    assert llm.call_count == 2
    assert autoai.feedback_calls == 3


def test_failed_run_is_only_followed_by_next_llm_decision():
    autoai = FakeAutoAI(['failed'])
    llm = FakeLLM([_run_decision(), RequestHumanDecision(decision='REQUEST_HUMAN', reason='failed')])
    result = run_agent(_cfg(), llm, autoai, sleep_fn=lambda _seconds: None)
    assert result['status'] == 'needs_human'
    assert llm.call_count == 2
    assert autoai.feedback_calls == 1


def test_timeout_stops_without_another_llm_call():
    autoai = FakeAutoAI(['queued'])
    llm = FakeLLM([_run_decision()])
    with pytest.raises(AgentTimeout):
        run_agent(_cfg(run_timeout_seconds=0), llm, autoai, sleep_fn=lambda _seconds: None)
    assert llm.call_count == 1
    assert autoai.feedback_calls == 1

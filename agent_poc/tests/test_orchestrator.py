from __future__ import annotations

import pytest

from agent_poc.orchestrator import AgentTimeout, _proposal_recipes, run_agent
from agent_poc.schemas import (
    FinalizeDecision,
    ReplanDecision,
    RequestHumanDecision,
    RunExperimentDecision,
)
from agent_poc.state import AgentConfig


class FakeAutoAI:
    def __init__(
        self,
        feedback_states,
        proposal_catalog=None,
        *,
        limited_replanning=False,
        failure_actions=None,
    ):
        self.feedback_states = list(feedback_states)
        self.feedback_calls = 0
        self.llm_poll_boundary = None
        self.experiments = []
        self.finalized = None
        self.proposal_catalog = proposal_catalog
        self.limited_replanning = limited_replanning
        self.failure_actions = failure_actions or ['choose_unused_proposal']

    def health(self):
        return {'status': 'ok', 'worker': {'available': True, 'compatible': True}}

    def create_session(self, payload):
        self.session_payload = payload
        return {'session_id': 'session-1', 'remaining_runs': payload['max_runs']}

    def get_session(self, _session_id):
        response = {
            'session_id': 'session-1',
            'remaining_runs': max(0, self.session_payload['max_runs'] - len(self.experiments)),
            'experiments': list(self.experiments),
        }
        if self.proposal_catalog is not None:
            response['locked_config'] = {
                'modules': {
                    'restricted_strategy_pool': True,
                    'limited_replanning': self.limited_replanning,
                },
            }
            response['context'] = {
                'proposal_catalog': {
                    'version': 'restricted-policy-v1',
                    'proposal_count': len(self.proposal_catalog),
                    'proposals': self.proposal_catalog,
                },
            }
        return response

    def create_experiment(self, _session_id, payload):
        run_id = f'run-{len(self.experiments) + 1}'
        self.experiments.append({
            'run_id': run_id,
            'state': 'queued',
            'parent_run_id': payload.get('parent_run_id'),
            'effective_action': payload,
            'validation_score': None,
        })
        return {'run_id': run_id, 'state': 'queued', 'effective_action': payload}

    def feedback(self, _session_id, run_id):
        self.feedback_calls += 1
        state = self.feedback_states.pop(0)
        experiment = next(item for item in self.experiments if item['run_id'] == run_id)
        if state == 'succeeded':
            experiment['state'] = 'succeeded'
            experiment['validation_score'] = 0.8
            return {
                'run_id': run_id,
                'state': 'succeeded',
                'validation': {'status': 'ready', 'metrics': {'macro_f1': 0.8}},
            }
        if state == 'failed':
            experiment['state'] = 'failed'
            experiment['diagnosis'] = {
                'status': 'ready',
                'allowed_action_ids': list(self.failure_actions),
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


class SchemaAwareFakeLLM(FakeLLM):
    def __init__(self, decisions):
        super().__init__(decisions)
        self.schemas = []

    def decide(self, _messages, *, decision_schema):
        self.call_count += 1
        self.schemas.append(decision_schema)
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


def test_restricted_catalog_recipe_is_forwarded_exactly():
    recipe = {
        'proposal_id': 'p_0123456789abcdef',
        'model_type': 'logistic_regression',
        'normalization': 'minmax',
        'class_balance': 'none',
    }
    autoai = FakeAutoAI(['succeeded'], proposal_catalog=[recipe])
    llm = FakeLLM([
        RunExperimentDecision(
            decision='RUN_EXPERIMENT',
            **recipe,
            rationale='compare',
        ),
        FinalizeDecision(
            decision='FINALIZE',
            selected_run_id='run-1',
            rationale='select_best',
        ),
    ])
    result = run_agent(_cfg(), llm, autoai, sleep_fn=lambda _seconds: None)
    assert result['status'] == 'finalized'
    assert (
        autoai.experiments[0]['effective_action']['proposal_id']
        == recipe['proposal_id']
    )


def test_proposal_catalog_metadata_and_ids_are_validated():
    recipe = {
        'proposal_id': 'p_0123456789abcdef',
        'model_type': 'svm',
        'normalization': 'zscore',
        'class_balance': 'none',
    }
    wrong_version = {
        'context': {
            'proposal_catalog': {
                'version': 'unknown',
                'proposal_count': 1,
                'proposals': [recipe],
            }
        }
    }
    duplicate = {
        'context': {
            'proposal_catalog': {
                'version': 'restricted-policy-v1',
                'proposal_count': 2,
                'proposals': [recipe, recipe],
            }
        }
    }
    assert _proposal_recipes(wrong_version) == ()
    assert _proposal_recipes(duplicate) == ()


def test_exhausted_catalog_switches_schema_to_finalize_before_max_runs():
    recipe = {
        'proposal_id': 'p_0123456789abcdef',
        'model_type': 'logistic_regression',
        'normalization': 'zscore',
        'class_balance': 'none',
    }
    autoai = FakeAutoAI(['succeeded'], proposal_catalog=[recipe])
    llm = SchemaAwareFakeLLM([
        RunExperimentDecision(
            decision='RUN_EXPERIMENT',
            **recipe,
            rationale='baseline',
        ),
        FinalizeDecision(
            decision='FINALIZE',
            selected_run_id='run-1',
            rationale='finalize',
        ),
    ])
    result = run_agent(
        _cfg(max_runs=2),
        llm,
        autoai,
        sleep_fn=lambda _seconds: None,
    )
    assert result['status'] == 'finalized'
    assert llm.schemas[1]['properties']['decision']['const'] == 'FINALIZE'


def test_unrestricted_mode_rejects_unexpected_proposal_id():
    decision = RunExperimentDecision(
        decision='RUN_EXPERIMENT',
        proposal_id='p_0123456789abcdef',
        model_type='logistic_regression',
        rationale='baseline',
    )
    autoai = FakeAutoAI([])
    result = run_agent(
        _cfg(),
        FakeLLM([decision]),
        autoai,
        sleep_fn=lambda _seconds: None,
    )
    assert result['status'] == 'needs_human'
    assert autoai.experiments == []


def test_failed_run_can_only_replan_to_unused_canonical_recipe():
    first = {
        'proposal_id': 'p_0123456789abcdef',
        'model_type': 'logistic_regression',
        'normalization': 'zscore',
        'class_balance': 'none',
    }
    second = {
        'proposal_id': 'p_fedcba9876543210',
        'model_type': 'svm',
        'normalization': 'zscore',
        'class_balance': 'none',
    }
    autoai = FakeAutoAI(
        ['failed', 'succeeded'],
        proposal_catalog=[first, second],
        limited_replanning=True,
    )
    llm = SchemaAwareFakeLLM([
        RunExperimentDecision(
            decision='RUN_EXPERIMENT', **first, rationale='baseline'
        ),
        ReplanDecision(
            decision='REPLAN',
            action_id='choose_unused_proposal',
            parent_run_id='run-1',
            **second,
            rationale='retry',
        ),
        FinalizeDecision(
            decision='FINALIZE',
            selected_run_id='run-2',
            rationale='finalize',
        ),
    ])
    result = run_agent(
        _cfg(max_runs=2), llm, autoai, sleep_fn=lambda _seconds: None
    )
    assert result['status'] == 'finalized'
    assert autoai.experiments[1]['effective_action']['parent_run_id'] == 'run-1'
    assert autoai.experiments[1]['effective_action']['action_id'] == (
        'choose_unused_proposal'
    )
    assert llm.schemas[1]['oneOf'][0]['properties']['decision']['const'] == 'REPLAN'


def test_stop_diagnosis_finalizes_previous_success_after_later_failure():
    first = {
        'proposal_id': 'p_0123456789abcdef',
        'model_type': 'logistic_regression',
        'normalization': 'zscore',
        'class_balance': 'none',
    }
    second = {
        'proposal_id': 'p_fedcba9876543210',
        'model_type': 'svm',
        'normalization': 'zscore',
        'class_balance': 'none',
    }
    third = {
        'proposal_id': 'p_aaaaaaaaaaaaaaaa',
        'model_type': 'random_forest',
        'normalization': 'zscore',
        'class_balance': 'none',
    }
    autoai = FakeAutoAI(
        ['succeeded', 'failed'],
        proposal_catalog=[first, second, third],
        limited_replanning=True,
        failure_actions=['choose_unused_proposal', 'stop'],
    )
    llm = SchemaAwareFakeLLM([
        RunExperimentDecision(
            decision='RUN_EXPERIMENT', **first, rationale='baseline'
        ),
        RunExperimentDecision(
            decision='RUN_EXPERIMENT', **second, rationale='compare'
        ),
        FinalizeDecision(
            decision='FINALIZE',
            selected_run_id='run-1',
            rationale='stop',
        ),
    ])
    result = run_agent(
        _cfg(max_runs=3), llm, autoai, sleep_fn=lambda _seconds: None
    )
    assert result['status'] == 'finalized'
    assert result['selected_run_id'] == 'run-1'
    final_schema = llm.schemas[2]
    decision_kinds = set()
    for branch in final_schema['oneOf']:
        if 'properties' in branch:
            decision_kinds.add(branch['properties']['decision']['const'])
        else:
            decision_kinds.add(
                branch['oneOf'][0]['properties']['decision']['const']
            )
    assert decision_kinds == {'REPLAN', 'FINALIZE'}


def test_replan_child_failure_cannot_exceed_chain_depth_one():
    first = {
        'proposal_id': 'p_0123456789abcdef',
        'model_type': 'logistic_regression',
        'normalization': 'zscore',
        'class_balance': 'none',
    }
    second = {
        'proposal_id': 'p_fedcba9876543210',
        'model_type': 'svm',
        'normalization': 'zscore',
        'class_balance': 'none',
    }
    third = {
        'proposal_id': 'p_aaaaaaaaaaaaaaaa',
        'model_type': 'random_forest',
        'normalization': 'zscore',
        'class_balance': 'none',
    }
    autoai = FakeAutoAI(
        ['failed', 'failed'],
        proposal_catalog=[first, second, third],
        limited_replanning=True,
    )
    llm = SchemaAwareFakeLLM([
        RunExperimentDecision(
            decision='RUN_EXPERIMENT', **first, rationale='baseline'
        ),
        ReplanDecision(
            decision='REPLAN',
            action_id='choose_unused_proposal',
            parent_run_id='run-1',
            **second,
            rationale='retry',
        ),
        RequestHumanDecision(
            decision='REQUEST_HUMAN', reason='needs_human'
        ),
    ])
    result = run_agent(
        _cfg(max_runs=3), llm, autoai, sleep_fn=lambda _seconds: None
    )
    assert result['status'] == 'needs_human'
    assert len(autoai.experiments) == 2
    assert llm.schemas[2]['properties']['decision']['const'] == 'REQUEST_HUMAN'

from __future__ import annotations

import json

import httpx
import pytest

import agent_poc.orchestrator as orchestrator_module

from agent_poc.orchestrator import (
    AgentTimeout,
    _failed_replan_context,
    _proposal_recipes,
    run_agent,
)
from agent_poc.budget import BudgetController, BudgetExceeded
from agent_poc.priors import load_prior_catalog
from agent_poc.schemas import (
    FinalizeDecision,
    ReplanDecision,
    RequestHumanDecision,
    RunExperimentDecision,
)
from agent_poc.state import AgentBudgetConfig, AgentConfig, AgentModuleConfig


class FakeAutoAI:
    def __init__(
        self,
        feedback_states,
        proposal_catalog=None,
        *,
        limited_replanning=False,
        failure_actions=None,
        decision_support=None,
        evidence_risks=None,
    ):
        self.feedback_states = list(feedback_states)
        self.feedback_calls = 0
        self.llm_poll_boundary = None
        self.experiments = []
        self.finalized = None
        self.proposal_catalog = proposal_catalog
        self.limited_replanning = limited_replanning
        self.failure_actions = failure_actions or ['choose_unused_proposal']
        self.decision_support = decision_support
        self.evidence_risks = evidence_risks

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
            if self.evidence_risks is not None:
                response['context']['evidence_card'] = {
                    'schema_version': 'small-sample-evidence-v1',
                    'status': 'ready',
                    'scope': 'train_only',
                    'statistics': {
                        'risk_codes': list(self.evidence_risks),
                    },
                }
        decision_support = (
            self.decision_support(self)
            if callable(self.decision_support)
            else self.decision_support
        )
        if decision_support is not None:
            response['decision_support'] = decision_support
            response['recommended_run_id'] = decision_support.get(
                'recommended_run_id'
            )
        budget = self.session_payload.get('budget')
        if isinstance(budget, dict):
            limit = budget['max_model_fits']
            response['budget_usage'] = {
                'schema_version': 'agent-budget-usage-v1',
                'model_fits': {
                    'limit': limit,
                    'actual': 0,
                    'reserved': 0,
                    'charged': 0,
                    'remaining': limit,
                }
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
        self.messages = []

    def decide(self, messages, *, decision_schema):
        self.call_count += 1
        self.schemas.append(decision_schema)
        self.messages.append(messages)
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


def _budget_cfg(**budget_overrides):
    budget_values = {
        'max_model_fits': 20,
        'max_llm_calls': 2,
        'max_api_calls': 100,
        'max_wall_clock_seconds': 300,
        'max_retry_attempts': 2,
    }
    budget_values.update(budget_overrides)
    return _cfg(
        modules=AgentModuleConfig(
            bounded_hpo=True,
            fail_fast_guard=True,
            budget_control=True,
        ),
        budget=AgentBudgetConfig(**budget_values),
    )


class BudgetAwareFakeLLM(FakeLLM):
    def set_budget_controller(self, controller):
        self.budget_controller = controller

    def decide(self, messages, *, decision_schema=None):
        self.budget_controller.before_llm_completion()
        return super().decide(messages)


def test_llm_budget_exhaustion_is_stable_and_prevents_finalize():
    autoai = FakeAutoAI(['succeeded'])
    llm = BudgetAwareFakeLLM([
        _run_decision(),
        FinalizeDecision(
            decision='FINALIZE',
            selected_run_id='run-1',
            rationale='must-not-be-called',
        ),
    ])
    result = run_agent(
        _budget_cfg(max_llm_calls=1),
        llm,
        autoai,
        sleep_fn=lambda _seconds: None,
    )
    assert result['status'] == 'budget_exhausted'
    assert result['dimension'] == 'max_llm_calls'
    assert result['session_id'] == 'session-1'
    assert result['agent_metrics']['llm_call_count'] == 1
    assert autoai.finalized is None
    assert len(autoai.experiments) == 1


def test_global_deadline_caps_sleep_and_stops_before_next_poll():
    class FakeClock:
        def __init__(self):
            self.value = 0.0

        def __call__(self):
            return self.value

        def sleep(self, seconds):
            sleeps.append(seconds)
            self.value += seconds

    class PollClient:
        def __init__(self):
            self.calls = 0

        def feedback(self, _session_id, _run_id):
            self.calls += 1
            return {'state': 'queued'}

    clock = FakeClock()
    sleeps = []
    controller = BudgetController(
        AgentBudgetConfig(
            max_model_fits=20,
            max_llm_calls=2,
            max_api_calls=100,
            max_wall_clock_seconds=2,
            max_retry_attempts=2,
        ),
        clock=clock,
    )
    autoai = PollClient()
    with pytest.raises(BudgetExceeded) as result:
        orchestrator_module.poll_until_terminal(
            autoai,
            session_id='session-1',
            run_id='run-1',
            interval_seconds=10,
            timeout_seconds=30,
            sleep_fn=clock.sleep,
            budget_controller=controller,
        )
    assert result.value.dimension == 'max_wall_clock_seconds'
    assert sleeps == [2]
    assert autoai.calls == 1


def test_agent_metrics_projects_backend_model_fit_actual():
    class FitUsageAutoAI(FakeAutoAI):
        def get_session(self, session_id):
            response = super().get_session(session_id)
            response['budget_usage'] = {
                'schema_version': 'agent-budget-usage-v1',
                'model_fits': {
                    'limit': 20,
                    'actual': 4,
                    'reserved': 0,
                    'charged': 4,
                    'remaining': 16,
                }
            }
            return response

    autoai = FitUsageAutoAI(['succeeded'])
    llm = BudgetAwareFakeLLM([
        _run_decision(),
        FinalizeDecision(
            decision='FINALIZE',
            selected_run_id='run-1',
            rationale='done',
        ),
    ])
    result = run_agent(
        _budget_cfg(),
        llm,
        autoai,
        sleep_fn=lambda _seconds: None,
    )
    assert result['status'] == 'finalized'
    assert result['agent_metrics']['model_fit_count'] == 4


def test_backend_model_fit_409_becomes_stable_budget_exhausted_without_get():
    class ModelBudgetAutoAI(FakeAutoAI):
        def __init__(self):
            super().__init__([])
            self.session_reads = 0

        def get_session(self, session_id):
            self.session_reads += 1
            response = super().get_session(session_id)
            response['budget_usage'] = {
                'schema_version': 'agent-budget-usage-v1',
                'model_fits': {
                    'limit': 3,
                    'actual': 0,
                    'reserved': 0,
                    'charged': 0,
                    'remaining': 3,
                }
            }
            return response

        def create_experiment(self, _session_id, _payload):
            request = httpx.Request('POST', 'http://autoai/experiments')
            response = httpx.Response(
                409,
                request=request,
                json={
                    'detail': {
                        'code': 'agent_budget_exhausted',
                        'dimension': 'max_model_fits',
                    }
                },
            )
            raise httpx.HTTPStatusError(
                'budget rejected',
                request=request,
                response=response,
            )

    autoai = ModelBudgetAutoAI()
    llm = BudgetAwareFakeLLM([_run_decision()])
    result = run_agent(
        _budget_cfg(max_model_fits=3),
        llm,
        autoai,
        sleep_fn=lambda _seconds: None,
    )
    assert result['status'] == 'budget_exhausted'
    assert result['dimension'] == 'max_model_fits'
    assert result['agent_metrics']['model_fit_count'] == 0
    assert autoai.session_reads == 1
    assert autoai.experiments == []


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


def test_decision_support_can_finalize_early_with_remaining_budget():
    support = {
        'recommended_run_id': 'run-1',
        'stop_recommendation': {
            'action': 'finalize',
            'reason': 'target_reached',
        },
    }
    autoai = FakeAutoAI(['succeeded'], decision_support=support)
    llm = SchemaAwareFakeLLM([
        _run_decision(),
        FinalizeDecision(
            decision='FINALIZE',
            selected_run_id='run-1',
            rationale='finalize',
        ),
    ])
    result = run_agent(
        _cfg(max_runs=3), llm, autoai, sleep_fn=lambda _seconds: None
    )
    assert result['status'] == 'finalized'
    assert len(autoai.experiments) == 1
    assert llm.schemas[1]['properties']['decision']['const'] == 'FINALIZE'


def test_stop_support_cannot_bypass_replan_only_failure_diagnosis():
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

    def support_after_failure(autoai):
        if not any(item['state'] == 'failed' for item in autoai.experiments):
            return None
        return {
            'recommended_run_id': 'run-1',
            'stop_recommendation': {
                'action': 'finalize',
                'reason': 'target_reached',
            },
        }

    autoai = FakeAutoAI(
        ['succeeded', 'failed', 'succeeded'],
        proposal_catalog=[first, second, third],
        limited_replanning=True,
        failure_actions=['choose_unused_proposal'],
        decision_support=support_after_failure,
    )
    llm = SchemaAwareFakeLLM([
        RunExperimentDecision(
            decision='RUN_EXPERIMENT', **first, rationale='baseline'
        ),
        RunExperimentDecision(
            decision='RUN_EXPERIMENT', **second, rationale='compare'
        ),
        ReplanDecision(
            decision='REPLAN',
            action_id='choose_unused_proposal',
            parent_run_id='run-2',
            **third,
            rationale='repair before stopping',
        ),
        FinalizeDecision(
            decision='FINALIZE',
            selected_run_id='run-1',
            rationale='finalize after repair',
        ),
    ])
    result = run_agent(
        _cfg(max_runs=3), llm, autoai, sleep_fn=lambda _seconds: None
    )
    assert result['status'] == 'finalized'
    assert len(autoai.experiments) == 3
    assert autoai.experiments[2]['effective_action']['parent_run_id'] == 'run-2'
    assert autoai.experiments[2]['effective_action']['action_id'] == (
        'choose_unused_proposal'
    )


def test_stop_support_honors_failure_diagnosis_when_replanning_disabled():
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

    def support_after_failure(autoai):
        if not any(item['state'] == 'failed' for item in autoai.experiments):
            return None
        return {
            'recommended_run_id': 'run-1',
            'stop_recommendation': {
                'action': 'finalize',
                'reason': 'proposal_catalog_exhausted',
            },
        }

    autoai = FakeAutoAI(
        ['succeeded', 'failed'],
        proposal_catalog=[first, second],
        limited_replanning=False,
        failure_actions=['request_human'],
        decision_support=support_after_failure,
    )
    llm = SchemaAwareFakeLLM([
        RunExperimentDecision(
            decision='RUN_EXPERIMENT', **first, rationale='baseline'
        ),
        RunExperimentDecision(
            decision='RUN_EXPERIMENT', **second, rationale='compare'
        ),
        RequestHumanDecision(
            decision='REQUEST_HUMAN', reason='failure requires operator'
        ),
    ])
    result = run_agent(
        _cfg(max_runs=3), llm, autoai, sleep_fn=lambda _seconds: None
    )
    assert result['status'] == 'needs_human'
    assert len(autoai.experiments) == 2
    assert llm.schemas[2]['properties']['decision']['const'] == 'REQUEST_HUMAN'


def test_missing_failure_diagnosis_fails_closed():
    observation = {
        'experiments': [{
            'run_id': 'run-failed',
            'state': 'failed',
            'effective_action': {'model_type': 'svm'},
        }],
    }
    assert _failed_replan_context(observation) == (('run-failed',), ())


def test_ordinary_parent_link_does_not_mark_failure_as_replanned():
    observation = {
        'experiments': [
            {
                'run_id': 'run-failed',
                'state': 'failed',
                'diagnosis': {'allowed_action_ids': ['request_human']},
                'effective_action': {'model_type': 'svm'},
            },
            {
                'run_id': 'run-ordinary-child',
                'state': 'succeeded',
                'parent_run_id': 'run-failed',
                'effective_action': {
                    'model_type': 'logistic_regression',
                },
            },
        ],
    }
    assert _failed_replan_context(observation) == (
        ('run-failed',),
        ('request_human',),
    )


def test_static_prior_catalog_is_loaded_once_per_agent_invocation(monkeypatch):
    catalog = load_prior_catalog()
    calls = []

    def counted_loader():
        calls.append('loaded')
        return catalog

    monkeypatch.setattr(orchestrator_module, 'load_prior_catalog', counted_loader)
    autoai = FakeAutoAI(['succeeded'])
    llm = FakeLLM([
        _run_decision(),
        FinalizeDecision(
            decision='FINALIZE',
            selected_run_id='run-1',
            rationale='finalize',
        ),
    ])
    result = run_agent(
        _cfg(modules=AgentModuleConfig(case_memory=True)),
        llm,
        autoai,
        sleep_fn=lambda _seconds: None,
    )
    assert result['status'] == 'finalized'
    assert calls == ['loaded']


def test_disabled_case_memory_does_not_load_or_inject_static_prior(monkeypatch):
    def unexpected_loader():
        raise AssertionError('disabled static prior must not be loaded')

    monkeypatch.setattr(
        orchestrator_module,
        'load_prior_catalog',
        unexpected_loader,
    )
    autoai = FakeAutoAI(['succeeded'])
    llm = FakeLLM([
        _run_decision(),
        FinalizeDecision(
            decision='FINALIZE',
            selected_run_id='run-1',
            rationale='finalize',
        ),
    ])
    result = run_agent(_cfg(), llm, autoai, sleep_fn=lambda _seconds: None)
    assert result['status'] == 'finalized'


def test_orchestrator_injects_prior_only_while_mapped_proposal_is_available():
    recipe = {
        'proposal_id': 'p_0123456789abcdef',
        'model_type': 'logistic_regression',
        'normalization': 'zscore',
        'class_balance': 'none',
    }
    autoai = FakeAutoAI(
        ['succeeded'],
        proposal_catalog=[recipe],
        evidence_risks=['very_small_train_partition'],
    )
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
        _cfg(
            modules=AgentModuleConfig(
                evidence_card=True,
                restricted_strategy_pool=True,
                case_memory=True,
            )
        ),
        llm,
        autoai,
        sleep_fn=lambda _seconds: None,
    )
    assert result['status'] == 'finalized'
    first_payload = json.loads(llm.messages[0][1]['content'])
    assert first_payload['static_prior']['recommendations'][0][
        'proposal_id'
    ] == recipe['proposal_id']
    second_payload = json.loads(llm.messages[1][1]['content'])
    assert 'static_prior' not in second_payload

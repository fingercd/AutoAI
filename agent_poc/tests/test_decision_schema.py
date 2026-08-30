from __future__ import annotations

import pytest

from agent_poc.schemas import DECISION_ADAPTER, DecisionParseError, decision_json_schema
from agent_poc.prompts import build_repair_messages


def test_decision_union_accepts_run_finalize_and_request_human():
    run = DECISION_ADAPTER.validate_python({
        'decision': 'RUN_EXPERIMENT',
        'model_type': 'logistic_regression',
        'rationale': 'baseline',
    })
    finalize = DECISION_ADAPTER.validate_python({
        'decision': 'FINALIZE',
        'selected_run_id': 'run-1',
        'rationale': 'validation is ready',
    })
    human = DECISION_ADAPTER.validate_python({
        'decision': 'REQUEST_HUMAN',
        'reason': 'no safe action',
    })
    assert run.decision == 'RUN_EXPERIMENT'
    assert finalize.decision == 'FINALIZE'
    assert human.decision == 'REQUEST_HUMAN'


def test_extra_or_invalid_fields_are_rejected():
    with pytest.raises(Exception):
        DECISION_ADAPTER.validate_json(
            '{"decision":"RUN_EXPERIMENT","model_type":"svm",'
            '"rationale":"x","unexpected":true}'
        )
    with pytest.raises(Exception):
        DECISION_ADAPTER.validate_json('{"decision":"RUN_EXPERIMENT"}')


def test_wire_schema_locks_run_to_canonical_proposal_recipe():
    recipe = {
        'proposal_id': 'p_0123456789abcdef',
        'model_type': 'svm',
        'normalization': 'minmax',
        'class_balance': 'class_weight',
    }
    schema = decision_json_schema(
        allowed_decisions=('RUN_EXPERIMENT',),
        allowed_models=('svm',),
        proposal_recipes=(recipe,),
    )
    branch = schema['oneOf'][0]
    assert branch['properties']['proposal_id']['const'] == recipe['proposal_id']
    assert branch['properties']['normalization']['const'] == 'minmax'
    decision = DECISION_ADAPTER.validate_python({
        'decision': 'RUN_EXPERIMENT',
        **recipe,
        'rationale': 'compare',
    })
    assert decision.proposal_id == recipe['proposal_id']


def test_replan_schema_locks_failed_parent_action_and_recipe():
    recipe = {
        'proposal_id': 'p_0123456789abcdef',
        'model_type': 'svm',
        'normalization': 'zscore',
        'class_balance': 'none',
    }
    schema = decision_json_schema(
        allowed_decisions=('REPLAN',),
        allowed_models=('svm',),
        proposal_recipes=(recipe,),
        failed_run_ids=('run-failed',),
        allowed_replan_actions=('choose_unused_proposal',),
    )
    branch = schema['oneOf'][0]
    assert branch['properties']['decision']['const'] == 'REPLAN'
    assert branch['properties']['parent_run_id']['enum'] == ['run-failed']
    assert branch['properties']['action_id']['const'] == 'choose_unused_proposal'


def test_repair_prompt_defers_to_current_wire_schema():
    repaired = build_repair_messages([
        {'role': 'system', 'content': 'schema'},
        {'role': 'user', 'content': 'observation'},
    ])
    instruction = repaired[-1]['content']
    assert 'decision branch allowed by the schema' in instruction
    assert 'RUN_EXPERIMENT, FINALIZE' not in instruction

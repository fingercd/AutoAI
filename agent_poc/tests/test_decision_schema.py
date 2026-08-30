from __future__ import annotations

import pytest

from agent_poc.schemas import DECISION_ADAPTER, DecisionParseError


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

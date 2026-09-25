"""Budget State freezes policy identity independently of mutable cost projections."""

import time

import pytest

from agent_poc.orchestration.state import apply_patch, new_state, validate_state
from backend.app.agent.budget import BudgetPolicy, DIMENSIONS


def test_v7_policy_and_awareness_are_immutable():
    now = time.time()
    limits = {name: 100 for name in DIMENSIONS}
    limits.update(experiments=1, model_fits=2, training_epochs=0,
                  llm_calls=6, api_calls=12, input_tokens=None,
                  cached_tokens=None, output_tokens=6144)
    policy = BudgetPolicy('state-v7-task', now, now + 180, now + 120, limits)
    state = new_state(dataset_id='dataset', allowed_models=['logistic_regression'],
        backend_fingerprint='a' * 64, principal_fingerprint='b' * 64,
        llm_config_fingerprint='c' * 64, task_id=policy.task_id,
        max_llm_calls=6, max_api_calls=12, timeout_seconds=180, now=now,
        wire_version='agent-state-v7', processing_mode='fixed',
        search_mode='fixed', max_trials=1,
        budget_policy=policy.as_dict(), budget_awareness='off',
        canonical_journal_sha256='d' * 64)
    assert validate_state(state).versions.state == 'agent-state-v7'
    assert state['task']['budget_policy_digest'] == policy.digest
    with pytest.raises(ValueError, match='frozen|budget'):
        apply_patch(state, {'task': {'budget_awareness': 'on'}})

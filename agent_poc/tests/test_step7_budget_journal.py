"""Call owner: preparation, dispatch and settlement have distinct costs."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_poc.orchestration.persistence import CallJournal, PersistenceError
from backend.app.agent.budget import BudgetPolicy, DIMENSIONS


def policy() -> BudgetPolicy:
    limits = {name: 20 for name in DIMENSIONS}
    limits.update(experiments=1, llm_calls=6, api_calls=12,
                  input_tokens=None, cached_tokens=None)
    return BudgetPolicy('task-journal', 100.0, 200.0, 140.0, limits)


def begin(journal: CallJournal, operation: str, *, phase='work') -> int:
    return journal.begin(operation_id=operation, kind='api', name='inspect_ml_session',
                         maximum=12, max_attempts=3, deadline=200.0, now=101.0,
                         task_id='task-journal', phase=phase)


def test_prepare_failure_is_not_physical_and_dispatch_crash_holds(tmp_path: Path):
    journal = CallJournal(tmp_path / 'calls.sqlite', 'thread-1')
    journal.bind_budget(policy())
    prepared = begin(journal, 'prepare')
    assert journal.budget_summary()['api_calls']['held_reserved'] == 1
    journal.finish(prepared, error_code='prompt_invalid')
    assert journal.budget_summary()['api_calls']['remaining'] == 12
    sent = begin(journal, 'sent')
    journal.mark_dispatched(sent)
    journal.close()
    recovered = CallJournal(tmp_path / 'calls.sqlite', 'thread-1')
    recovered.bind_budget(policy())
    assert recovered.budget_summary()['api_calls']['held_reserved'] == 1
    recovered.finish(sent, error_code='timeout')
    summary = recovered.budget_summary()['api_calls']
    assert (summary['known_actual'], summary['held_reserved'], summary['remaining']) == (1, 0, 11)
    recovered.close()


def test_finalization_bucket_and_policy_conflict(tmp_path: Path):
    journal = CallJournal(tmp_path / 'calls.sqlite', 'thread-1')
    frozen = policy()
    journal.bind_budget(frozen)
    for index in range(3):
        call_id = begin(journal, f'ordinary-{index}')
        journal.mark_dispatched(call_id)
        journal.finish(call_id)
    with pytest.raises(PersistenceError, match='budget_insufficient'):
        begin(journal, 'ordinary-overflow')
    call_id = begin(journal, 'final', phase='finalization')
    journal.mark_dispatched(call_id)
    journal.finish(call_id)
    with pytest.raises(Exception, match='budget_policy_conflict'):
        changed = BudgetPolicy('task-journal', 100.0, 201.0, 140.0, frozen.limits)
        journal.bind_budget(changed)
    journal.close()

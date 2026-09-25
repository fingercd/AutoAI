"""Call owner: preparation, dispatch and settlement have distinct costs."""
from __future__ import annotations

from pathlib import Path
from contextlib import closing
import sqlite3

import pytest

from agent_poc.orchestration.persistence import CallJournal, PersistenceError
from backend.app.agent.budget import BudgetPolicy, DIMENSIONS
from agent_poc.tests.test_llm import adapter, context, envelope, action
from agent_poc.orchestration.llm import LLMError, _usage
from agent_poc.clients.autoai_client import AutoAIClient, AgentTimeoutError
from agent_poc.tests.test_autoai_client import FakeTransport, Response, HEALTH
import json


def policy() -> BudgetPolicy:
    limits = {name: 20 for name in DIMENSIONS}
    limits.update(experiments=1, llm_calls=6, api_calls=12,
                  input_tokens=None, cached_tokens=None, output_tokens=6144)
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
    journal.mark_dispatched(sent, now=101.0)
    journal.close()
    recovered = CallJournal(tmp_path / 'calls.sqlite', 'thread-1')
    recovered.bind_budget(policy())
    assert recovered.budget_summary()['api_calls']['held_unknown'] == 1
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
        journal.mark_dispatched(call_id, now=101.0)
        journal.finish(call_id)
    with pytest.raises(PersistenceError, match='budget_insufficient'):
        begin(journal, 'ordinary-overflow')
    call_id = begin(journal, 'final', phase='finalization')
    journal.mark_dispatched(call_id, now=199.0)
    journal.finish(call_id)
    with pytest.raises(Exception, match='budget_policy_conflict'):
        changed = BudgetPolicy('task-journal', 100.0, 201.0, 140.0, frozen.limits)
        journal.bind_budget(changed)
    journal.close()


def test_budget_journal_binding_rejects_another_thread_or_copied_store(tmp_path: Path):
    source = tmp_path / 'calls.sqlite'
    journal = CallJournal(source, 'thread-1')
    frozen = policy()
    journal.bind_budget(frozen)
    journal.close()

    with closing(CallJournal(source, 'thread-2')) as other_thread:
        with pytest.raises(PersistenceError, match='budget_journal_binding_conflict'):
            other_thread.bind_budget(frozen)

    copied = tmp_path / 'copied.sqlite'
    original = CallJournal(source, 'thread-1')
    with sqlite3.connect(copied) as destination:
        original.connection.backup(destination)
    original.close()
    with closing(CallJournal(copied, 'thread-1')) as wrong_path:
        with pytest.raises(PersistenceError, match='budget_journal_binding_conflict'):
            wrong_path.bind_budget(frozen)


def test_budget_binding_cannot_retroactively_ignore_legacy_calls(tmp_path: Path):
    with closing(CallJournal(tmp_path / 'calls.sqlite', 'thread-1')) as journal:
        journal.begin(operation_id='old', kind='api', name='inspect', maximum=12,
                      max_attempts=3, deadline=200.0, now=101.0)
        with pytest.raises(PersistenceError, match='budget_binding_after_calls'):
            journal.bind_budget(policy())


def test_reopened_budget_journal_rejects_unbound_begin_and_finish(tmp_path: Path):
    path = tmp_path / 'calls.sqlite'
    frozen = policy()
    with closing(CallJournal(path, 'thread-1')) as first:
        first.bind_budget(frozen)
        pending = begin(first, 'pending')
    with closing(CallJournal(path, 'thread-1')) as reopened:
        with pytest.raises(PersistenceError, match='budget_binding_required'):
            begin(reopened, 'free-work')
        with pytest.raises(PersistenceError, match='budget_binding_required'):
            reopened.finish(pending)
        reopened.bind_budget(frozen)
        reopened.finish(pending, error_code='prepare_failed')
        assert reopened.budget_summary()['api_calls']['known_actual'] == 0
        assert reopened.budget_summary()['api_calls']['held_reserved'] == 0


def test_copied_budget_journal_rejects_unbound_calls(tmp_path: Path):
    source, copy = tmp_path / 'calls.sqlite', tmp_path / 'copy.sqlite'
    with closing(CallJournal(source, 'thread-1')) as first:
        first.bind_budget(policy())
        with sqlite3.connect(copy) as destination:
            first.connection.backup(destination)
    with closing(CallJournal(copy, 'thread-1')) as copied:
        with pytest.raises(PersistenceError, match='budget_journal_binding_conflict'):
            begin(copied, 'copied-free-work')


def test_rebinding_detects_an_older_unmetered_call_and_policy_tampering(tmp_path: Path):
    path = tmp_path / 'calls.sqlite'
    with closing(CallJournal(path, 'thread-1')) as journal:
        journal.bind_budget(policy())
    with sqlite3.connect(path) as db:
        db.execute('''INSERT INTO orchestration_calls_v1
            (thread_id,operation_id,kind,name,status,started_at,task_id)
            VALUES('thread-1','leaked','api','inspect','confirmed',101,'task-journal')''')
    with closing(CallJournal(path, 'thread-1')) as reopened:
        with pytest.raises(PersistenceError, match='budget_unmetered_call_conflict'):
            reopened.bind_budget(policy())
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM orchestration_calls_v1 WHERE operation_id='leaked'")
        db.execute("UPDATE task_budget_policies_v1 SET policy_json='{}' WHERE task_id='task-journal'")
    with closing(CallJournal(path, 'thread-1')) as reopened:
        reopened.bind_budget(policy())
        with pytest.raises(PersistenceError, match='budget_policy_conflict'):
            begin(reopened, 'tampered')


@pytest.mark.parametrize(('phase', 'prepared_at', 'dispatched_at'), [
    ('work', 139.0, 140.0), ('work', 139.0, 201.0),
    ('finalization', 199.0, 200.0),
])
def test_expired_prepared_llm_cannot_send_and_releases_hold(
        tmp_path: Path, phase: str, prepared_at: float, dispatched_at: float):
    with closing(CallJournal(tmp_path / 'calls.sqlite', 'thread-1')) as journal:
        journal.bind_budget(policy())
        call_id = journal.begin(operation_id='slow-prompt', kind='llm', name='submit',
            maximum=6, max_attempts=3, deadline=200.0, now=prepared_at,
            task_id='task-journal', phase=phase)
        llm = adapter(envelope(json.dumps(action())))
        with pytest.raises(PersistenceError, match='budget_deadline_exceeded'):
            llm.propose('submit', context(), on_dispatch=lambda: journal.mark_dispatched(
                call_id, now=dispatched_at))
        journal.finish(call_id, error_code='budget_deadline_exceeded')
        assert llm._transport.calls == []
        assert journal.snapshot()[0]['status'] == 'prepare_failed'
        assert journal.budget_summary()['llm_calls']['held_reserved'] == 0
        assert journal.budget_summary()['llm_calls']['known_actual'] == 0


def test_api_dispatch_deadline_and_transport_timeout_use_current_remaining(tmp_path: Path):
    with closing(CallJournal(tmp_path / 'calls.sqlite', 'thread-1')) as journal:
        journal.bind_budget(policy())
        transport = FakeTransport([Response(200, HEALTH)])
        client = AutoAIClient('http://127.0.0.1:9000', max_retries=0, transport=transport)
        call_id = begin(journal, 'expired-api')
        client.on_dispatch = lambda: journal.mark_dispatched(call_id, now=140.0)
        with pytest.raises(PersistenceError, match='budget_deadline_exceeded'):
            client._request('GET', '/api/agent/health', operation='inspect_ml_capabilities')
        assert transport.calls == []
        assert journal.budget_summary()['api_calls']['held_reserved'] == 0
        live = begin(journal, 'short-api')
        client.on_dispatch = lambda: journal.mark_dispatched(live, now=139.75)
        client._request('GET', '/api/agent/health', operation='inspect_ml_capabilities')
        assert transport.calls[0][4] == pytest.approx(0.25)
        final = begin(journal, 'expired-final-api', phase='finalization')
        client.on_dispatch = lambda: journal.mark_dispatched(final, now=200.0)
        with pytest.raises(PersistenceError, match='budget_deadline_exceeded'):
            client._request('GET', '/api/agent/health', operation='inspect_ml_capabilities')
        assert len(transport.calls) == 1


def test_llm_transport_timeout_uses_post_prepare_remaining(tmp_path: Path):
    with closing(CallJournal(tmp_path / 'calls.sqlite', 'thread-1')) as journal:
        journal.bind_budget(policy())
        call_id = journal.begin(operation_id='short-llm', kind='llm', name='submit',
            maximum=6, max_attempts=3, deadline=200.0, now=139.0,
            task_id='task-journal')
        llm = adapter(envelope(json.dumps(action())))
        llm.propose('submit', context(), timeout_seconds=60,
            on_dispatch=lambda: journal.mark_dispatched(call_id, now=139.75))
        assert llm._transport.calls[0][4] == pytest.approx(0.25)


def test_llm_prepare_failure_does_not_mark_send_and_valid_request_does(tmp_path: Path):
    journal = CallJournal(tmp_path / 'calls.sqlite', 'thread-1')
    journal.bind_budget(policy())
    first = journal.begin(operation_id='llm-prepare', kind='llm', name='submit',
        maximum=6, max_attempts=3, deadline=200.0, now=101.0, task_id='task-journal')
    llm = adapter(envelope(json.dumps(action())))
    with pytest.raises(LLMError):
        llm.propose('submit', {}, on_dispatch=lambda: journal.mark_dispatched(first))
    journal.finish(first, error_code='llm_context_invalid')
    assert len(llm._transport.calls) == 0
    assert journal.budget_summary()['llm_calls']['known_actual'] == 0
    second = journal.begin(operation_id='llm-send', kind='llm', name='submit',
        maximum=6, max_attempts=3, deadline=200.0, now=101.0, task_id='task-journal')
    llm.propose('submit', context(), on_dispatch=lambda: journal.mark_dispatched(second, now=101.0))
    journal.finish(second)
    assert journal.budget_summary()['llm_calls']['known_actual'] == 1
    journal.close()


def test_api_callback_is_at_transport_boundary(tmp_path: Path):
    class BlockedTransport:
        def request(self, *args, **kwargs):
            raise TimeoutError('simulated timeout')

    journal = CallJournal(tmp_path / 'calls.sqlite', 'thread-1')
    journal.bind_budget(policy())
    call_id = begin(journal, 'api-send')
    client = AutoAIClient('http://127.0.0.1:9000', max_retries=0,
                          transport=BlockedTransport())
    client.on_dispatch = lambda: journal.mark_dispatched(call_id, now=101.0)
    with pytest.raises(AgentTimeoutError):
        client._request('GET', '/api/agent/capabilities', operation='inspect')
    assert journal.budget_summary()['api_calls']['held_unknown'] == 1
    journal.finish(call_id, error_code='api_timeout')
    assert journal.budget_summary()['api_calls']['known_actual'] == 1
    journal.close()


def test_provider_cached_and_total_only_usage_are_distinct():
    cached = _usage({'usage': {'prompt_tokens': 10, 'completion_tokens': 4,
        'total_tokens': 14, 'prompt_tokens_details': {'cached_tokens': 6}}})
    assert (cached.prompt_tokens, cached.cached_tokens, cached.total_tokens) == (10, 6, 14)
    missing = _usage({'usage': {'prompt_tokens': 10, 'completion_tokens': 4,
        'total_tokens': 14}})
    assert missing.cached_tokens is None
    total_only = _usage({'usage': {'total_tokens': 14}})
    assert total_only.status == 'partial'
    assert total_only.prompt_tokens is None and total_only.completion_tokens is None
    with pytest.raises(ValueError, match='inconsistent cached usage'):
        _usage({'usage': {'prompt_tokens': 10,
            'prompt_tokens_details': {'cached_tokens': 11}}})


def test_llm_timing_is_split_at_actual_request_boundary():
    payload = envelope(json.dumps(action()))
    payload['usage']['prompt_tokens_details'] = {'cached_tokens': 6}
    llm = adapter(payload)
    proposal = llm.propose('submit', context())
    timing = llm.last_request_measurement
    assert proposal.usage.cached_tokens == 6
    assert timing['prepare_duration_seconds'] >= 0
    assert timing['request_duration_seconds'] >= 0
    assert timing['parse_duration_seconds'] >= 0


def test_output_reservation_unknown_and_provider_overrun(tmp_path: Path):
    journal = CallJournal(tmp_path / 'calls.sqlite', 'thread-1')
    journal.bind_budget(policy())
    call_id = journal.begin(operation_id='llm-1', kind='llm', name='submit',
        maximum=6, max_attempts=3, deadline=200.0, now=101.0, task_id='task-journal')
    assert journal.budget_summary()['output_tokens']['held_reserved'] == 1024
    journal.mark_dispatched(call_id, now=101.0)
    assert journal.budget_summary()['output_tokens']['held_unknown'] == 1024
    journal.finish(call_id, output_tokens=7000)
    tokens = journal.budget_summary()['output_tokens']
    assert tokens['known_actual'] == 7000 and tokens['budget_breach']
    with pytest.raises(PersistenceError, match='budget_insufficient'):
        journal.begin(operation_id='llm-2', kind='llm', name='submit',
            maximum=6, max_attempts=3, deadline=200.0, now=102.0, task_id='task-journal')
    journal.close()


def test_hard_input_and_cached_limit_reserve_verified_full_bound(tmp_path: Path):
    base = policy()
    limits = dict(base.limits)
    limits['input_tokens'] = limits['cached_tokens'] = 40
    frozen = BudgetPolicy('task-journal', 100.0, 200.0, 140.0, limits,
        input_tokens_per_call_bound=10, input_bound_source='provider_verified')
    journal = CallJournal(tmp_path / 'calls.sqlite', 'thread-1')
    journal.bind_budget(frozen)
    call_id = journal.begin(operation_id='llm-1', kind='llm', name='submit',
        maximum=6, max_attempts=3, deadline=200.0, now=101.0, task_id='task-journal')
    assert journal.budget_summary()['cached_tokens']['held_reserved'] == 10
    journal.mark_dispatched(call_id, now=101.0)
    journal.finish(call_id, input_tokens=8, output_tokens=4, cached_tokens=None)
    assert journal.budget_summary()['input_tokens']['known_actual'] == 8
    cached = journal.budget_summary()['cached_tokens']
    assert cached['known_actual'] == 0 and cached['held_unknown'] == 10
    with pytest.raises(PersistenceError, match='budget_insufficient'):
        journal.begin(operation_id='llm-2', kind='llm', name='submit',
            maximum=6, max_attempts=3, deadline=200.0, now=102.0, task_id='task-journal')
    journal.close()

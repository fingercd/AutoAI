"""New tasks bind the only Journal before Graph work and reject moved storage."""

import shutil
import sqlite3

import pytest
from langgraph.graph import END, START, StateGraph

from agent_poc.clients.autoai_client import AutoAIClient
from agent_poc.orchestration.llm import LLMAdapter
from agent_poc.orchestration.runtime import RuntimeErrorCode, resume_task, start_task
from agent_poc.orchestration.state import GraphState, apply_patch
from agent_poc.tests.test_orchestration_runtime import fixture_config


def _terminal_graph(dependencies, saver):
    def terminal(state):
        assert dependencies.journal.budget_policy is not None
        return apply_patch(state, {'lifecycle': {
            'status': 'failed', 'stage': 'ended', 'next_action': None,
            'reason_code': 'fixture_completed', 'ended_at': dependencies.clock()},
            'finalization': {'status': 'unselected',
                             'termination_reason': 'fixture_completed'}})
    graph = StateGraph(GraphState)
    graph.add_node('terminal', terminal)
    graph.add_edge(START, 'terminal')
    graph.add_edge('terminal', END)
    return graph.compile(checkpointer=saver)


def test_runtime_binds_v7_journal_and_rejects_copied_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(LLMAdapter, 'validate_prompt_budget', lambda self: None)
    config = fixture_config()
    client = AutoAIClient(config.backend_url, api_version='v2',
        execution_profile='train-evidence-recipes-v1',
        protocol_revision='agent-recipes-revision-v5', max_retries=0)
    storage = tmp_path / 'canonical'
    state = start_task(config, storage=storage, client=client,
        dataset_id='dataset', allowed_models=['logistic_regression'],
        thread_id='v7-thread', task_id='v7-task', knowledge=False,
        budget_awareness='off', graph_factory=_terminal_graph)
    assert state['versions']['state'] == 'agent-state-v7'
    with sqlite3.connect(storage / 'calls.sqlite') as db:
        row = db.execute('''SELECT task_id,thread_id,policy_digest FROM
            task_budget_journal_bindings_v1''').fetchone()
    assert row == ('v7-task', 'v7-thread', state['task']['budget_policy_digest'])
    assert resume_task(config, storage=storage, thread_id='v7-thread')['identity']['task_id'] == 'v7-task'
    moved = tmp_path / 'copied'
    shutil.copytree(storage, moved)
    with pytest.raises(RuntimeErrorCode, match='budget_journal_binding_mismatch'):
        resume_task(config, storage=moved, thread_id='v7-thread')

"""Independent acceptance probes; use only pytest temporary storage."""
import pytest
import time
from datetime import datetime, timezone

from backend.tests.test_agent_model_sessions import api
from agent_poc.tests.test_orchestration_integration import BackendTransport
from agent_poc.tests.test_graph_model_execution import ScriptedDecision
from agent_poc.clients.autoai_client import AutoAIClient
from agent_poc.orchestration.llm import LLMConfig, LLMAdapter
from agent_poc.orchestration.runtime import RuntimeConfig, start_task, resume_task, TaskInterrupted, read_status
from agent_poc.orchestration.graph import Nodes


class CountTransport(BackendTransport):
    def __init__(self, api_client):
        super().__init__(api_client)
        self.requests = []
        self.timeouts = []

    def request(self, method, url, **kwargs):
        self.requests.append((method, url))
        self.timeouts.append(kwargs['timeout'])
        return super().request(method, url, **kwargs)


def components(api, model):
    transport, storage, dataset = api
    wire = CountTransport(transport)
    config = LLMConfig('http://scripted.invalid/v1', 'scripted', prompt_version='agent-decision-step2-v1')
    runtime = RuntimeConfig('http://backend.invalid', 'local', config)
    llm = LLMAdapter(config, transport=ScriptedDecision(model))
    def client():
        return AutoAIClient(runtime.backend_url, transport=wire, api_version='v2', max_retries=0)
    return wire, runtime, llm, client, dataset


@pytest.mark.parametrize('model', ['logistic_regression', 'cnn1d'])
@pytest.mark.parametrize('after_choose_tick', [False, True, 'session'])
@pytest.mark.parametrize('limit', [4, 60])
def test_resume_submit_counts_every_http_request(api, tmp_path, monkeypatch, after_choose_tick, model, limit):
    wire, runtime, llm, client, dataset = components(api, model)
    target = 'session' if after_choose_tick == 'session' else 'submit'
    original = getattr(Nodes, target)
    ticks = [time.time()]
    clock = lambda: ticks[0]
    def interrupt_before_submit(self, state):
        raise KeyboardInterrupt()
    if after_choose_tick is not True:
        monkeypatch.setattr(Nodes, target, interrupt_before_submit)
    def interrupting_graph_factory(deps, saver):
        from agent_poc.orchestration.graph import build_graph
        graph = build_graph(deps, saver)
        class Proxy:
            def __getattr__(self, name):
                return getattr(graph, name)
            def invoke(self, *args, **kwargs):
                state = graph.invoke(*args, **kwargs)
                if after_choose_tick is True and state['lifecycle']['next_action'] == 'submit':
                    # Graph.invoke already durably wrote the choose END checkpoint.
                    raise KeyboardInterrupt()
                return state
        return Proxy()
    with pytest.raises(TaskInterrupted):
        start_task(runtime, dataset_id=dataset, allowed_models=[model], model_configs={model:{'epochs':2}} if model=='cnn1d' else {},
                   storage=tmp_path/'checkpoint', thread_id='audit', client=client(), llm=llm,
                   max_api_calls=limit, graph_factory=interrupting_graph_factory, timeout_seconds=30, clock=clock)
    assert len(wire.requests) == (1 if after_choose_tick == 'session' else 3)
    before = len(wire.requests)
    ticks[0] += 25
    monkeypatch.setattr(Nodes, target, original)
    restored_client = client()
    from agent_poc.tools import ToolDispatcher
    original_dispatch = ToolDispatcher.dispatch
    def traced_dispatch(self, name, arguments):
        try:
            return original_dispatch(self, name, arguments)
        except Exception as error:
            print({'dispatch_failure_type': type(error).__name__, 'tool': name,
                   'argument_keys': sorted(arguments),
                   'schema_keys': sorted(self.client.tool_schemas[name]['properties'])})
            raise
    monkeypatch.setattr(ToolDispatcher, 'dispatch', traced_dispatch)
    state = resume_task(runtime, storage=tmp_path/'checkpoint', thread_id='audit',
                        client=restored_client, llm=llm, clock=clock)
    actual = state['budget']['api_calls']['actual']
    assert len(wire.requests) == actual
    assert all(0 < t <= 5 for t in wire.timeouts[before:])
    if limit == 4: assert actual == 4
    assert state['execution']['run_id'] is not None
    print({'physical_http': len(wire.requests), 'journal_api_actual': actual,
           'lifecycle': state['lifecycle'], 'requests': wire.requests})
    if after_choose_tick:
        # Fresh tick runs prepare and restores schema, but no Session cache.
        assert len(wire.requests) == actual
    else:
        assert state['execution']['run_id'] is not None, 'Prepared v2 submit could not resume with a fresh Client'

    from backend.app.runs.repository import RunRepository
    repo = RunRepository(api[1]/'runs.sqlite3')
    assert len(repo.list()) == 1
    if limit == 60:
        from backend.app.runs.execution import execute_claimed_run
        from backend.app.runs.worker import RunWorker
        from backend.app.runs.status_projection import project_status
        worker = RunWorker(repository=repo, worker_id='review-recovery',
            execute=lambda r: execute_claimed_run(r, repository=repo),
            now=lambda: datetime.now(timezone.utc), heartbeat_seconds=60,
            project_status=lambda r: project_status(api[1]/'runs'/r.run_id, r))
        assert worker.run_once()
        state = resume_task(runtime, storage=tmp_path/'checkpoint', thread_id='audit',
            client=client(), llm=llm, clock=clock, wait=True,
            sleep=lambda n: ticks.__setitem__(0, ticks[0]+n))
        assert state['lifecycle']['status'] == 'completed', state['lifecycle']
        assert state['finalization']['selected_run_id'] == repo.list()[0].run_id
    count = len(wire.requests)
    assert resume_task(runtime, storage=tmp_path/'checkpoint', thread_id='audit',
        client=client(), llm=llm, clock=clock) == state
    assert len(wire.requests) == count == state['budget']['api_calls']['actual']


def test_uncached_submit_never_hides_http(api):
    from agent_poc.clients.autoai_client import AgentContractError
    wire, runtime, llm, client, dataset = components(api, 'logistic_regression')
    with pytest.raises(AgentContractError):
        client().submit_ml_experiment('uncached', model_type='logistic_regression')
    assert wire.requests == []

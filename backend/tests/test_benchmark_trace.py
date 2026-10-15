import json
from types import SimpleNamespace
from backend.app import benchmark_driver as driver
from agent_poc.orchestration.llm import LLMError, Proposal, TokenUsage


def test_numeric_call_ids_persist_semantic_trace_on_failure_and_success(tmp_path):
    request = driver.make_session_request(tmp_path, dataset_id="dataset", models=["cnn1d"],
        budget=1, role="development", session_key="trace-fixture")
    flow = driver.SessionFlow(tmp_path, SimpleNamespace(), request)
    flow.session = {"session_id": "session"}
    context = {"context_version": "agent-context-budget-v1", "allowed_actions": ["finalize_ml_session"],
               "bindings": {"session_id": "session", "selected_run_id": "owned-run"}}
    online = SimpleNamespace(ensure_online=lambda *a, **k: None, emit=lambda *a, **k: None)
    class Adapter:
        last_request_measurement = None
        last_semantic_trace = None
        attempts = 0
        def propose(self, phase, shown, **kwargs):
            self.attempts += 1
            kwargs["on_dispatch"]()
            self.last_semantic_trace = {"actual_request": {"phase": phase},
                "native_validation_status": "invalid" if self.attempts == 1 else "valid"}
            self.last_request_measurement = {"request_duration_seconds": .25, "phase": phase,
                "model_id": "qwen35_9b", "protocol": "json_action"}
            if self.attempts == 1:
                raise LLMError("llm_output_invalid", usage=TokenUsage(3, 4, 7, "known", 0))
            return Proposal("finalize_ml_session", context["bindings"], "fixture",
                usage=TokenUsage(5, 6, 11, "known", 0))
    adapter = Adapter()
    try:
        assert flow.lm("finalize", context, adapter=adapter, online=online, online_task_id="fixture").arguments == context["bindings"]
        traces = list((tmp_path / "semantic-traces").glob("*.json"))
        assert len(traces) == 2 and all(p.stem.isdigit() for p in traces)
        assert {json.loads(p.read_text())["native_validation_status"] for p in traces} == {"invalid", "valid"}
        assert flow.costs()["known_lm_tokens"] == 18
        flow.lm("finalize", context, adapter=adapter, online=online, online_task_id="fixture")
        assert adapter.attempts == 2
    finally:
        flow.close()

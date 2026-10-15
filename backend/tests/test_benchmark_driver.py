"""Offline driver tests: no real fit, provider request, or CUDA allocation."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backend.app import benchmark_driver as driver
from backend.app.benchmark_tracking import atomic_json
from backend.app.runs.repository import RunRepository


@pytest.fixture
def prepared_root(tmp_path, monkeypatch):
    root = tmp_path / "benchmark"
    (root / "approval").mkdir(parents=True)
    official = Path(__file__).resolve().parent / 'fixtures/benchmark_protocol_v3.json'
    (root / "approval" / "protocol.json").write_bytes(official.read_bytes())
    snapshot = root / "source"
    snapshot.mkdir()
    atomic_json(root / "approval" / "source-freeze.json",
                dict(source_digest="f" * 64, snapshot_path=str(snapshot), fixture_only=True))
    directory = root / "data"
    directory.mkdir()
    values = np.random.default_rng(5).normal(size=(40, 32))
    frame = pd.DataFrame(values, columns=[str(i) for i in range(32)])
    frame.insert(0, "Name", [f"name_{i}" for i in range(40)])
    frame.insert(0, "Sample_ID", [f"group_{i}" for i in range(40)])
    frame.insert(0, "Label", ["A"] * 20 + ["B"] * 20)
    frame.insert(0, "Index", list(range(1, 41)))
    path = directory / "main.csv"
    frame.to_csv(path, index=False)
    task = dict(task_id="toy", suite="test", family_id="test:toy", role="development",
                axis_kind="spectral", main_path=str(path), test_path=None)
    atomic_json(directory / "manifest.json", dict(status="complete", tasks=[task]))
    monkeypatch.setenv("AUTOAI_BENCHMARK_ROOT", str(root))
    return root


def limited_availability(model):
    return (model in {"logistic_regression", "cnn1d"}, "fixture_unavailable")


def test_requests_reuse_frozen_timestamps_and_hard_limits(tmp_path):
    kwargs = dict(dataset_id="dataset", models=["cnn1d"], budget=3,
                  role="locked", session_key="session-fixture")
    first = driver.make_session_request(tmp_path, **kwargs, now=100)
    second = driver.make_session_request(tmp_path, **kwargs, now=999)
    assert first == second
    from backend.app.agent.budget import BudgetPolicy
    policy = BudgetPolicy.from_dict(first["budget_policy"])
    assert policy.started_at == 100
    assert policy.limits["model_fits"] == 4
    assert policy.limits["training_epochs"] == 600
    assert policy.limits["input_tokens"] is None
    assert policy.limits["cached_tokens"] is None
    assert first["budget_awareness"] == "off"
    assert first["feedback_diagnosis"] == "off"


def test_rounds_only_never_freezes_sessions_without_source(prepared_root, monkeypatch):
    (prepared_root / "approval" / "source-freeze.json").unlink()
    def unexpected(*args, **kwargs):
        pytest.fail("round-only preparation must not create a Session")
    controller = driver.BenchmarkDriver(prepared_root, services_factory=unexpected)
    body = controller.prepare(rounds_only=True)
    assert len(body["rounds"]) == 5
    assert body["source_digest"] is None
    assert body["jobs"] == []
    assert not (controller.directory / "sessions").exists()
    with pytest.raises(RuntimeError, match="source-freeze"):
        controller.prepare()


def test_unordered_axis_excludes_convolution_models_before_selection(prepared_root):
    data = driver.read_json(prepared_root / "data" / "manifest.json")
    data["tasks"][0]["axis_kind"] = "feature_index"
    atomic_json(prepared_root / "data" / "manifest.json", data)
    body = driver.BenchmarkDriver(prepared_root).prepare(rounds_only=True)
    assert all(set(row["candidate_models"]) == set(driver.MODELS[:7]) for row in body["rounds"])


@pytest.fixture
def full_preparation(prepared_root):
    controller = driver.BenchmarkDriver(prepared_root, availability=limited_availability)
    body = controller.prepare()
    return controller, body


def test_preparation_freezes_real_recipes_and_precise_paired_counts(full_preparation):
    controller, body = full_preparation
    assert body["preparation_status"] == "sessions_frozen"
    assert body["pair_count"] == 10
    assert body["run_views"] == 20
    assert body["actual_fit_upper"] == 5 * (5 + 3)
    assert body["actual_epoch_upper"] == 5 * 600
    for row in body["rounds"]:
        assert row["eligible_models_by_budget"] == {"1": ["logistic_regression", "cnn1d"],
                                                   "3": ["logistic_regression", "cnn1d"]}
        assert row["lm_sessions"]["1"]["session_id"] != row["lm_sessions"]["3"]["session_id"]
        assert row["matrix_sessions"]["1"]["cnn1d"]["session_id"] != row["lm_sessions"]["1"]["session_id"]
        assert row["effective_candidates"]["cnn1d"] == {"1": 1, "3": 3}
    assert controller.prepare() == body
    assert controller.status()["job_counts"] == {"pending": 10}


def test_explicit_execution_shares_pair_bundle_but_not_budget_identity():
    row = dict(task_id="toy", suite="test", round_id="01", protocol_digest="a" * 64)
    initial = driver.execution_options("/tmp/toy", row, "f" * 64)
    actual = driver.execution_options("/tmp/toy", row, "f" * 64, "cnn1d")
    assert "execution_device" not in initial
    assert actual["execution_device"] == "cuda"
    assert actual["cpu_threads"] == 1
    assert actual["benchmark_context"]["tracking_task_id"] == "test:toy::ffffffffffff"
    assert actual["benchmark_context"]["bundle_id"] == driver.execution_options("/tmp/toy", row, "f" * 64, "cnn1d")["benchmark_context"]["bundle_id"]


def test_grouped_prediction_averages_probabilities_and_checks_coverage(tmp_path):
    path = tmp_path / "predictions.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Sample_ID", "true_label", "prob_A", "prob_B"])
        writer.writeheader()
        writer.writerows([dict(Sample_ID="group", true_label="A", prob_A=.8, prob_B=.2),
                          dict(Sample_ID="group", true_label="A", prob_A=.4, prob_B=.6)])
    rows = driver.grouped_predictions(path, ["A", "B"], ["group"])
    assert len(rows) == 1
    assert rows[0]["predicted_label"] == "A"
    assert rows[0]["probabilities"]["A"] == pytest.approx(.6)
    with pytest.raises(ValueError, match="coverage"):
        driver.grouped_predictions(path, ["A", "B"], ["another"])


def test_precise_claim_preserves_fifo_default(tmp_path):
    from backend.app.runs.contracts import Principal
    repository = RunRepository(tmp_path / "runs.sqlite3")
    repository.initialize()
    now = driver.utc_now()
    first = repository.create_queued(dataset_id="first", config={}, principal=Principal("owner", "tenant"))
    second = repository.create_queued(dataset_id="second", config={}, principal=Principal("owner", "tenant"))
    chosen = repository.claim_next(worker_id="worker", now=now, run_id=second.run_id)
    assert chosen.run_id == second.run_id
    assert repository.get(first.run_id).state == "queued"
    assert repository.claim_next(worker_id="worker", now=now).run_id == first.run_id


def test_unknown_api_dispatch_reconciles_own_session_without_replaying(full_preparation):
    controller, body = full_preparation
    row = body["rounds"][0]
    flow = controller.session_flow(row, 1, ["logistic_regression"], "matrix:logistic_regression")
    try:
        flow.create()
        def after_commit():
            raise OSError("simulated response failure")
        with pytest.raises(OSError):
            flow.api("uncertain_submit", after_commit)
        count = []
        response = flow.api("uncertain_submit", lambda: count.append("replayed"),
                            replay_check=lambda: {"run_id": "owned"})
        assert response == {"run_id": "owned"}
        assert count == []
    finally:
        flow.close()


def test_unknown_lm_send_never_replays_and_nonlm_cost_is_zero(full_preparation):
    controller, body = full_preparation
    row = body["rounds"][0]
    flow = controller.session_flow(row, 1, ["logistic_regression"], "matrix:logistic_regression")
    try:
        flow.create()
        assert flow.costs()["llm_calls"] == 0
        assert flow.costs()["known_lm_tokens"] == 0
        operation = driver.identity("lm", flow.request["client_request_id"], "submit")
        call_id = flow.calls.begin(operation_id=operation, kind="llm", name="submit", maximum=6,
            max_attempts=3, deadline=flow.policy.deadline_at, task_id=flow.policy.task_id,
            session_id=flow.session["session_id"])
        flow.calls.mark_dispatched(call_id)
        adapter = SimpleNamespace(propose=lambda *args, **kwargs: pytest.fail("unknown LM must not be replayed"))
        online = SimpleNamespace(ensure_online=lambda *args, **kwargs: pytest.fail("must fail before online work"))
        with pytest.raises(driver.UncertainOperation, match="no confirmed outcome"):
            flow.lm("submit", {}, adapter=adapter, online=online, online_task_id="tracking")
    finally:
        flow.close()


def test_claim_execute_settle_and_publication_contract(full_preparation, monkeypatch):
    controller, body = full_preparation
    row = body["rounds"][0]
    flow = controller.session_flow(row, 1, ["logistic_regression"], "matrix:logistic_regression",
        execution=driver.execution_options(controller.root, row, "f" * 64, "logistic_regression"))
    events = []
    from backend.app.agent import training_budget
    from backend.app.runs import guard, status_projection
    checked = SimpleNamespace(stage="pre_publish", status="passed", bindings={"owned": True})
    class MockExecution:
        def __init__(self, *, repository, run_dir, training_budget):
            self.repository, self.run_dir, self.budget = repository, run_dir, training_budget
        def execute(self, record, *, data_path):
            events.append(("execute", record.run_id))
            # Exercise actual hard-budget events without fitting any learner.
            self.budget.enter("fit-fixture", dimension="model_fits", kind="trial")
            self.budget.complete("fit-fixture")
            return dict(manifest_name="manifest.json", guard_report_id="mock-report")
    monkeypatch.setattr(guard, "check_outputs", lambda **kwargs: events.append(("publication", kwargs["usage"]["model_fits"])) or checked)
    monkeypatch.setattr(flow.services.runs, "get_guard_report", lambda *args, **kwargs: checked)
    def finish(run_id, **kwargs):
        events.append(("finish", run_id))
        # Guard internals are tested separately. This fixture isolates execution
        # ordering while preserving the Run claim/state transition.
        kwargs.pop("publication_report")
        # A frozen guard requires a full report; delegate the transition via
        # a narrow mock instead of weakening the real repository.
        return SimpleNamespace(run_id=run_id, state="succeeded")
    monkeypatch.setattr(flow.services.runs, "finish_success", finish)
    monkeypatch.setattr(status_projection, "project_status", lambda *args, **kwargs: None)
    online = SimpleNamespace(ensure_online=lambda *args, **kwargs: events.append(("online", None)))
    try:
        record = driver.execute_budget(flow, row, model="logistic_regression", online=online,
            online_task_id="fixture", execution_factory=MockExecution)
        assert record.state == "succeeded"
        assert [event[0] for event in events] == ["online", "execute", "publication", "finish"]
        assert events[2][1] == 1
        scoped = flow.services.agent.get_session(session_id=flow.session["session_id"], principal=flow.services.principal)
        assert len(scoped["experiments"]) == 1
        actual = flow.services.runs.get(record.run_id)
        binding = training_budget.load_training_binding(flow.services.sessions.database_path, actual,
                                                        runs_database=flow.services.runs.database_path)
        import sqlite3
        with sqlite3.connect(flow.services.sessions.database_path) as connection:
            status, = connection.execute("SELECT state FROM task_training_executions_v1 WHERE reservation_id=?",
                                         (binding.reservation_id,)).fetchone()
        assert status == "settled"
    finally:
        flow.close()


def test_source_versions_preserve_prior_work_and_share_resource_root(prepared_root):
    first = driver.BenchmarkDriver(prepared_root)
    marker = first.directory / "prior-record.json"
    atomic_json(marker, {"preserve": True})
    body = driver.read_json(prepared_root / "approval" / "source-freeze.json")
    body["source_digest"] = "e" * 64
    atomic_json(prepared_root / "approval" / "source-freeze.json", body)
    second = driver.BenchmarkDriver(prepared_root)
    assert first.directory != second.directory
    assert first.directory.name == "f" * 12
    assert second.directory.name == "e" * 12
    assert driver.read_json(marker) == {"preserve": True}
    assert first.root / "weights" == second.root / "weights"


def test_role_gate_rejects_locked_before_formal_development_completion(full_preparation):
    controller, manifest = full_preparation
    source = driver.source_reference(controller.root)
    with pytest.raises(RuntimeError, match="formal-freeze"):
        controller._require_locked(manifest, source)
    atomic_json(controller.root / "approval" / "formal-freeze.json",
                dict(source_digest=source["source_digest"], protocol_digest=manifest["protocol_digest"],
                     fixed_models={"1": {"model_id": "cnn1d"}}))
    with pytest.raises(RuntimeError, match="development matrix"):
        controller._require_locked(manifest, source)


def test_coffee_acceptance_is_separate_and_uses_exact_approved_processing(prepared_root):
    data = driver.read_json(prepared_root / "data" / "manifest.json")
    original = data["tasks"][0]
    frame = pd.read_csv(original["main_path"])
    main = prepared_root / "data" / "coffee-main.csv"
    test = prepared_root / "data" / "coffee-test.csv"
    frame.iloc[list(range(15)) + list(range(20, 35))].to_csv(main, index=False)
    frame.iloc[list(range(15, 20)) + list(range(35, 40))].to_csv(test, index=False)
    coffee = {**original, "task_id": "Coffee", "suite": "ucr85", "role": "excluded",
              "main_path": str(main), "test_path": str(test), "family_id": "ucr85:Coffee"}
    data["tasks"].append(coffee)
    atomic_json(prepared_root / "data" / "manifest.json", data)
    controller = driver.BenchmarkDriver(prepared_root)
    manifest = controller.prepare_acceptance()
    assert manifest["acceptance_only"] is True
    assert manifest["paper_evidence"] is False
    assert manifest["independence_status"] == "group_archive_case"
    assert manifest["actual_fit_upper"] == 3
    assert [job["budgets"] for job in manifest["jobs"]] == [[1], [1]]
    row = manifest["rounds"][0]
    settings = row["matrix_sessions"]["1"]
    assert settings["logistic_regression"]["recipe"]["preprocessing"]["normalization"] == "minmax"
    assert settings["cnn1d"]["recipe"]["preprocessing"]["normalization"] == "area"
    assert all(item["recipe"]["class_balance"] == "class_weight" for item in settings.values())
    assert not (controller.directory / "manifest.json").exists()


def test_lm_repairs_and_success_are_online_measured_and_durable(tmp_path):
    from agent_poc.orchestration.llm import LLMError, Proposal, TokenUsage
    request = driver.make_session_request(tmp_path, dataset_id="dataset", models=["cnn1d"],
        budget=1, role="development", session_key="fixture")
    flow = driver.SessionFlow(tmp_path, SimpleNamespace(), request)
    flow.session = {"session_id": "session"}
    context = {"context_version": "agent-context-budget-v1", "allowed_actions": ["finalize_ml_session"],
               "bindings": {"session_id": "session", "selected_run_id": "owned-run"}}
    events, attempts = [], []
    online = SimpleNamespace(ensure_online=lambda *args, **kwargs: events.append("online"),
                             emit=lambda task, kind, data: events.append((kind, data)))
    class Adapter:
        last_request_measurement = None
        def propose(self, phase, displayed, **kwargs):
            events.append("request")
            kwargs["on_dispatch"]()
            attempts.append(kwargs["repair_code"])
            self.last_request_measurement = {"request_duration_seconds": .25, "phase": phase,
                                             "model_id": "qwen35_9b", "protocol": "json_action"}
            if len(attempts) == 1:
                raise LLMError("llm_output_invalid", usage=TokenUsage(3, 4, 7, "known", 0))
            return Proposal("finalize_ml_session", context["bindings"], "fixture",
                            usage=TokenUsage(5, 6, 11, "known", 0))
    adapter = Adapter()
    try:
        proposal = flow.lm("finalize", context, adapter=adapter, online=online, online_task_id="fixture")
        assert proposal.arguments == context["bindings"]
        assert attempts == [None, "llm_output_invalid"]
        assert events[:2] == ["online", "request"]
        assert len([event for event in events if isinstance(event, tuple) and event[0] == "lm.request"]) == 2
        assert flow.costs()["known_lm_tokens"] == 18
        flow.lm("finalize", context, adapter=adapter, online=online, online_task_id="fixture")
        assert len(attempts) == 2
    finally:
        flow.close()


def test_worker_runs_pair_in_order_and_emits_test_metrics_without_real_fit(prepared_root, monkeypatch):
    from backend.app import execution_runtime
    monkeypatch.setattr(execution_runtime, "initialize_execution_runtime", lambda count: None)
    controller = driver.BenchmarkDriver(prepared_root)
    row = dict(task_id="fixture", suite="test", family_id="test:fixture", role="locked", round_id="01",
               protocol_digest="a" * 64, labels=["A", "B"], data_path="not-used",
               eligible_models_by_budget={"1": ["logistic_regression"], "3": ["logistic_regression"]},
               plan={"groups": {"test": ["gA", "gB"]}})
    row_path = controller.directory / "round.json"
    atomic_json(row_path, row)
    job_path = controller.directory / "jobs" / "fixture.json"
    atomic_json(job_path, dict(job_id="fixture", model_id="logistic_regression", source_digest="f" * 64,
               round_path=str(row_path), budgets=[1, 3], assigned_choices={"1": None, "3": None},
               cpu_threads=4, gpu_id=None, source_order=0))
    order, bundles, events = [], [], []
    online = SimpleNamespace(emit=lambda task, kind, data: events.append((kind, data)))
    controller.online = lambda: online
    run_root = controller.directory / "mock-runs"
    class Flow:
        def __init__(self, budget):
            self.request = {"budget": budget}
            self.directory = controller.directory / f"session-{budget}"
            self.services = SimpleNamespace(run_root=run_root,
                agent=SimpleNamespace(finalize_session=lambda **kwargs: {"state": "finalized"}), principal=None)
            self.session = {"session_id": f"session-{budget}"}
        def api(self, operation, function, **kwargs):
            return function()
        def costs(self):
            return {"llm_calls": 0, "known_lm_tokens": 0, "unknown_count": 0, "api_calls": 3}
        def close(self):
            pass
    def session(row, budget, models, kind, execution):
        bundles.append(execution["benchmark_context"]["bundle_id"])
        return Flow(budget)
    controller.session_flow = session
    def execute(flow, row, **kwargs):
        budget = flow.request["budget"]
        order.append(budget)
        run_id = f"run-{budget}"
        directory = run_root / run_id
        atomic_json(directory / "search_summary.json", {"selected": [{"selection_score": .75}],
                    "final_refit_count": 1 if budget == 1 else 0})
        atomic_json(directory / "search_trials.json", [] if budget == 1 else [{"cached": True, "source_fit_seconds": .4}])
        atomic_json(directory / "telemetry" / "experiment_summary.json", dict(model_fits_completed=2, model_fits_started=2, executed_epochs=0,
                    total_wall_seconds=.5, default_weight_ref=None, cached_seconds=.1,
                    cached_final_fit_count=0 if budget == 1 else 1, cached_final_fit_source_seconds=.2))
        with (directory / "predictions.csv").open("w") as handle:
            handle.write("Sample_ID,true_label,prob_A,prob_B\ngA,A,0.8,0.2\ngB,B,0.1,0.9\n")
        return SimpleNamespace(run_id=run_id, state="succeeded")
    monkeypatch.setattr(driver, "execute_budget", execute)
    result = driver.worker(prepared_root, job_path, driver=controller)
    assert result["status"] == "completed"
    assert order == [1, 3]
    assert bundles[0] == bundles[1]
    facts = controller.records()
    assert {fact["budget"] for fact in facts} == {1, 3}
    assert all(fact["cost"]["llm_calls"] == 0 for fact in facts)
    assert all(fact["metrics"]["test_balanced_accuracy"] == 1 for fact in facts)
    assert all(event[1]["validation_balanced_accuracy"] == .75 for event in events)
    driver.worker(prepared_root, job_path, driver=controller)
    assert order == [1, 3]


def test_development_fixed_choice_ignores_locked_scores_and_uses_declared_common_models(prepared_root):
    controller = driver.BenchmarkDriver(prepared_root)
    tasks = [dict(task_id="dev", suite="test", family_id="dev", role="development", labels=["A", "B"]),
             dict(task_id="locked", suite="test", family_id="locked", role="locked", labels=["A", "B"])]
    rounds = [dict(task_id=t["task_id"], round_id="01", eligible_models_by_budget={"1": ["cnn1d", "logistic_regression"],
                                                                              "3": ["cnn1d", "logistic_regression"]}) for t in tasks]
    jobs = [dict(task_id="dev", round_id="01", model_id=model, role="development", budgets=[1, 3]) for model in ("cnn1d", "logistic_regression")]
    atomic_json(controller.directory / "manifest.json", dict(tasks=tasks, rounds=rounds, jobs=jobs))
    for task in tasks:
        for model in ("cnn1d", "logistic_regression"):
            for budget in (1, 3):
                correct = (model == "logistic_regression") if task["role"] == "development" else (model == "cnn1d")
                rows = [dict(true_label="A", predicted_label="A" if correct else "B"),
                        dict(true_label="B", predicted_label="B" if correct else "A")]
                atomic_json(controller.directory / "records" / f'{task["task_id"]}-{model}-{budget}.json',
                            dict(task_id=task["task_id"], role=task["role"], round_id="01", model_id=model, budget=budget,
                                 status="completed", prediction_rows=rows))
    result = controller.development_fixed_models()
    assert result["1"]["model_id"] == "logistic_regression"
    assert result["3"]["model_id"] == "logistic_regression"


def test_cuda_gate_requires_complete_real_evidence_fields(prepared_root):
    controller = driver.BenchmarkDriver(prepared_root)
    with pytest.raises(RuntimeError, match="real GPU"):
        controller._require_cuda_acceptance([], "f" * 64)
    records = []
    for model in set(driver.MODELS) - driver.TRADITIONAL:
        first, second = prepared_root / "storage" / model / "first", prepared_root / "storage" / model / "second"
        atomic_json(first / "telemetry" / "cuda_explanation_acceptance.json",
                    dict(status="ready", sample_count=1, actual_device="cuda:0", dtype="torch.float32",
                         additional_fits=0, source_digest="f" * 64, model_id=model, source_run_id="b1"))
        atomic_json(second / "telemetry" / "checkpoint_reload.json",
                    dict(status="succeeded", loaded_device="cuda:0", dtype="torch.float32", additional_fits=0,
                         source_checkpoint={"sha256": "a" * 64}))
        atomic_json(second / "search_summary.json", {"cache_hit_count": 1})
        atomic_json(second / "search_trials.json", [{"trial_index": 0, "cached": True, "executed_epochs": 0}])
        records += [dict(task_id="fixture", round_id="01", model_id=model, role="development", budget=1,
                         run_id="b1", status="completed", artifact_directory=str(first), cost={"actual_fits": 1},
                         prediction_rows=[{"fixture_only": True}]),
                    dict(task_id="fixture", round_id="01", model_id=model, role="development", budget=3,
                         run_id="b3", status="completed", artifact_directory=str(second))]
    controller._require_cuda_acceptance(records, "f" * 64)
    missing = records[0]
    evidence_path = Path(missing["artifact_directory"]) / "telemetry" / "cuda_explanation_acceptance.json"
    body = driver.read_json(evidence_path)
    body["dtype"] = "torch.float16"
    atomic_json(evidence_path, body)
    with pytest.raises(RuntimeError, match="real GPU"):
        controller._require_cuda_acceptance(records, "f" * 64)


def test_controller_mock_runs_only_development_and_preserves_locked_queue(prepared_root, monkeypatch):
    events, launched, selections = [], [], []
    class Online:
        def start_task(self, task_id, config):
            events.append(("start", task_id))
        def emit(self, task_id, kind, data):
            events.append((kind, task_id))
        def finish_task(self, task_id, summary):
            assert "fit_duration_sum_seconds" in summary
            assert "wall_seconds" not in summary
            events.append(("finish", task_id))
        def wait_verified(self, task_id, timeout):
            return {"verified": True, "cloud_ref": "offline-fixture"}
    class Scheduler:
        def __init__(self, **kwargs):
            assert kwargs == {"gpu_ids": (1, 2, 3, 5), "cpu_tokens": 32, "allow_shared_gpu": False}
        def allocate(self, kind):
            assert kind == "cpu"
            return SimpleNamespace(cpu_threads=4, gpu_id=None, release=lambda: None)
        def register_pid(self, lease, pid):
            events.append(("pid", pid))
        def record_completion(self, lease, work, elapsed):
            assert elapsed > 0
        def metrics(self):
            return {"cpu/threads_in_use": 0}
        def close(self):
            pass
    controller = driver.BenchmarkDriver(prepared_root, online_factory=lambda root: Online(), scheduler_factory=Scheduler)
    tasks = [dict(task_id="dev", suite="test", family_id="dev", role="development", labels=["A", "B"]),
             dict(task_id="locked", suite="test", family_id="locked", role="locked", labels=["A", "B"])]
    rounds, jobs = [], []
    for task in tasks:
        row = {**task, "round_id": "01", "protocol_digest": "a" * 64,
               "eligible_models_by_budget": {"1": ["logistic_regression"], "3": ["logistic_regression"]}}
        path = controller.directory / "rounds" / (task["task_id"] + ".json")
        atomic_json(path, row)
        rounds.append(row)
        jobs.append(dict(job_id=task["task_id"], task_id=task["task_id"], suite="test", family_id=task["family_id"],
                         role=task["role"], round_id="01", model_id="logistic_regression", round_path=str(path),
                         budgets=[1, 3], kind="cpu", bundle_id=task["task_id"], source_order=len(jobs)))
    atomic_json(controller.directory / "manifest.json", dict(source_digest="f" * 64, protocol_digest="a" * 64,
        tasks=tasks, rounds=rounds, jobs=jobs, run_views=4))
    controller._tracking_config = lambda task, manifest, source: {"canonical_id": driver.canonical_task(task)}
    controller.select_round = lambda row: selections.append(row["task_id"]) or []
    def popen(command, **kwargs):
        job = driver.read_json(command[-1])
        launched.append(job["task_id"])
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["AUTOAI_BENCHMARK_ROOT"] == str(prepared_root)
        assert kwargs["start_new_session"] is True
        for budget in job["budgets"]:
            record = dict(task_id=job["task_id"], suite="test", family_id="dev", role="development", round_id="01",
                model_id="logistic_regression", budget=budget, status="completed", validation_score=.75,
                prediction_rows=[dict(sample_id="A", true_label="A", predicted_label="A", probabilities={"A": .8, "B": .2}),
                                 dict(sample_id="B", true_label="B", predicted_label="B", probabilities={"A": .2, "B": .8})],
                cost=dict(actual_fits=0, epochs=0, wall_seconds=.1, known_lm_tokens=0, unknown_count=0))
            path = controller.directory / "records" / (driver.identity("record", "dev", "01", "logistic_regression", budget) + ".json")
            atomic_json(path, record)
        atomic_json(Path(command[-1]).with_suffix(".result.json"), dict(status="completed", work_units=0))
        return SimpleNamespace(pid=987654, poll=lambda: 0)
    controller.popen = popen
    controller.run(role="development")
    state = controller.status()
    assert launched == ["dev"]
    assert selections == ["dev"]
    assert state["completed_scope"] == "development"
    assert state["full_goal_complete"] is False
    assert state["job_counts"] == {"completed": 1, "pending": 1}
    assert state["task_wall_seconds"]["dev"] >= 0
    assert not any("locked" in str(event) for event in events)



def _two_official_preparation_tasks(root):
    original = driver.read_json(root / 'data/manifest.json')['tasks'][0]
    frame = pd.read_csv(original['main_path'])
    main, test = root / 'data/official-main.csv', root / 'data/official-test.csv'
    frame.iloc[list(range(15)) + list(range(20, 35))].to_csv(main, index=False)
    frame.iloc[list(range(15, 20)) + list(range(35, 40))].to_csv(test, index=False)
    tasks = [{**original, 'task_id': name, 'family_id': 'test:' + name, 'role': role,
              'axis_kind': 'feature_index', 'main_path': str(main), 'test_path': str(test)}
             for name, role in [('first', 'development'), ('second', 'locked')]]
    data = {'status': 'complete', 'tasks': tasks}
    atomic_json(root / 'data/manifest.json', data)
    return data


def _prepared_api_receipts(controller):
    return {str(path.relative_to(controller.directory)): path.read_bytes()
            for path in (controller.directory / 'sessions').glob('*/api/*.json')}


def test_spawn_parallel_preparation_matches_serial_manifest_and_resumes_without_api_replay(prepared_root):
    data = _two_official_preparation_tasks(prepared_root)
    controller = driver.BenchmarkDriver(prepared_root)
    serial = controller.prepare(workers=1)
    receipts = _prepared_api_receipts(controller)
    bindings = {str(path): path.read_bytes() for path in (prepared_root / 'storage').rglob('benchmark-binding.json')}
    assert len(serial['rounds']) == 2
    # One round is complete; the other has durable Sessions but needs its atomic
    # round record recovered, exercising both continuation paths under spawn.
    (controller.directory / 'rounds' / (driver.identity('round', driver.canonical_task(serial['rounds'][1]), '01') + '.json')).unlink()
    parallel = controller.prepare(workers=2)
    assert parallel == serial
    assert parallel['data_manifest_digest'] == driver.digest(data)
    assert [job['job_id'] for job in parallel['jobs']] == [job['job_id'] for job in serial['jobs']]
    assert _prepared_api_receipts(controller) == receipts
    assert {str(path): path.read_bytes() for path in (prepared_root / 'storage').rglob('benchmark-binding.json')} == bindings
    assert driver.read_json(controller.directory / 'manifest.json') == serial
    assert all(not response['response'].get('experiments') for response in
               (json.loads(body) for body in receipts.values()) if response['response'].get('session_id'))


def test_parallel_merge_preserves_task_round_model_order_despite_reverse_completion(full_preparation, monkeypatch):
    import os
    from concurrent.futures import Future
    controller, serial = full_preparation
    parallel_controller = driver.BenchmarkDriver(controller.root)
    environment = {name: os.environ.get(name) for name in driver.PREPARATION_THREAD_VARIABLES}
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    seen = []

    class Executor:
        def __init__(self, **kwargs):
            assert kwargs['max_workers'] == 3
            assert kwargs['mp_context'].get_start_method() == 'spawn'
            assert kwargs['initializer'] is driver.initialize_preparation_worker
            assert all(os.environ[name] == '1' for name in driver.PREPARATION_THREAD_VARIABLES)
            assert os.environ.get('CUDA_VISIBLE_DEVICES') == cuda_visible
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def submit(self, function, *args):
            future = Future()
            future.set_result(function(*args))
            seen.append(future.result()['round_id'])
            return future

    monkeypatch.setattr(driver, 'ProcessPoolExecutor', Executor)
    monkeypatch.setattr(driver, 'as_completed', lambda futures: iter(reversed(list(futures))))
    assert parallel_controller.prepare(workers=3) == serial
    assert seen == ['01', '02', '03', '04', '05']
    assert {name: os.environ.get(name) for name in driver.PREPARATION_THREAD_VARIABLES} == environment
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == cuda_visible


def test_parallel_first_error_never_writes_complete_manifest(prepared_root, monkeypatch):
    from concurrent.futures import Future
    controller = driver.BenchmarkDriver(prepared_root)
    shutdown = []

    class Executor:
        def __init__(self, **kwargs):
            self.count = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def submit(self, *args):
            future = Future()
            future.set_exception(RuntimeError('round preparation failed'))
            return future
        def shutdown(self, **kwargs):
            shutdown.append(kwargs)

    monkeypatch.setattr(driver, 'ProcessPoolExecutor', Executor)
    with pytest.raises(RuntimeError, match='round preparation failed'):
        controller.prepare(workers=2)
    assert shutdown == [{'wait': True, 'cancel_futures': True}]
    assert not (controller.directory / 'manifest.json').exists()
    assert not (controller.directory / 'state.json').exists()


@pytest.mark.parametrize('workers', [0, 33, 1.5, True])
def test_preparation_workers_are_bounded(prepared_root, workers):
    with pytest.raises(ValueError, match='1 to 32'):
        driver.BenchmarkDriver(prepared_root).prepare(workers=workers)


def test_injected_preparation_factories_require_serial_workers(prepared_root):
    controller = driver.BenchmarkDriver(prepared_root, availability=limited_availability)
    with pytest.raises(ValueError, match='workers=1'):
        controller.prepare(workers=2)


def test_completed_round_resume_validates_full_source_and_partition_bindings(full_preparation):
    controller, body = full_preparation
    row = body['rounds'][0]
    base = {key: value for key, value in row.items() if key not in
            {'eligible_models_by_budget', 'exclusions', 'matrix_sessions', 'lm_sessions', 'effective_candidates', 'source_digest'}}
    assert controller._prepare_round(base, body['source_digest']) == row
    with pytest.raises(ValueError, match='binding changed'):
        controller._prepare_round({**base, 'dataset_sha256': '0' * 64}, body['source_digest'])
    with pytest.raises(ValueError, match='binding changed'):
        controller._prepare_round(base, 'e' * 64)



@pytest.fixture
def reusable_preparation(prepared_root):
    data = _two_official_preparation_tasks(prepared_root)
    data['tasks'] = data['tasks'][:1]
    atomic_json(prepared_root / 'data/manifest.json', data)
    old = driver.BenchmarkDriver(prepared_root, availability=limited_availability)
    manifest = old.prepare()
    source = driver.read_json(prepared_root / 'approval/source-freeze.json')
    source.update(source_digest='e' * 64, preparation_source_digest='f' * 64)
    atomic_json(prepared_root / 'approval/source-freeze.json', source)
    return old, manifest, driver.BenchmarkDriver(prepared_root, availability=limited_availability)


def test_explicit_pure_matrix_reuse_keeps_old_lm_failure_and_creates_fresh_current_source_lm(reusable_preparation, monkeypatch):
    import sqlite3
    old, original, current = reusable_preparation
    row = original['rounds'][0]
    old_lm_directory = Path(row['lm_sessions']['1']['directory'])
    with sqlite3.connect(old_lm_directory / 'calls.sqlite3') as connection:
        connection.execute("INSERT INTO orchestration_calls_v1(thread_id,operation_id,kind,name,status,started_at,ended_at,"
                           "input_tokens,output_tokens,total_tokens,error_code,request_outcome) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                           ('offline-old-lm', 'failed-lm', 'llm', 'submit', 'failed', 1, 2, 2141, 66, 2207,
                            'llm_output_invalid', 'response_received'))
    old_lm_bytes = (old_lm_directory / 'calls.sqlite3').read_bytes()
    old_matrix_api = {str(path): path.read_bytes() for handle in row['matrix_sessions']['1'].values()
                      for path in Path(handle['directory']).glob('api/*.json')}
    original_create = driver.SessionFlow.create
    created = []

    def create(flow):
        assert flow.directory.parent.parent == current.directory
        created.append(flow.directory)
        return original_create(flow)

    monkeypatch.setattr(driver.SessionFlow, 'create', create)
    manifest = current.prepare()
    assert len(created) == 2
    assert manifest['preparation_source_digest'] == 'f' * 64
    assert manifest['source_digest'] == 'e' * 64
    for key in ('data_manifest_digest', 'protocol_digest', 'pair_count', 'run_views', 'actual_fit_upper', 'actual_epoch_upper'):
        assert manifest[key] == original[key]
    reused = manifest['rounds'][0]
    for key in ('eligible_models_by_budget', 'effective_candidates', 'matrix_sessions', 'exclusions'):
        assert reused[key] == row[key]
    assert reused['source_digest'] == 'e' * 64
    assert all(reused['lm_sessions'][budget]['directory'] != row['lm_sessions'][budget]['directory'] for budget in ('1', '3'))
    assert all(job['job_id'] != old_job['job_id'] for job, old_job in zip(manifest['jobs'], original['jobs']))
    assert (old_lm_directory / 'calls.sqlite3').read_bytes() == old_lm_bytes
    assert {str(path): path.read_bytes() for handle in row['matrix_sessions']['1'].values()
            for path in Path(handle['directory']).glob('api/*.json')} == old_matrix_api
    flow = current.session_flow(reused, 1, ['logistic_regression'], 'matrix:logistic_regression',
        execution=driver.execution_options(current.root, reused, 'e' * 64, 'logistic_regression'))
    try:
        assert str(flow.directory) == reused['matrix_sessions']['1']['logistic_regression']['directory']
        assert flow.request['client_request_id'] == driver.identity('session', driver.canonical_task(row), '01', 1,
                                                                  'matrix:logistic_regression', 'f' * 64)
        assert flow.services.submissions.execution['benchmark_context']['source_digest'] == 'e' * 64
        assert flow.costs()['llm_calls'] == 0
    finally:
        flow.close()


@pytest.mark.parametrize('mismatch', ['source', 'protocol', 'data', 'spec', 'candidates'])
def test_explicit_matrix_reuse_rejects_all_frozen_binding_mismatches(reusable_preparation, mismatch):
    old, manifest, current = reusable_preparation
    if mismatch in {'source', 'protocol', 'data'}:
        key = {'source': 'source_digest', 'protocol': 'protocol_digest', 'data': 'data_manifest_digest'}[mismatch]
        manifest[key] = '0' * 64
    elif mismatch == 'spec':
        manifest['rounds'][0]['dataset_sha256'] = '0' * 64
    else:
        manifest['rounds'][0]['candidate_models'] = ['logistic_regression']
    atomic_json(old.directory / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='preparation reuse'):
        current.prepare()
    assert not (current.directory / 'manifest.json').exists()
    assert not (current.directory / 'sessions').exists()


@pytest.mark.parametrize('activity', ['run_file', 'uncertain_api', 'lm_journal', 'backend_run'])
def test_explicit_matrix_reuse_rejects_any_nonpure_matrix_session(reusable_preparation, activity):
    import sqlite3
    old, manifest, current = reusable_preparation
    row = manifest['rounds'][0]
    handle = row['matrix_sessions']['1']['logistic_regression']
    directory = Path(handle['directory'])
    request = driver.read_json(directory / 'request.json')
    if activity == 'run_file':
        atomic_json(directory / 'run.json', {'run_id': 'existing'})
    elif activity == 'uncertain_api':
        path = directory / 'api' / (driver.identity('op', 'create_session') + '.json')
        receipt = driver.read_json(path)
        receipt['status'] = 'dispatched'
        atomic_json(path, receipt)
    elif activity == 'lm_journal':
        with sqlite3.connect(directory / 'calls.sqlite3') as connection:
            connection.execute("INSERT INTO orchestration_calls_v1(thread_id,operation_id,kind,name,status,started_at) VALUES(?,?,?,?,?,?)",
                               (request['client_request_id'], 'unexpected-lm', 'llm', 'submit', 'failed', 1))
    else:
        storage = (current.root / 'storage' / driver.digest([row['suite'], str(row['task_id'])])[:16]
                   / driver.digest(str(row['round_id']))[:16] / row['plan']['plan_digest'][:16])
        from backend.app.runs.contracts import Principal
        RunRepository(storage / 'runs.sqlite3').create_queued(dataset_id=request['dataset_id'],
            config={'execution_budget_task_id': request['client_request_id']}, principal=Principal('benchmark-owner', 'benchmark-v3'))
    with pytest.raises(ValueError, match='preparation reuse'):
        current.prepare()
    assert not (current.directory / 'manifest.json').exists()
    assert not (current.directory / 'sessions').exists()



def test_spawn_workers_reuse_explicit_matrix_preparation_with_fresh_lm_sessions(reusable_preparation):
    old, original, current = reusable_preparation
    controller = driver.BenchmarkDriver(current.root)
    before = {str(path): path.read_bytes() for handle in original['rounds'][0]['matrix_sessions']['1'].values()
              for path in Path(handle['directory']).glob('api/*.json')}
    manifest = controller.prepare(workers=2)
    assert manifest['source_digest'] == 'e' * 64
    assert manifest['preparation_source_digest'] == 'f' * 64
    assert manifest['rounds'][0]['matrix_sessions'] == original['rounds'][0]['matrix_sessions']
    assert all(Path(handle['directory']).parent.parent == controller.directory
               for handle in manifest['rounds'][0]['lm_sessions'].values())
    assert manifest['pair_count'] == original['pair_count']
    assert manifest['data_manifest_digest'] == original['data_manifest_digest']
    assert {str(path): path.read_bytes() for handle in original['rounds'][0]['matrix_sessions']['1'].values()
            for path in Path(handle['directory']).glob('api/*.json')} == before


def test_new_acceptance_session_does_not_require_old_preparation_session(full_preparation):
    controller, manifest = full_preparation
    source = driver.read_json(controller.root / 'approval/source-freeze.json')
    source['preparation_source_digest'] = 'e' * 64
    atomic_json(controller.root / 'approval/source-freeze.json', source)
    row = {**manifest['rounds'][0], 'task_id': 'toy_acceptance', 'role': 'acceptance_only'}
    flow = controller.session_flow(row, 1, ['logistic_regression'], 'matrix:logistic_regression',
        fixed_processing={'logistic_regression': {'normalization': 'minmax', 'class_balance': 'class_weight'}})
    try:
        created = flow.create()
        locked = flow.inspect()['locked_config']
        assert flow.directory.parent == controller.directory / 'sessions'
        assert locked['fixed_processing']['logistic_regression']['normalization'] == 'minmax'
        assert (flow.directory / 'request.json').exists()
    finally:
        flow.close()

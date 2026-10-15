"""Durable driver for the approved benchmark; no fitting occurs in preparation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from copy import deepcopy
from functools import lru_cache
import sqlite3
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

MODELS = ("pls_da", "pca_lda", "logistic_regression", "svm", "random_forest", "xgboost",
          "pca_mlp", "cnn1d", "cnn1d_se", "resnet1d", "inception1d", "tcn1d", "cnn_transformer1d")
TRADITIONAL = frozenset(MODELS[:6])
BUDGETS = (1, 3)
DEVELOPMENT_ORDER = ("Beef", "BeetleFly", "BirdChicken", "Coffee", "ECGFiveDays", "Meat", "OliveOil",
                     "SonyAIBORobotSurface", "SonyAIBORobotSurfaceII", "Wine", "analcatdata_aids",
                     "appendicitis", "parity5", "prnn_crabs", "glass2")
DEVELOPMENT_INDEX = {task: index for index, task in enumerate(DEVELOPMENT_ORDER)}
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "work" / "benchmark-20261010"


class UncertainOperation(RuntimeError):
    """An earlier send may have reached the server; do not silently replay."""


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def atomic_json(path, body):
    from .benchmark_tracking import atomic_json as write
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, body)


def digest(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def identity(prefix, *parts):
    return prefix + "-" + digest(list(parts))[:48]


def utc_now():
    return datetime.now(timezone.utc)


@contextmanager
def exclusive(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another controller or worker owns this operation") from None
        yield


def source_reference(root):
    body = read_json(Path(root) / "approval" / "source-freeze.json")
    if body is None:
        raise RuntimeError("source-freeze.json is required before freezing Sessions or running jobs")
    value = body.get("source_digest")
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid frozen source digest")
    preparation = body.get("preparation_source_digest")
    if preparation is not None and (type(preparation) is not str or len(preparation) != 64
            or any(c not in "0123456789abcdef" for c in preparation) or preparation == value):
        raise ValueError("invalid explicit preparation source digest")
    if not Path(body["snapshot_path"]).is_dir():
        raise ValueError("frozen source snapshot is unavailable")
    return body


def canonical_task(row):
    return str(row["suite"]) + ":" + str(row["task_id"])


def tracking_id(row, source_digest):
    return canonical_task(row) + "::" + source_digest[:12]


def execution_options(root, row, source_digest, model=None, cpu_threads=None):
    context = dict(suite_root=str(Path(root).resolve()),
        bundle_id=identity("pair" if model else "selection", canonical_task(row), row["round_id"], model, source_digest),
        source_digest=source_digest, protocol_digest=row["protocol_digest"],
        tracking_task_id=tracking_id(row, source_digest))
    result = {"benchmark_context": context}
    if model is not None:
        result.update(execution_device="cpu" if model in TRADITIONAL else "cuda",
                      cpu_threads=cpu_threads if cpu_threads is not None else 4 if model in TRADITIONAL else 1)
    return result


def budget_policy(task_id, budget, started_at):
    from .agent.budget import BudgetPolicy
    # Queue waiting is not a scientific budget. Fits/epochs/calls remain bounded.
    return BudgetPolicy(task_id=task_id, started_at=started_at,
        work_deadline_at=started_at + 365 * 86400, deadline_at=started_at + 365 * 86400 + 3600,
        limits=dict(experiments=1, model_fits=2 if budget == 1 else 4,
                    training_epochs=200 if budget == 1 else 600, llm_calls=6, api_calls=64,
                    input_tokens=None, output_tokens=8192, cached_tokens=None,
                    output_repairs=4, network_retries=8),
        finalization_llm_calls=3, finalization_api_calls=9, max_repair_attempts=2).as_dict()


def make_session_request(directory, *, dataset_id, models, budget, role, session_key,
                         fixed_processing=None, now=None):
    path = Path(directory) / "request.json"
    existing = read_json(path)
    if existing is not None:
        if (existing["dataset_id"] != dataset_id or existing["allowed_models"] != list(models)
                or existing["max_trials"] != budget or existing["client_request_id"] != session_key):
            raise ValueError("persisted Session request differs from the approved job")
        return existing
    started = time.time() if now is None else now
    body = dict(dataset_id=dataset_id, selection_metric="balanced_accuracy", allowed_models=list(models),
        max_runs=1, seed=42, evaluation=dict(split_mode="stratified_holdout", split_train=8, split_valid=1, split_test=1),
        client_request_id=session_key, execution_profile="train-evidence-recipes-v1",
        protocol_revision="agent-recipes-revision-v7", decision_mode="recipe_id", processing_mode="fixed",
        search_mode="bounded", max_trials=budget,
        modules=["train_evidence", "legal_recipes", "bounded_hpo", "fail_fast_guard"],
        context_policy=dict(source_role="development" if role == "development" else "benchmark",
                            case_write=False, evidence=True, risks=True),
        budget_policy=budget_policy(session_key, budget, started), budget_awareness="off",
        fail_fast_guard="on", feedback_diagnosis="off")
    if fixed_processing is not None:
        body["fixed_processing"] = fixed_processing
    atomic_json(path, body)
    return body


class SessionFlow:
    """A scoped Session with frozen API/LM budgets and durable send boundaries."""
    def __init__(self, directory, services, request):
        from .agent.budget import BudgetPolicy
        from agent_poc.orchestration.persistence import CallJournal
        self.directory, self.services, self.request = Path(directory), services, request
        self.policy = BudgetPolicy.from_dict(request["budget_policy"])
        self.calls = CallJournal(self.directory / "calls.sqlite3", request["client_request_id"])
        self.calls.bind_budget(self.policy)
        self.session = read_json(self.directory / "session.json")

    def close(self):
        self.calls.connection.close()

    def api(self, operation, function, *, phase="work", replay_check=None):
        path = self.directory / "api" / (identity("op", operation) + ".json")
        state = read_json(path)
        if state and state["status"] == "confirmed":
            return state["response"]
        if state and state["status"] == "dispatched":
            if replay_check is None:
                raise UncertainOperation("unresolved API dispatch: " + operation)
            recovered = replay_check()
            if recovered is None:
                raise UncertainOperation("API outcome is still unknown: " + operation)
            # finish() is idempotent only for an unsettled row. A recorded local
            # exception can still accompany a committed backend transaction.
            row = next(r for r in self.calls.snapshot() if r["id"] == state["call_id"])
            if row["status"] == "dispatched":
                self.calls.finish(state["call_id"])
            atomic_json(path, {**state, "status": "confirmed", "response": recovered, "reconciled": True})
            return recovered
        call_id = self.calls.begin(operation_id=identity("api", operation), kind="api", name=operation,
            maximum=64, max_attempts=3, deadline=self.policy.deadline_at, task_id=self.policy.task_id,
            session_id=self.session["session_id"] if self.session else None, phase=phase)
        atomic_json(path, {"status": "prepared", "call_id": call_id})
        self.calls.mark_dispatched(call_id)
        atomic_json(path, {"status": "dispatched", "call_id": call_id})
        try:
            response = function()
        except BaseException as error:
            code = getattr(error, "code", type(error).__name__)
            self.calls.finish(call_id, error_code=code)
            atomic_json(path, {"status": "dispatched", "call_id": call_id, "error_code": code})
            raise
        self.calls.finish(call_id)
        atomic_json(path, {"status": "confirmed", "call_id": call_id, "response": response})
        return response

    def create(self):
        from .agent.contracts import CreateAgentSessionRequestV2
        svc = self.services
        def recover():
            row = svc.sessions.find_session_request_scoped(self.request["client_request_id"],
                payload_hash=None, principal=svc.principal)
            return None if row is None else svc.agent.get_session(session_id=row.session_id, principal=svc.principal)
        response = self.api("create_session", lambda: svc.agent.create_session(
            CreateAgentSessionRequestV2.model_validate(self.request), principal=svc.principal), replay_check=recover)
        self.session = response
        atomic_json(self.directory / "session.json", response)
        return response

    def inspect(self, operation="prepared_session", *, fresh=False, phase="work"):
        svc = self.services
        name = operation + (":" + str(time.time_ns()) if fresh else "")
        return self.api(name, lambda: svc.agent.get_session(session_id=self.session["session_id"],
                                                          principal=svc.principal), phase=phase)

    def lm(self, phase, context, *, adapter, online, online_task_id):
        from agent_poc.orchestration.llm import LLMError, StoredProposal, store_proposal
        operation = identity("lm", self.request["client_request_id"], phase)
        context_path = self.directory / (phase + "-context.json")
        frozen = read_json(context_path)
        if frozen is not None and frozen != context:
            raise ValueError("LM context differs from its frozen projection")
        atomic_json(context_path, context)
        stored = self.calls.proposal(operation)
        if stored is not None:
            return StoredProposal.model_validate_json(stored).bind(context)
        attempted = [row for row in self.calls.snapshot() if row["operation_id"] == operation]
        if any(row["status"] in {"dispatched", "prepared"} for row in attempted):
            raise UncertainOperation("an earlier LM request has no confirmed outcome")
        if attempted and attempted[-1]["error_code"] not in {"llm_output_invalid", "llm_output_too_large"}:
            raise UncertainOperation("a prior LM request failed with an unknown provider outcome")
        repairs = len(attempted)
        repair_code = attempted[-1]["error_code"] if attempted else None
        while repairs <= self.policy.max_repair_attempts:
            online.ensure_online(online_task_id, timeout=30)
            call_id = self.calls.begin(operation_id=operation, kind="llm", name=phase,
                maximum=6, max_attempts=3, deadline=self.policy.deadline_at, task_id=self.policy.task_id,
                session_id=self.session["session_id"], phase="finalization" if phase == "finalize" else "work")
            try:
                proposal = adapter.propose(phase, context, timeout_seconds=120, repair_code=repair_code,
                    on_dispatch=lambda: self.calls.mark_dispatched(call_id))
            except BaseException as error:
                usage = getattr(error, "usage", None)
                measured = getattr(adapter, "last_request_measurement", None)
                code = getattr(error, "code", "llm_unexpected")
                values = dict(input_tokens=getattr(usage, "prompt_tokens", None),
                    output_tokens=getattr(usage, "completion_tokens", None), total_tokens=getattr(usage, "total_tokens", None),
                    cached_tokens=getattr(usage, "cached_tokens", None), token_status=getattr(usage, "status", "unknown"))
                self.calls.finish(call_id, error_code=code, measurement=measured, **values)
                online.emit(online_task_id, "lm.request", dict(phase=phase, status="failed", error_code=code,
                    session_id=self.session["session_id"], measurement=measured, **values))
                if not isinstance(error, LLMError) or code not in {"llm_output_invalid", "llm_output_too_large"}:
                    raise
                repairs += 1; repair_code = code
                continue
            finally:
                semantic_trace = getattr(adapter, "last_semantic_trace", None)
                if semantic_trace is not None:
                    atomic_json(self.directory / "semantic-traces" / (str(call_id) + ".json"), semantic_trace)
            usage = proposal.usage
            values = dict(input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens,
                          total_tokens=usage.total_tokens, cached_tokens=usage.cached_tokens, token_status=usage.status)
            measured = getattr(adapter, "last_request_measurement", None)
            self.calls.finish(call_id, measurement=measured, proposal=store_proposal(proposal, context), **values)
            online.emit(online_task_id, "lm.request", dict(phase=phase, status="confirmed",
                session_id=self.session["session_id"], measurement=measured, **values))
            return proposal
        raise RuntimeError("LM format repair budget exhausted")

    def costs(self):
        requests = self.calls.export_llm_requests()
        known = [row for row in requests if row["total_tokens"] is not None]
        cached = [row["cached_tokens"] for row in requests if row["cached_tokens"] is not None]
        return dict(llm_calls=len(requests), api_calls=sum(row["kind"] == "api" for row in self.calls.snapshot()),
            known_lm_tokens=sum(row["total_tokens"] for row in known), unknown_count=len(requests) - len(known),
            cached_lm_tokens_estimate=sum(cached) if cached else 0 if not requests else None,
            lm_request_seconds=sum(row["request_duration_seconds"] or 0 for row in requests),
            llm_request_measurements=requests)


def grouped_predictions(path, labels, expected_groups=None):
    """Average all curve probabilities within Sample_ID before predicting."""
    from .probability_evaluation import grouped_probability_predictions
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return grouped_probability_predictions(csv.DictReader(handle), labels, expected_groups)


def classification_metrics(rows, labels):
    """Group-level Test metrics; this output never enters an LM context."""
    actual = Counter(row["true_label"] for row in rows)
    predicted = Counter(row["predicted_label"] for row in rows)
    correct = Counter(row["true_label"] for row in rows if row["true_label"] == row["predicted_label"])
    return dict(test_balanced_accuracy=sum(correct[label] / actual[label] for label in labels) / len(labels),
                test_accuracy=sum(correct.values()) / len(rows),
                test_macro_f1=sum(2 * correct[label] / (actual[label] + predicted[label]) for label in labels) / len(labels))


def fitted_costs(run_dir, telemetry):
    trials = read_json(Path(run_dir) / "search_trials.json", [])
    cached = [row for row in trials if row.get("cached")]
    final_cached = telemetry["cached_final_fit_count"]
    known_source_seconds = sum(row["source_fit_seconds"] for row in cached if row.get("source_fit_seconds") is not None) + telemetry["cached_final_fit_source_seconds"]
    all_source_times_known = all(row.get("source_fit_seconds") is not None for row in cached)
    return dict(actual_cached_fits=len(cached) + final_cached,
                actual_cached_seconds=telemetry.get("cached_seconds"),
                uncached_estimated_fits=telemetry["model_fits_completed"] + len(cached) + final_cached,
                uncached_estimated_seconds=telemetry["total_wall_seconds"] - telemetry["cached_seconds"] + known_source_seconds if all_source_times_known else None,
                known_cached_source_fit_seconds=known_source_seconds,
                uncached_estimate_scope="same-configuration source-fit timing; not an actual rerun")



PREPARATION_THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


@contextmanager
def preparation_cpu_environment():
    previous = {name: os.environ.get(name) for name in PREPARATION_THREAD_VARIABLES}
    try:
        for name in PREPARATION_THREAD_VARIABLES:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def initialize_preparation_worker():
    for name in PREPARATION_THREAD_VARIABLES:
        os.environ[name] = "1"


def prepare_round_worker(root, source_digest, row):
    """Spawn-safe preparation entry: freeze one round and never fit or call LM."""
    os.environ["AUTOAI_BENCHMARK_ROOT"] = str(Path(root).resolve())
    controller = preparation_worker_controller(root, source_digest)
    if source_reference(controller.root)["source_digest"] != source_digest:
        raise ValueError("preparation worker source binding changed")
    return controller._prepare_round(row, source_digest)



@lru_cache(maxsize=1)
def preparation_worker_controller(root, source_digest):
    controller = BenchmarkDriver(root)
    if source_reference(controller.root)['source_digest'] != source_digest:
        raise ValueError('preparation worker source binding changed')
    return controller


def prepared_round_jobs(root, directory, row, source_digest):
    jobs = []
    round_path = Path(directory) / "rounds" / (identity("round", canonical_task(row), row["round_id"]) + ".json")
    source_order = row["development_task_order"]
    effective = row["effective_candidates"]
    for model in row["eligible_models_by_budget"]["3"]:
        key = identity("job", canonical_task(row), row["round_id"], model, source_digest)
        count = effective[model]["3"]
        jobs.append(dict(job_id=key, task_id=row["task_id"], suite=row["suite"], family_id=row["family_id"],
            role=row["role"], round_id=row["round_id"], model_id=model, round_path=str(round_path),
            budgets=[1, 3], effective_candidates=effective[model],
            kind="cpu" if model in TRADITIONAL else "gpu", default_candidate_fits=1,
            pair_fit_upper=count + 2 if model in TRADITIONAL else count,
            pair_epoch_upper=0 if model in TRADITIONAL else count * 200,
            bundle_id=execution_options(root, row, source_digest, model)["benchmark_context"]["bundle_id"],
            source_order=source_order * 100 + int(row["round_id"]),
            development_task_order=source_order))
    return jobs

class BenchmarkDriver:
    def __init__(self, root=DEFAULT_ROOT, *, services_factory=None, rounds_builder=None,
                 availability=None, online_factory=None, scheduler_factory=None, popen=None):
        self.root = Path(root).resolve()
        reference = read_json(self.root / "approval" / "source-freeze.json")
        self.directory = (self.root / "driver" / "versions" / reference["source_digest"][:12]
                          if reference else self.root / "driver")
        self.directory.mkdir(parents=True, exist_ok=True)
        if reference:
            previous = read_json(self.directory / "version.json")
            if previous and previous["source_digest"] != reference["source_digest"]:
                raise ValueError("source version prefix collision")
            atomic_json(self.directory / "version.json", dict(source_digest=reference["source_digest"],
                        snapshot_path=reference["snapshot_path"]))
        self._preparation_reuse_cache = None
        self.services_factory, self.rounds_builder = services_factory, rounds_builder
        self.availability, self.online_factory = availability, online_factory
        self.scheduler_factory, self.popen = scheduler_factory, popen or subprocess.Popen

    def services(self, row, execution):
        from .benchmark_adapter import make_services
        return (self.services_factory or make_services)(self.root, row, execution)

    def online(self):
        from .benchmark_tracking import OnlineJournal
        return (self.online_factory or OnlineJournal)(self.root)

    def session_flow(self, row, budget, models, kind, execution=None, fixed_processing=None):
        source = source_reference(self.root)
        preparation_source = (source.get('preparation_source_digest', source['source_digest'])
                              if kind.startswith('matrix:') and row['role'] != 'acceptance_only'
                              else source['source_digest'])
        key = identity("session", canonical_task(row), row["round_id"], budget, kind, preparation_source)
        directory = (self.root / 'driver' / 'versions' / preparation_source[:12] / 'sessions' / key
                     if preparation_source != source['source_digest'] else self.directory / 'sessions' / key)
        if preparation_source != source['source_digest'] and not (directory / 'request.json').exists():
            raise ValueError('explicit preparation reuse Matrix Session request is missing')
        # Identity and budgets retain their original preparation provenance;
        # every execution and LM context is bound to the current source.
        services = self.services(row, execution or execution_options(self.root, row, source["source_digest"]))
        request = make_session_request(directory, dataset_id=services.dataset_id, models=models,
            budget=budget, role=row["role"], session_key=key, fixed_processing=fixed_processing)
        return SessionFlow(directory, services, request)

    def _preparation_reference(self):
        source = source_reference(self.root)
        preparation_source = source.get('preparation_source_digest')
        if preparation_source is None:
            return None
        if self._preparation_reuse_cache is None:
            from .benchmark_protocol import load_approved_protocol
            directory = self.root / 'driver' / 'versions' / preparation_source[:12]
            version, manifest = read_json(directory / 'version.json'), read_json(directory / 'manifest.json')
            protocol = load_approved_protocol(self.root / 'approval/protocol.json')
            data = read_json(self.root / 'data/manifest.json')
            if (version is None or version.get('source_digest') != preparation_source
                    or manifest is None or manifest.get('schema_version') != 'benchmark-job-manifest-v1'
                    or manifest.get('preparation_status') != 'sessions_frozen'
                    or manifest.get('source_digest') != preparation_source
                    or manifest.get('protocol_digest') != protocol['approval_digest']
                    or manifest.get('data_manifest_digest') != digest(data)
                    or manifest.get('pair_count') != len(manifest.get('jobs', []))
                    or manifest.get('run_views') != 2 * len(manifest.get('jobs', []))
                    or manifest.get('actual_fit_upper') != sum(job['pair_fit_upper'] for job in manifest.get('jobs', []))
                    or manifest.get('actual_epoch_upper') != sum(job['pair_epoch_upper'] for job in manifest.get('jobs', []))):
                raise ValueError('explicit preparation reuse source/protocol/data/complete-manifest binding differs')
            expected_jobs = [job for row in manifest['rounds']
                             for job in prepared_round_jobs(self.root, directory, row, preparation_source)]
            if manifest['jobs'] != expected_jobs:
                raise ValueError('explicit preparation reuse complete job coverage differs')
            self._preparation_reuse_cache = (preparation_source, directory, manifest,
                {(canonical_task(row), row['round_id']): row for row in manifest['rounds']})
        return self._preparation_reuse_cache

    def _verified_matrix_preparation(self, row):
        reference = self._preparation_reference()
        if reference is None:
            return None
        preparation_source, old_directory, manifest, old_rounds = reference
        old = old_rounds.get((canonical_task(row), row['round_id']))
        if (old is None or old.get('source_digest') != preparation_source
                or any(old.get(key) != value for key, value in row.items())):
            raise ValueError('explicit preparation reuse round specification or candidate models differ')
        storage = (self.root / 'storage' / digest([row['suite'], str(row['task_id'])])[:16]
                   / digest(str(row['round_id']))[:16] / row['plan']['plan_digest'][:16])
        with sqlite3.connect(f'file:{storage / "agent.sqlite3"}?mode=ro', uri=True) as agent, \
                sqlite3.connect(f'file:{storage / "runs.sqlite3"}?mode=ro', uri=True) as runs:
            for budget in BUDGETS:
                models = old['eligible_models_by_budget'][str(budget)]
                if set(old['matrix_sessions'][str(budget)]) != set(models):
                    raise ValueError('explicit preparation reuse matrix coverage differs')
                for model in models:
                    handle = old['matrix_sessions'][str(budget)][model]
                    key = identity('session', canonical_task(row), row['round_id'], budget,
                                   'matrix:' + model, preparation_source)
                    directory = old_directory / 'sessions' / key
                    if handle['directory'] != str(directory) or (directory / 'run.json').exists():
                        raise ValueError('explicit preparation reuse Matrix Session has Run activity or wrong identity')
                    request = read_json(directory / 'request.json')
                    session = read_json(directory / 'session.json')
                    if (request is None or session is None or request['client_request_id'] != key
                            or request['allowed_models'] != [model] or request['max_trials'] != budget
                            or session['session_id'] != handle['session_id']):
                        raise ValueError('explicit preparation reuse Matrix request/session binding differs')
                    operations = {'create_session', 'prepared_session'}
                    expected_api = {identity('op', operation) + '.json' for operation in operations}
                    if {path.name for path in (directory / 'api').glob('*.json')} != expected_api:
                        raise ValueError('explicit preparation reuse Matrix Session has non-preparation API activity')
                    for operation in operations:
                        receipt = read_json(directory / 'api' / (identity('op', operation) + '.json'))
                        if (receipt['status'] != 'confirmed'
                                or receipt['response']['session_id'] != handle['session_id']):
                            raise ValueError('explicit preparation reuse requires fully confirmed preparation APIs')
                    prepared_receipt = read_json(directory / 'api' / (identity('op', 'prepared_session') + '.json'))
                    locked = prepared_receipt['response']['locked_config']
                    preparation = locked['preparation']
                    if (locked['allowed_models'] != [model] or locked['budget_policy'] != request['budget_policy']
                            or preparation['catalog']['catalog_digest'] != handle['catalog_digest']
                            or handle['recipe'] not in preparation['catalog']['recipes']
                            or preparation['search_plans'][handle['recipe']['search_plan_digest']]['effective_trials']
                               != old['effective_candidates'][model][str(budget)]):
                        raise ValueError('explicit preparation reuse native catalog or budget binding differs')
                    with sqlite3.connect(f'file:{directory / "calls.sqlite3"}?mode=ro', uri=True) as calls:
                        journal = calls.execute('SELECT kind,name,status,run_id FROM orchestration_calls_v1').fetchall()
                    if (len(journal) != 2 or {item[1] for item in journal} != operations
                            or any(kind != 'api' or state != 'confirmed' or run_id is not None
                                   for kind, name, state, run_id in journal)):
                        raise ValueError('explicit preparation reuse Matrix Session has LM, submission, or uncertain API activity')
                    current = agent.execute('SELECT state,selected_run_id,dataset_sha256,budget_task_id,terminated_at '
                                            'FROM agent_sessions_v1 WHERE session_id=?', (handle['session_id'],)).fetchone()
                    reservations, = agent.execute('SELECT COUNT(*) FROM agent_experiment_reservations_v1 '
                                                   'WHERE session_id=?', (handle['session_id'],)).fetchone()
                    training, = agent.execute('SELECT COUNT(*) FROM task_training_executions_v1 WHERE task_id=?', (key,)).fetchone()
                    run_count, = runs.execute("SELECT COUNT(*) FROM runs WHERE json_extract(config_json, '$.execution_budget_task_id')=?",
                                             (key,)).fetchone()
                    if (current != ('open', None, row['dataset_sha256'], key, None)
                            or reservations or training or run_count):
                        raise ValueError('explicit preparation reuse Matrix Session has backend Run/submission/training activity')
        return old

    def _prepare_round(self, row, source_digest):
        """Freeze one round sequentially, or recover its completed atomic binding."""
        round_path = self.directory / "rounds" / (identity("round", canonical_task(row), row["round_id"]) + ".json")
        with exclusive(round_path.with_suffix(".prepare.lock")):
            previous = read_json(round_path)
            if previous is not None:
                if (previous.get("source_digest") != source_digest
                        or any(previous.get(key) != value for key, value in row.items())):
                    raise ValueError("frozen round eligibility or Session binding changed")
                return previous
            from .model_catalog import model_availability
            candidates = row["candidate_models"]
            reused = self._verified_matrix_preparation(row)
            if reused is not None:
                eligible, exclusions, sessions, effective = (deepcopy(reused[key]) for key in
                    ('eligible_models_by_budget', 'exclusions', 'matrix_sessions', 'effective_candidates'))
            else:
                eligible, exclusions, sessions, effective = {}, {}, {}, {}
                for budget in BUDGETS:
                    accepted, rejected, handles = [], {}, {}
                    for model in candidates:
                        available, reason = (self.availability or model_availability)(model)
                        if not available:
                            rejected[model] = "unavailable: " + str(reason)
                            continue
                        flow = self.session_flow(row, budget, [model], "matrix:" + model)
                        try:
                            flow.create()
                            prepared = flow.inspect()["locked_config"]["preparation"]
                            recipes = [r for r in prepared["catalog"]["recipes"] if r["model_id"] == model]
                            if len(recipes) != 1:
                                raise ValueError("fixed processing must yield exactly one recipe per model")
                            plan = prepared["search_plans"][recipes[0]["search_plan_digest"]]
                            effective.setdefault(model, {})[str(budget)] = plan["effective_trials"]
                            handles[model] = dict(directory=str(flow.directory), session_id=flow.session["session_id"],
                                recipe=recipes[0], catalog_digest=prepared["catalog"]["catalog_digest"])
                            accepted.append(model)
                        except Exception as error:
                            if getattr(error, "code", None) not in {"agent_model_unavailable", "agent_preparation_failed"}:
                                raise
                            rejected[model] = getattr(error, "code")
                        finally:
                            flow.close()
                    if not accepted:
                        raise RuntimeError("no eligible models for approved task/round: " + canonical_task(row))
                    eligible[str(budget)], exclusions[str(budget)], sessions[str(budget)] = accepted, rejected, handles
                if eligible["1"] != eligible["3"]:
                    raise ValueError("eligibility differs between paired budgets")
            lm_sessions = {}
            for budget in BUDGETS:
                flow = self.session_flow(row, budget, eligible[str(budget)], "A0" if budget == 1 else "A1")
                try:
                    flow.create()
                    prepared = flow.inspect()["locked_config"]["preparation"]
                    if sorted(r["model_id"] for r in prepared["catalog"]["recipes"]) != sorted(eligible[str(budget)]):
                        raise ValueError("LM catalog differs from shared matrix eligibility")
                    lm_sessions[str(budget)] = dict(directory=str(flow.directory), session_id=flow.session["session_id"])
                finally:
                    flow.close()
            row.update(eligible_models_by_budget=eligible, exclusions=exclusions, matrix_sessions=sessions,
                       lm_sessions=lm_sessions, effective_candidates=effective, source_digest=source_digest)
            round_path = self.directory / "rounds" / (identity("round", canonical_task(row), row["round_id"]) + ".json")
            atomic_json(round_path, row)
            return row

    def prepare(self, *, rounds_only=False, workers=1):
        if type(workers) is not int or not 1 <= workers <= 32:
            raise ValueError("preparation workers must be an integer from 1 to 32")
        if workers > 1 and not rounds_only and any(factory is not None for factory in
                (self.services_factory, self.rounds_builder, self.availability)):
            raise ValueError("injected preparation factories require workers=1")
        with exclusive(self.directory / "prepare.lock"):
            return self._prepare(rounds_only=rounds_only, workers=workers)

    def _prepare(self, *, rounds_only, workers):
        from .benchmark_protocol import load_approved_protocol, prepare_task_rounds, UNORDERED_MODELS
        protocol = load_approved_protocol(self.root / "approval" / "protocol.json")
        data = read_json(self.root / "data" / "manifest.json")
        if not data or data["status"] != "complete":
            raise RuntimeError("the full data manifest must be terminal before preparation")
        tasks = [task for task in data["tasks"] if task["role"] in {"development", "locked"}]
        if len({task["task_id"] for task in tasks}) != len(tasks):
            raise ValueError("report requires globally unique task_id")
        tasks.sort(key=lambda task: (task["role"] != "development",
            DEVELOPMENT_INDEX.get(task["task_id"], len(DEVELOPMENT_ORDER)) if task["role"] == "development" else 0,
            task["suite"], task["task_id"]))
        source = None if rounds_only else source_reference(self.root)
        data_digest = digest(data)
        specifications, task_specs = [], []
        for task_index, task in enumerate(tasks):
            source_order = DEVELOPMENT_INDEX.get(task["task_id"], len(DEVELOPMENT_ORDER) + task_index)
            built = (self.rounds_builder or prepare_task_rounds)(task, self.root)
            indices = []
            for original in built:
                row = {**original, "protocol_digest": protocol["approval_digest"], "development_task_order": source_order}
                if task["axis_kind"] not in {"feature_index", "spectral", "time_series"}:
                    raise ValueError("unapproved feature axis kind")
                row["candidate_models"] = [model for model in MODELS
                    if task["axis_kind"] != "feature_index" or model in UNORDERED_MODELS]
                indices.append(len(specifications))
                specifications.append(row)
            task_specs.append((task, built[0]["labels"], indices))
        round_keys = [(canonical_task(row), row["round_id"]) for row in specifications]
        if len(set(round_keys)) != len(round_keys):
            raise ValueError("approved round specifications must be unique")
        reference = None if rounds_only else self._preparation_reference()
        if reference is not None:
            old_rounds = reference[3]
            if len(specifications) != len(reference[2]['rounds']) or set(round_keys) != set(old_rounds) or any(
                    any(old_rounds[(canonical_task(row), row['round_id'])].get(key) != value
                        for key, value in row.items()) for row in specifications):
                raise ValueError('explicit preparation reuse complete round specifications or candidate models differ')
        if rounds_only:
            rounds = [{**row, "session_freeze_status": "awaiting_source_freeze"} for row in specifications]
        elif workers == 1:
            rounds = [self._prepare_round(row, source["source_digest"]) for row in specifications]
        else:
            # Children use spawn and one CPU thread. GPU visibility is preserved,
            # so preparation observes the same hardware capabilities as serial.
            completed = {}
            with preparation_cpu_environment():
                with ProcessPoolExecutor(max_workers=workers,
                        mp_context=multiprocessing.get_context("spawn"),
                        initializer=initialize_preparation_worker) as executor:
                    futures = {executor.submit(prepare_round_worker, str(self.root), source["source_digest"], row): index
                               for index, row in enumerate(specifications)}
                    try:
                        for future in as_completed(futures):
                            completed[futures[future]] = future.result()
                    except BaseException:
                        for future in futures:
                            future.cancel()
                        executor.shutdown(wait=True, cancel_futures=True)
                        raise
            rounds = [completed[index] for index in range(len(specifications))]
        report_tasks, jobs = [], []
        for task, labels, indices in task_specs:
            task_rounds = []
            for index in indices:
                row = rounds[index]
                if rounds_only:
                    task_rounds.append(dict(round_id=row["round_id"], candidate_models=row["candidate_models"]))
                else:
                    task_rounds.append(dict(round_id=row["round_id"], eligible_models_by_budget=row["eligible_models_by_budget"]))
                    jobs.extend(prepared_round_jobs(self.root, self.directory, row, source["source_digest"]))
            report_tasks.append(dict(task_id=task["task_id"], suite=task["suite"], family_id=task["family_id"],
                role=task["role"], labels=labels, rounds=task_rounds))
        if digest(read_json(self.root / "data" / "manifest.json")) != data_digest:
            raise ValueError("full data manifest changed during preparation")
        if source is not None and source_reference(self.root) != source:
            raise ValueError("source freeze changed during preparation")
        if load_approved_protocol(self.root / "approval" / "protocol.json")["approval_digest"] != protocol["approval_digest"]:
            raise ValueError("approved protocol changed during preparation")
        body = dict(schema_version="benchmark-job-manifest-v1", protocol_digest=protocol["approval_digest"],
            data_manifest_digest=data_digest, source_digest=source["source_digest"] if source else None,
            preparation_status="rounds_only" if rounds_only else "sessions_frozen",
            tasks=report_tasks, rounds=rounds, jobs=jobs, pair_count=len(jobs), run_views=2 * len(jobs),
            actual_fit_upper=sum(job["pair_fit_upper"] for job in jobs),
            actual_epoch_upper=sum(job["pair_epoch_upper"] for job in jobs))
        if source is not None and source.get("preparation_source_digest") is not None:
            body["preparation_source_digest"] = source["preparation_source_digest"]
        path = (self.root / "driver" / "rounds-only.json") if rounds_only else self.directory / "manifest.json"
        previous = read_json(path)
        if not rounds_only and previous is not None and previous != body:
            raise ValueError("prepared manifest differs; do not silently refreeze")
        atomic_json(path, body)
        self._state(preparation=body["preparation_status"], phase="prepared", job_counts={"pending": len(jobs)})
        return body

    def _state(self, **changes):
        from .benchmark_tracking import cloud_required
        if not cloud_required(self.root):
            changes.update(recording_mode='local_only', cloud_verified=False, cloud_requirement_waived_by_user=True)
        current = read_json(self.directory / "state.json", {})
        clocks = [read_json(path) for path in (self.directory / "task-clocks").glob("*.json")]
        current.update(changes, updated_at=utc_now().isoformat(),
                       task_wall_seconds={clock["task_id"]: clock.get("task_wall_seconds", time.time() - clock["started_at"])
                                          for clock in clocks})
        atomic_json(self.directory / "state.json", current)
        if self.directory != self.root / "driver":
            atomic_json(self.root / "driver" / "state.json", {**current, "active_version": self.directory.name,
                        "version_directory": str(self.directory)})
        return current

    def status(self):
        manifest = read_json(self.directory / "manifest.json", {})
        counts = Counter()
        for job in manifest.get("jobs", []):
            state = read_json(self.directory / "jobs" / (job["job_id"] + ".state.json"), {})
            counts[state.get("status", "pending")] += 1
        return {**read_json(self.directory / "state.json", {}), "job_counts": dict(counts),
                "total_jobs": len(manifest.get("jobs", [])), "run_views": manifest.get("run_views", 0)}

    def records(self):
        return [read_json(path) for path in sorted((self.directory / "records").glob("*.json"))]

    def choices(self):
        return [read_json(path) for path in sorted((self.directory / "choices").glob("*.json"))]

    def report(self):
        from .benchmark_report import summarize, write_report
        manifest = read_json(self.directory / "manifest.json")
        if manifest is None:
            raise RuntimeError("Session preparation is required for report coverage")
        task_ids = {task["task_id"] for task in manifest["tasks"]}
        records = [record for record in self.records() if record["task_id"] in task_ids]
        choices = [choice for choice in self.choices() if choice["task_id"] in task_ids
                   and (Path(choice["session_directory"]) / "finalized.json").exists()]
        driver_state = {**read_json(self.directory / "state.json", {}),
                        "task_clocks": [read_json(path) for path in (self.directory / "task-clocks").glob("*.json")]}
        body = summarize(records, choices, manifest["tasks"], driver_state=driver_state)
        output = self.root / "reports" / source_reference(self.root)["source_digest"][:12]
        write_report(body, output)
        frozen = read_json(self.root / "approval" / "formal-freeze.json")
        if frozen and all(item.get("available") for item in body["fixed_models"].values()) and frozen["fixed_models"] != body["fixed_models"]:
            raise ValueError("completed report fixed models differ from the pre-locked formal freeze")
        return body

    def _tracking_config(self, task, manifest, source):
        path = self.directory / "tracking-config" / (identity("task", task["suite"], task["task_id"]) + ".json")
        existing = read_json(path)
        if existing is not None:
            if existing["source_digest"] != source["source_digest"]:
                raise ValueError("tracking config source binding changed")
            return existing
        rounds = [row for row in manifest["rounds"] if row["task_id"] == task["task_id"]]
        from importlib.metadata import version, PackageNotFoundError
        packages = {}
        for name in ("torch", "numpy", "scikit-learn", "xgboost", "swanlab"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        nvidia = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        config = dict(canonical_id=canonical_task(task), role=task["role"], suite=task["suite"],
            protocol_digest=manifest["protocol_digest"], source_digest=source["source_digest"],
            source_snapshot=source["snapshot_path"], seed=42,
            budgets=sorted({b for job in manifest["jobs"] for b in job["budgets"]}), knowledge="off",
            memory="off", feedback_diagnosis="off", budget_awareness="off",
            partitions=[dict(round_id=row["round_id"], dataset_sha256=row["dataset_sha256"],
                plan_digest=row["plan"]["plan_digest"], partition_digest=row["plan"]["partition_digest"],
                row_counts={name: len(indices) for name, indices in row["plan"]["indices"].items()},
                group_counts={name: len(groups) for name, groups in row["plan"]["groups"].items()},
                eligibility=row["eligible_models_by_budget"]) for row in rounds],
            llm_config=dict(model="qwen35_9b", protocol="json_action", prompt_version="agent-decision-budget-v1",
                            temperature=0, top_p=1, max_tokens=1024, timeout_seconds=120),
            native_model_parameters={}, native_processing={}, environment=dict(hostname=socket.gethostname(), platform=platform.platform(),
                python=sys.version, executable=sys.executable, packages=packages, nvidia_inventory=nvidia.stdout.strip(),
                nvidia_inventory_status=nvidia.returncode, gpu_ids=[1, 2, 3, 5], cpu_thread_limit=32))
        for row in rounds:
            for model, handle in row["matrix_sessions"]["1"].items():
                state = read_json(Path(handle["directory"]) / "api" / (identity("op", "prepared_session") + ".json"))
                config["native_model_parameters"].setdefault(row["round_id"], {})[model] = state["response"]["locked_config"]["capability_snapshot"]["model_configs"][model]
                config["native_processing"].setdefault(row["round_id"], {})[model] = state["response"]["locked_config"]["fixed_processing"][model]
        atomic_json(path, config)
        return config

    def _adapter(self):
        os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",localhost,127.0.0.1"
        from agent_poc.orchestration.llm import LLMConfig
        from backend.app.benchmark_llm import BenchmarkLLMAdapter
        from backend.app.benchmark_protocol import runtime_policies, load_approved_protocol
        protocol = load_approved_protocol(self.root / "approval" / "protocol.json")
        return BenchmarkLLMAdapter(LLMConfig(base_url="http://127.0.0.1:8101/v1", model="qwen35_9b", protocol="json_action",
            prompt_version="agent-decision-budget-v1", temperature=0, top_p=1, max_tokens=1024, timeout=120), token=None,
            decision_prompt_version=runtime_policies(protocol)['decision_prompt_version'])

    def select_round(self, row, *, adapter=None):
        from agent_poc.orchestration.projection import recipe_selection_context
        source = source_reference(self.root)
        choices = []
        for budget in BUDGETS:
            path = self.directory / "choices" / (identity("choice", row["task_id"], row["round_id"], budget) + ".json")
            previous = read_json(path)
            if previous is not None:
                choices.append(previous)
                continue
            flow = self.session_flow(row, budget, row["eligible_models_by_budget"][str(budget)], "A0" if budget == 1 else "A1")
            try:
                flow.create()
                locked = flow.inspect()["locked_config"]
                task = dict(selection_metric="balanced_accuracy", seed=42, allowed_models=flow.request["allowed_models"],
                    evaluation_config=locked["evaluation_config"], decision_mode="recipe_id", processing_mode="fixed",
                    fixed_processing=locked["fixed_processing"], search_mode="bounded", max_trials=budget, budget_awareness="off")
                context = recipe_selection_context(task=task, session_id=flow.session["session_id"],
                    preparation=locked["preparation"], context_policy=dict(evidence=True, risks=True,
                    projection="agent-context-budget-v1"), model_configs=locked["capability_snapshot"]["model_configs"])
                proposal = flow.lm("submit", context, adapter=adapter or self._adapter(), online=self.online(),
                                   online_task_id=tracking_id(row, source["source_digest"]))
                if proposal.tool_name != "submit_ml_experiment" or proposal.arguments.get("session_id") != flow.session["session_id"]:
                    raise ValueError("LM choice has an invalid Session binding")
                recipe = next(recipe for recipe in locked["preparation"]["catalog"]["recipes"]
                              if recipe["recipe_id"] == proposal.arguments["recipe_id"])
                choice = dict(task_id=row["task_id"], round_id=row["round_id"], budget=budget, model_id=recipe["model_id"],
                    session_id=flow.session["session_id"], session_directory=str(flow.directory),
                    experiment_arguments=proposal.arguments, rationale=proposal.rationale)
                if choice["model_id"] not in row["eligible_models_by_budget"][str(budget)]:
                    raise ValueError("LM chose a model outside shared eligibility")
                atomic_json(path, choice)
                choices.append(choice)
            finally:
                flow.close()
        return choices

    def _finalize_choice(self, row, choice):
        from agent_poc.orchestration.projection import finalization_context
        source = source_reference(self.root)
        path = Path(choice["session_directory"])
        if read_json(path / "finalized.json"):
            return
        execution = execution_options(self.root, row, source["source_digest"], choice["model_id"])
        flow = self.session_flow(row, choice["budget"], row["eligible_models_by_budget"][str(choice["budget"])],
                                 "A0" if choice["budget"] == 1 else "A1", execution=execution)
        try:
            flow.create()
            info = read_json(path / "run.json")
            if info is None:
                raise UncertainOperation("selected Agent Session has no durable Run binding")
            svc, run_id = flow.services, info["run_id"]
            feedback = flow.api("feedback", lambda: svc.agent.get_feedback(session_id=flow.session["session_id"],
                run_id=run_id, principal=svc.principal), phase="finalization")
            if feedback["validation"]["status"] != "ready":
                raise RuntimeError("selected Agent Run has no usable Validation feedback")
            context = finalization_context(task=dict(selection_metric="balanced_accuracy", seed=42,
                evaluation_config=dict(mode="stratified_holdout", train_weight=8,
                                                                     validation_weight=1, heldout_weight=1)),
                session_id=flow.session["session_id"], run_id=run_id, validation=feedback["validation"],
                validation_score=feedback["validation_score"], allowed_actions=feedback["allowed_actions"],
                context_version="agent-context-budget-v1", budget_awareness="off")
            proposal = flow.lm("finalize", context, adapter=self._adapter(), online=self.online(),
                              online_task_id=tracking_id(row, source["source_digest"]))
            if proposal.tool_name != "finalize_ml_session" or proposal.arguments != context["bindings"]:
                raise ValueError("LM finalization differs from its own Session/Run")
            def recover():
                session = svc.agent.get_session(session_id=flow.session["session_id"], principal=svc.principal)
                return session if session["state"] == "finalized" and session["selected_run_id"] == run_id else None
            response = flow.api("finalize", lambda: svc.agent.finalize_session(session_id=flow.session["session_id"],
                selected_run_id=run_id, principal=svc.principal), phase="finalization", replay_check=recover)
            atomic_json(path / "finalized.json", response)
            record_path = self.directory / "records" / (identity("record", row["task_id"], row["round_id"], choice["model_id"], choice["budget"]) + ".json")
            record = read_json(record_path)
            if record is not None:
                record["cost"].update(flow.costs())
                atomic_json(record_path, record)
        finally:
            flow.close()


    def prepare_acceptance(self):
        """Two approved Coffee functionality cases; no independence claim."""
        from .benchmark_protocol import prepare_task_rounds, load_approved_protocol
        source = source_reference(self.root)
        protocol = load_approved_protocol(self.root / "approval" / "protocol.json")
        data = read_json(self.root / "data" / "manifest.json")
        coffee = next(task for task in data["tasks"] if task["suite"] == "ucr85" and task["task_id"] == "Coffee")
        task = {**coffee, "task_id": "Coffee_functional_acceptance", "role": "acceptance_only",
                "family_id": "group_archive_case", "independence_status": "group_archive_case"}
        rows = prepare_task_rounds(task, self.root)
        if len(rows) != 1:
            raise ValueError("Coffee technical acceptance requires the archived official Test")
        row = {**rows[0], "protocol_digest": protocol["approval_digest"], "source_digest": source["source_digest"],
               "eligible_models_by_budget": {"1": ["logistic_regression", "cnn1d"]},
               "independence_status": "group_archive_case", "paper_evidence": False}
        settings = {"logistic_regression": dict(normalization="minmax", class_balance="class_weight"),
                    "cnn1d": dict(normalization="area", class_balance="class_weight")}
        handles = {}
        for model, processing in settings.items():
            flow = self.session_flow(row, 1, [model], "matrix:" + model, fixed_processing={model: processing})
            try:
                flow.create()
                prepared = flow.inspect()["locked_config"]["preparation"]
                recipe = prepared["catalog"]["recipes"][0]
                plan = prepared["search_plans"][recipe["search_plan_digest"]]
                if recipe["model_id"] != model or plan["effective_trials"] != 1:
                    raise ValueError("functional acceptance differs from its single approved configuration")
                handles[model] = dict(directory=str(flow.directory), session_id=flow.session["session_id"], recipe=recipe,
                                      catalog_digest=prepared["catalog"]["catalog_digest"])
            finally:
                flow.close()
        row["matrix_sessions"] = {"1": handles}
        round_path = self.directory / "rounds" / (identity("round", canonical_task(row), row["round_id"]) + ".json")
        atomic_json(round_path, row)
        jobs = [dict(job_id=identity("job", canonical_task(row), row["round_id"], model, source["source_digest"]),
                     task_id=row["task_id"], suite=row["suite"], family_id=row["family_id"], role="acceptance_only",
                     round_id=row["round_id"], model_id=model, round_path=str(round_path), budgets=[1],
                     kind="cpu" if model in TRADITIONAL else "gpu", pair_fit_upper=2 if model in TRADITIONAL else 1,
                     pair_epoch_upper=0 if model in TRADITIONAL else 200, source_order=index,
                     bundle_id=execution_options(self.root, row, source["source_digest"], model)["benchmark_context"]["bundle_id"])
                for index, model in enumerate(settings)]
        body = dict(schema_version="benchmark-functional-acceptance-v1", source_digest=source["source_digest"],
                    protocol_digest=protocol["approval_digest"], preparation_status="sessions_frozen",
                    acceptance_only=True, paper_evidence=False, independence_status="group_archive_case",
                    tasks=[dict(task_id=row["task_id"], suite=row["suite"], family_id=row["family_id"],
                                role=row["role"], labels=row["labels"])], rounds=[row], jobs=jobs,
                    pair_count=2, run_views=2, actual_fit_upper=3, actual_epoch_upper=200)
        path = self.directory / "acceptance-manifest.json"
        old = read_json(path)
        if old is not None and old != body:
            raise ValueError("frozen functionality acceptance differs")
        atomic_json(path, body)
        return body

    def development_fixed_models(self):
        """Choose F0/F1 before reading any locked task prediction or score."""
        manifest = read_json(self.directory / "manifest.json")
        common = {str(b): set(MODELS) for b in BUDGETS}
        for row in manifest["rounds"]:
            for budget in BUDGETS:
                common[str(budget)] &= set(row["eligible_models_by_budget"][str(budget)])
        development = [t for t in manifest["tasks"] if t["role"] == "development"]
        records = {(r["task_id"], r["round_id"], r["model_id"], r["budget"]): r for r in self.records()
                   if r["role"] == "development"}
        for job in manifest["jobs"]:
            if job["role"] != "development":
                continue
            for budget in job["budgets"]:
                record = records.get((job["task_id"], job["round_id"], job["model_id"], budget))
                if record is None or record["status"] != "completed":
                    raise RuntimeError("fixed model selection requires the complete development matrix")
        result = {}
        for budget in BUDGETS:
            candidates = sorted(common[str(budget)])
            if not candidates or not development:
                raise RuntimeError("no development pool or common eligible fixed model")
            means = {}
            for model in candidates:
                scores = []
                for task in development:
                    rows = [prediction for key, record in records.items()
                            if key[0] == task["task_id"] and key[2:] == (model, budget)
                            for prediction in record["prediction_rows"]]
                    counts = Counter(p["true_label"] for p in rows)
                    correct = Counter(p["true_label"] for p in rows if p["true_label"] == p["predicted_label"])
                    if set(counts) != set(task["labels"]):
                        raise ValueError("development OOF labels are incomplete")
                    from .benchmark_report import _confusion, _metrics
                    scores.append(_metrics(_confusion(rows, task["labels"]))["balanced_accuracy"])
                import statistics
                means[model] = statistics.fmean(scores)
            selected = sorted(candidates, key=lambda model: (-means[model], model))[0]
            result[str(budget)] = dict(budget=budget, available=True, model_id=selected,
                development_mean_balanced_accuracy=means[selected], candidate_models=candidates,
                candidate_development_means=means, reason=None)
        atomic_json(self.directory / "development-fixed-models.json", result)
        return result

    def _require_cuda_acceptance(self, records, source_digest):
        """Gate on real current-version GPU evidence, never unit-test outcomes."""
        accepted = set()
        pairs = {(r["task_id"], r["round_id"], r["model_id"], r["budget"]): r for r in records}
        for record in records:
            model = record["model_id"]
            if record["role"] != "development" or record["budget"] != 1 or model in TRADITIONAL or record["status"] != "completed":
                continue
            paired = pairs.get((record["task_id"], record["round_id"], model, 3))
            if paired is None or paired["status"] != "completed":
                continue
            first = Path(record["artifact_directory"]).resolve()
            second = Path(paired["artifact_directory"]).resolve()
            if self.root not in first.parents or self.root not in second.parents:
                raise ValueError("GPU acceptance artifact path is outside the benchmark root")
            explanation = read_json(first / "telemetry" / "cuda_explanation_acceptance.json", {})
            reload = read_json(second / "telemetry" / "checkpoint_reload.json", {})
            search = read_json(second / "search_summary.json", {})
            trials = read_json(second / "search_trials.json", {})
            trial_rows = trials.get("trials", []) if isinstance(trials, dict) else trials
            default = next((t for t in trial_rows if t["trial_index"] == 0), None)
            if (record["cost"]["actual_fits"] >= 1 and record["prediction_rows"]
                    and explanation.get("status") == "ready" and explanation.get("sample_count") == 1
                    and explanation.get("actual_device", "").startswith("cuda")
                    and explanation.get("dtype") == "torch.float32" and explanation.get("additional_fits") == 0
                    and explanation.get("source_digest") == source_digest
                    and explanation.get("model_id") == model and explanation.get("source_run_id") == record["run_id"]
                    and reload.get("status") == "succeeded" and reload.get("loaded_device", "").startswith("cuda")
                    and reload.get("dtype") == "torch.float32" and reload.get("additional_fits") == 0
                    and len(reload.get("source_checkpoint", {}).get("sha256", "")) == 64
                    and search.get("cache_hit_count", 0) >= 1 and default is not None
                    and default.get("cached") is True and default.get("executed_epochs") == 0):
                accepted.add(model)
        required = set(MODELS) - TRADITIONAL
        if accepted != required:
            raise RuntimeError("real GPU fit/prediction/checkpoint/FP32 explanation evidence is missing for " + ",".join(sorted(required - accepted)))

    def _require_locked(self, manifest, source):
        """Locked tasks are gated by complete current-version development evidence."""
        frozen = read_json(self.root / "approval" / "formal-freeze.json")
        if not frozen or (frozen.get("source_digest") != source["source_digest"]
                          or frozen.get("protocol_digest") != manifest["protocol_digest"]
                          or not frozen.get("fixed_models")):
            raise RuntimeError("locked tasks require formal-freeze.json with current source/protocol/fixed models")
        records = {(r["task_id"], r["round_id"], r["model_id"], r["budget"]): r for r in self.records()}
        for job in manifest["jobs"]:
            if job["role"] != "development":
                continue
            for budget in job["budgets"]:
                record = records.get((job["task_id"], job["round_id"], job["model_id"], budget))
                if record is None or record["status"] != "completed":
                    raise RuntimeError("development matrix is incomplete; locked tasks remain gated")
        if frozen["fixed_models"] != self.development_fixed_models():
            raise ValueError("formal fixed models differ from development-only selection")
        self._require_cuda_acceptance(list(records.values()), source["source_digest"])
        online = self.online()
        for task in manifest["tasks"]:
            if task["role"] == "development":
                online.wait_verified(tracking_id(task, source["source_digest"]), timeout=0)
        acceptance = [r for r in self.records() if r["task_id"] == "Coffee_functional_acceptance"]
        if len(acceptance) != 2 or any(r["status"] != "completed" for r in acceptance):
            raise RuntimeError("both approved functional cases must finish before locked tasks")
        accepted = dict(task_id="Coffee_functional_acceptance", suite="ucr85")
        online.wait_verified(tracking_id(accepted, source["source_digest"]), timeout=0)
        return frozen

    def run(self, *, resume=False, acceptance_only=False, role="development"):
        if role not in {"development", "locked", "all"}:
            raise ValueError("role must be development, locked or all")
        from .benchmark_resources import ResourceScheduler, WeightStore
        with exclusive(self.root / "driver" / "controller.lock"):
            manifest = read_json(self.directory / ("acceptance-manifest.json" if acceptance_only else "manifest.json"))
            if manifest is None:
                manifest = self.prepare_acceptance() if acceptance_only else self.prepare()
            source = source_reference(self.root)
            if source["source_digest"] != manifest["source_digest"]:
                raise ValueError("prepared jobs belong to another source snapshot")
            if not acceptance_only and role in {"locked", "all"}:
                self._require_locked(manifest, source)
            if not acceptance_only and role != "all":
                manifest = {**manifest, "tasks": [t for t in manifest["tasks"] if t["role"] == role],
                            "rounds": [r for r in manifest["rounds"] if r["role"] == role],
                            "jobs": [j for j in manifest["jobs"] if j["role"] == role]}
            online, store = self.online(), WeightStore(self.root / "weights")
            self._state(phase="selecting", controller_pid=os.getpid(), resume=resume,
                execution_scope="acceptance_only" if acceptance_only else role,
                scope_task_count=len(manifest["tasks"]), scope_job_count=len(manifest["jobs"]),
                retry_policy=dict(resource_failure_same_configuration_max_retries=1,
                    automatic_retry=False, unresolved_resource_failure="needs_attention",
                    reason="requires verified prior execution and same-configuration Session recovery"))
            for task in manifest["tasks"]:
                config = self._tracking_config(task, manifest, source)
                clock_path = self.directory / "task-clocks" / (identity("task", canonical_task(task)) + ".json")
                if read_json(clock_path) is None:
                    atomic_json(clock_path, dict(started_at=time.time(), task_id=task["task_id"], canonical_id=canonical_task(task)))
                online.start_task(tracking_id(task, source["source_digest"]), config)
            choices = []
            for row in ([] if acceptance_only else manifest["rounds"]):
                self._state(active_selection=dict(task_id=row["task_id"], round_id=row["round_id"]))
                choices.extend(self.select_round(row))
            selected = {(c["task_id"], c["round_id"], c["model_id"]) for c in choices}
            jobs = sorted(manifest["jobs"], key=lambda j: ((j["task_id"], j["round_id"], j["model_id"]) not in selected,
                                                          j["source_order"]))
            scheduler = (self.scheduler_factory or ResourceScheduler)(gpu_ids=(1, 2, 3, 5), cpu_tokens=32, allow_shared_gpu=False)
            active, pending, finalize_errors = {}, [], []
            jobs_directory = self.directory / "jobs"
            jobs_directory.mkdir(exist_ok=True)
            # Orphan workers retain all resources conservatively: no new jobs
            # are admitted until their observed processes are gone.
            for job in jobs:
                path = jobs_directory / (job["job_id"] + ".state.json")
                state = read_json(path, {})
                result_path = jobs_directory / (job["job_id"] + ".result.json")
                if result_path.exists():
                    result = read_json(result_path)
                    atomic_json(path, {**state, "status": result["status"], "result_path": str(result_path)})
                elif state.get("status") in {"running", "starting"}:
                    pid = state.get("pid")
                    while pid and _pid_alive(pid) and not result_path.exists():
                        self._state(phase="waiting_prior_worker", active_pid=pid, job_id=job["job_id"])
                        time.sleep(5)
                    if result_path.exists():
                        result = read_json(result_path)
                        atomic_json(path, {**state, "status": result["status"], "result_path": str(result_path)})
                    else:
                        atomic_json(path, {**state, "status": "failed",
                                          "failure_reason": "interrupted_worker_outcome_unknown"})
                elif state.get("status") not in {"completed", "failed"}:
                    pending.append(job)
            last_metrics = last_cleanup = 0.0
            try:
                while pending or active:
                    for job in list(pending):
                        lease = scheduler.allocate(job["kind"])
                        if lease is None:
                            continue
                        assigned = {str(budget): next((c for c in choices if c["task_id"] == job["task_id"]
                            and c["round_id"] == job["round_id"] and c["budget"] == budget
                            and c["model_id"] == job["model_id"]), None) for budget in BUDGETS}
                        payload = {**job, "assigned_choices": assigned, "source_digest": source["source_digest"],
                                   "cpu_threads": lease.cpu_threads, "gpu_id": lease.gpu_id}
                        job_path = jobs_directory / (job["job_id"] + ".json")
                        atomic_json(job_path, payload)
                        state_path = jobs_directory / (job["job_id"] + ".state.json")
                        atomic_json(state_path, dict(status="starting", job_id=job["job_id"], started_at=time.time(),
                                                    cpu_threads=lease.cpu_threads, gpu_id=lease.gpu_id))
                        environment = dict(os.environ)
                        environment["AUTOAI_BENCHMARK_ROOT"] = str(self.root)
                        environment["CUDA_VISIBLE_DEVICES"] = "" if lease.gpu_id is None else str(lease.gpu_id)
                        environment["NO_PROXY"] = environment.get("NO_PROXY", "") + ",localhost,127.0.0.1"
                        snapshot = Path(source["snapshot_path"])
                        environment["PYTHONPATH"] = str(snapshot) + os.pathsep + environment.get("PYTHONPATH", "")
                        command = [sys.executable, str(snapshot / "scripts" / "benchmark.py"), "--root", str(self.root),
                                   "_worker", "--job", str(job_path)]
                        log = (jobs_directory / (job["job_id"] + ".log")).open("ab")
                        try:
                            process = self.popen(command, cwd=snapshot, env=environment, stdout=log,
                                                 stderr=subprocess.STDOUT, start_new_session=True)
                        except BaseException:
                            log.close()
                            lease.release()
                            raise
                        scheduler.register_pid(lease, process.pid)
                        atomic_json(state_path, dict(status="running", job_id=job["job_id"], pid=process.pid,
                            started_at=time.time(), cpu_threads=lease.cpu_threads, gpu_id=lease.gpu_id))
                        active[job["job_id"]] = (process, lease, log, time.monotonic(), job)
                        pending.remove(job)
                    for key, (process, lease, log, started, job) in list(active.items()):
                        code = process.poll()
                        if code is None:
                            continue
                        log.close()
                        result_path = jobs_directory / (key + ".result.json")
                        result = read_json(result_path, dict(status="failed", failure_reason="worker_exit_without_result", work_units=0))
                        atomic_json(jobs_directory / (key + ".state.json"), {**result, "exit_code": code, "pid": process.pid})
                        scheduler.record_completion(lease, result.get("work_units", 0), max(.001, time.monotonic() - started))
                        lease.release()
                        active.pop(key)
                        row = read_json(job["round_path"])
                        if result_path.exists():
                            # Both budget outcomes are terminal and necessary
                            # checkpoints remain pinned until this pair boundary.
                            store.release_group(job["bundle_id"])
                        for choice in choices:
                            if (choice["task_id"], choice["round_id"], choice["model_id"]) == (job["task_id"], job["round_id"], job["model_id"]):
                                try:
                                    self._finalize_choice(row, choice)
                                except Exception as error:
                                    finalize_errors.append(dict(task_id=choice["task_id"], round_id=choice["round_id"],
                                        budget=choice["budget"], error_code=getattr(error, "code", type(error).__name__)))
                    now = time.monotonic()
                    if now - last_metrics >= 5:
                        metrics = scheduler.metrics()
                        self._state(phase="training", job_counts=self.status()["job_counts"], resources=metrics,
                                    pending_jobs=len(pending), active_jobs=len(active), finalization_errors=finalize_errors)
                        for task in manifest["tasks"]:
                            online.emit(tracking_id(task, source["source_digest"]), "scheduler.sample", metrics)
                        last_metrics = now
                    if now - last_cleanup >= 60:
                        store.cleanup()
                        last_cleanup = now
                    if pending or active:
                        time.sleep(.5)
            finally:
                scheduler.close()
            # Resume also completes a previously interrupted finalization stage
            # using its stored proposal, never repeating an unknown LM dispatch.
            for row in manifest["rounds"]:
                for choice in choices:
                    if (choice["task_id"], choice["round_id"]) == (row["task_id"], row["round_id"]):
                        try:
                            self._finalize_choice(row, choice)
                        except Exception as error:
                            item = dict(task_id=choice["task_id"], round_id=choice["round_id"],
                                        budget=choice["budget"], error_code=getattr(error, "code", type(error).__name__))
                            if item not in finalize_errors:
                                finalize_errors.append(item)
            self._state(phase="reporting", job_counts=self.status()["job_counts"])
            if acceptance_only:
                report = dict(status="completed", acceptance_only=True, paper_evidence=False,
                    independence_status="group_archive_case", records=[r for r in self.records()
                    if r["task_id"] == "Coffee_functional_acceptance"])
                atomic_json(self.root / "reports" / source["source_digest"][:12] / "functional-acceptance.json", report)
            else:
                report = self.report()
            expected = {(j["task_id"], j["round_id"], j["model_id"], b) for j in manifest["jobs"] for b in j["budgets"]}
            present = {(r["task_id"], r["round_id"], r["model_id"], r["budget"]) for r in self.records()}
            missing_records = sorted(expected - present)
            failed_records = [r for r in self.records() if r["task_id"] in {t["task_id"] for t in manifest["tasks"]}
                              and r["status"] != "completed"]
            for task in manifest["tasks"]:
                records = [r for r in self.records() if r["task_id"] == task["task_id"]]
                if (any(r["status"] != "completed" for r in records)
                        or any(key[0] == task["task_id"] for key in missing_records)
                        or any(e["task_id"] == task["task_id"] for e in finalize_errors)):
                    online.emit(tracking_id(task, source["source_digest"]), "task.attention_required",
                                dict(records_failed=sum(r["status"] != "completed" for r in records),
                                     retry_limit=1, automatic_retry=False))
                    continue
                clock_path = self.directory / "task-clocks" / (identity("task", canonical_task(task)) + ".json")
                clock = read_json(clock_path)
                summary = dict(canonical_id=canonical_task(task), role=task["role"],
                    records_completed=sum(r["status"] == "completed" for r in records),
                    records_failed=sum(r["status"] == "failed" for r in records),
                    actual_fits=sum(r.get("cost", {}).get("actual_fits", 0) for r in records),
                    epochs=sum(r.get("cost", {}).get("epochs", 0) for r in records),
                    fit_duration_sum_seconds=sum(r.get("cost", {}).get("wall_seconds", 0) for r in records),
                    task_work_wall_seconds=time.time() - clock["started_at"])
                task_id = tracking_id(task, source["source_digest"])
                summary_path = self.directory / "task-summaries" / (identity("task", canonical_task(task)) + ".json")
                existing_summary = read_json(summary_path)
                if existing_summary is None:
                    atomic_json(summary_path, summary)
                else:
                    summary = existing_summary
                online.finish_task(task_id, summary)
                verification = online.wait_verified(task_id, timeout=120)
                if "cloud_verified_at" not in clock:
                    ended = time.time()
                    atomic_json(clock_path, {**clock, "cloud_verified_at": ended,
                                "task_wall_seconds": ended - clock["started_at"], "verification": verification})
            if failed_records or finalize_errors or missing_records:
                self._state(phase="needs_attention", failed_records=len(failed_records), missing_records=missing_records,
                            finalization_errors=finalize_errors, cloud_verified=False,
                            resource_failures=[r for r in failed_records if r.get("resource_failure")])
                raise UncertainOperation("failed or unresolved work is retained; resource retries require verified recovery")
            self._state(phase="finished", cloud_verified=True, job_counts=self.status()["job_counts"],
                        completed_scope="acceptance_only" if acceptance_only else role,
                        full_goal_complete=False)
            return report


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False


def execute_budget(flow, row, *, model, choice=None, online, online_task_id, execution_factory=None):
    """Submit, claim, renew, settle and publish exactly this Session's own Run."""
    from .agent.contracts import CreateKnowledgeExperimentRequest
    from .agent.training_budget import TrainingBudget, load_training_binding
    from .runs.execution import TrainingExecution
    from .runs.worker import LeaseGuard
    from .runs.guard import check_outputs, GuardError
    from .runs.status_projection import project_status
    svc = flow.services
    flow.create()
    session_id = flow.session["session_id"]
    catalog = flow.inspect()["locked_config"]["preparation"]["catalog"]
    recipe = next(r for r in catalog["recipes"] if r["model_id"] == model)
    arguments = dict(choice["experiment_arguments"] if choice else dict(recipe_id=recipe["recipe_id"],
        catalog_digest=catalog["catalog_digest"], recipe_digest=recipe["recipe_digest"], knowledge_refs=[]))
    if arguments.pop("session_id", session_id) != session_id:
        raise ValueError("experiment belongs to a different Session")
    arguments.pop("decision_mode", None)
    arguments["client_request_id"] = identity("experiment", flow.request["client_request_id"])
    arguments.setdefault("knowledge_refs", [])
    if arguments["recipe_id"] != recipe["recipe_id"]:
        raise ValueError("selected recipe differs from the owned Session catalog")
    arguments.update(catalog_digest=catalog["catalog_digest"], recipe_digest=recipe["recipe_digest"])
    if arguments["knowledge_refs"]:
        raise ValueError("primary benchmark knowledge must remain off")
    atomic_json(flow.directory / "experiment-request.json", arguments)
    def reconcile_submission():
        existing = svc.agent.get_session(session_id=session_id, principal=svc.principal)["experiments"]
        return existing[0] if len(existing) == 1 and existing[0].get("run_id") else None
    response = flow.api("create_experiment", lambda: svc.agent.create_experiment(session_id=session_id,
        payload=CreateKnowledgeExperimentRequest.model_validate(arguments), principal=svc.principal),
        replay_check=reconcile_submission)
    run_id = response["run_id"]
    atomic_json(flow.directory / "run.json", dict(run_id=run_id, session_id=session_id))
    record = svc.runs.get(run_id)
    if record.state == "succeeded":
        return record
    if record.state != "queued":
        raise UncertainOperation("owned Run is not queued; interrupted fits are never replayed")
    online.ensure_online(online_task_id, timeout=30)
    worker_id = "benchmark:" + str(os.getpid()) + ":" + run_id
    from .version import WORKER_CONTRACT_VERSION
    svc.runs.record_worker_heartbeat(worker_id=worker_id, now=utc_now(),
                                    contract_version=WORKER_CONTRACT_VERSION)
    claimed = svc.runs.claim_next(worker_id=worker_id, now=utc_now(), run_id=run_id)
    if claimed is None or claimed.run_id != run_id:
        raise UncertainOperation("the owned Run could not be claimed")
    def active():
        svc.runs.assert_active(run_id, claim_token=claimed.claim_token, now=utc_now())
    binding = load_training_binding(svc.sessions.database_path, claimed, runs_database=svc.runs.database_path)
    if binding is None:
        raise RuntimeError("benchmark requires a frozen hard-budget training binding")
    ledger = TrainingBudget(binding.agent_database, task_id=binding.task_id, policy_digest=binding.policy_digest,
        reservation_id=binding.reservation_id, run_id=run_id, claim_token=claimed.claim_token, claim_check=active)
    ledger.bind()
    try:
        with LeaseGuard(repository=svc.runs, run_id=run_id, claim_token=claimed.claim_token,
                        worker_id=worker_id, now=utc_now, heartbeat_seconds=5):
            execution = (execution_factory or TrainingExecution)(repository=svc.runs, run_dir=svc.run_root / run_id,
                                                                 training_budget=ledger)
            result = execution.execute(claimed, data_path=Path(row["data_path"]))
            ledger.settle(exit_confirmed=True)
            publication = check_outputs(record=claimed, repository=svc.runs, run_dir=svc.run_root / run_id,
                stage="publication", active=active, usage=ledger.guard_snapshot(settled=True))
            child = svc.runs.get_guard_report(result["guard_report_id"], principal=svc.principal)
            if child.stage != "pre_publish" or child.status != "passed" or child.bindings != publication.bindings:
                raise GuardError("guard_report_binding_mismatch", stage="publication")
            finished = svc.runs.finish_success(run_id, claim_token=claimed.claim_token, now=utc_now(),
                manifest_name=result["manifest_name"], publication_report=publication,
                deadline_at=datetime.fromtimestamp(binding.work_deadline_at, timezone.utc))
        project_status(svc.run_root / run_id, finished)
        return finished
    except BaseException as error:
        ledger.settle(exit_confirmed=True)
        from .runs.worker import _error_details
        details = _error_details(error)
        svc.runs.finish_failure(run_id, claim_token=claimed.claim_token, now=utc_now(),
            error=details["message"], error_details=details)
        raise


def worker(root, job_path, *, driver=None, execution_factory=None):
    job_path = Path(job_path)
    job = read_json(job_path)
    from .execution_runtime import initialize_execution_runtime
    initialize_execution_runtime(job["cpu_threads"])
    driver = driver or BenchmarkDriver(root)
    row = read_json(job["round_path"])
    source = source_reference(root)
    if source["source_digest"] != job["source_digest"]:
        raise ValueError("worker source binding differs")
    result_path = job_path.with_suffix(".result.json")
    with exclusive(job_path.with_suffix(".worker.lock")):
        previous = read_json(result_path)
        if previous is not None:
            return previous
        source_digest, model = source["source_digest"], job["model_id"]
        online, tracking = driver.online(), tracking_id(row, source_digest)
        results, work = [], 0
        for budget in job["budgets"]:
            choice = job["assigned_choices"][str(budget)]
            models = row["eligible_models_by_budget"][str(budget)] if choice else [model]
            kind = ("A0" if budget == 1 else "A1") if choice else "matrix:" + model
            flow = driver.session_flow(row, budget, models, kind,
                execution=execution_options(root, row, source_digest, model, job["cpu_threads"]))
            record_path = driver.directory / "records" / (identity("record", row["task_id"], row["round_id"], model, budget) + ".json")
            existing = read_json(record_path)
            if existing is not None:
                results.append(existing)
                work += existing["cost"]["actual_fits"]
                flow.close()
                continue
            started = time.monotonic()
            run_dir = None
            try:
                record = execute_budget(flow, row, model=model, choice=choice, online=online,
                    online_task_id=tracking, execution_factory=execution_factory)
                svc = flow.services
                if not choice:
                    flow.api("finalize", lambda: svc.agent.finalize_session(session_id=flow.session["session_id"],
                        selected_run_id=record.run_id, principal=svc.principal), phase="finalization")
                run_dir = svc.run_root / record.run_id
                search = read_json(run_dir / "search_summary.json")
                telemetry = read_json(run_dir / "telemetry" / "experiment_summary.json")
                score = search["selected"][0]["selection_score"]
                predictions = grouped_predictions(run_dir / "predictions.csv", row["labels"], row["plan"]["groups"]["test"])
                cost = dict(actual_fits=telemetry["model_fits_completed"], started_fits=telemetry["model_fits_started"],
                    completed_fits=telemetry["model_fits_completed"], epochs=telemetry["executed_epochs"],
                    wall_seconds=telemetry["total_wall_seconds"], driver_wall_seconds=time.monotonic() - started,
                    **flow.costs(), **fitted_costs(run_dir, telemetry))
                metrics = dict(validation_balanced_accuracy=score, **classification_metrics(predictions, row["labels"]))
                fact = dict(task_id=row["task_id"], suite=row["suite"], family_id=row["family_id"], role=row["role"],
                    round_id=row["round_id"], model_id=model, budget=budget, status="completed", run_id=record.run_id,
                    session_id=flow.session["session_id"], artifact_directory=str(run_dir),
                    validation_score=score, prediction_rows=predictions, metrics=metrics, cost=cost)
                work += cost["actual_fits"]
                if budget == 1 and row["role"] == "development":
                    reference = telemetry["default_weight_ref"]
                    if reference is None:
                        raise ValueError("default development checkpoint has no retention reference")
                    from .benchmark_resources import WeightStore
                    WeightStore(Path(root) / "weights").retain_representative(reference["relative_path"], model, job["source_order"])
                online.emit(tracking, "budget.completed", {**{k: fact[k] for k in ("round_id", "model_id", "budget", "status", "run_id", "cost")},
                    **metrics})
            except Exception as error:
                binding = read_json(flow.directory / "run.json")
                telemetry = read_json(flow.services.run_root / binding["run_id"] / "telemetry" / "experiment_summary.json", {}) if binding else {}
                cost = dict(actual_fits=telemetry.get("model_fits_started", 0), started_fits=telemetry.get("model_fits_started", 0),
                    completed_fits=telemetry.get("model_fits_completed", 0), epochs=telemetry.get("executed_epochs", 0),
                    wall_seconds=telemetry.get("total_wall_seconds", time.monotonic() - started), **flow.costs())
                work += cost["actual_fits"]
                fact = dict(task_id=row["task_id"], suite=row["suite"], family_id=row["family_id"], role=row["role"],
                    round_id=row["round_id"], model_id=model, budget=budget, status="failed", prediction_rows=[],
                    validation_score=None, failure_reason=getattr(error, "code", type(error).__name__), cost=cost,
                    resource_failure=type(error).__name__ in {"OutOfMemoryError", "MemoryError"}
                    or "out of memory" in str(error).lower() or "cannot allocate memory" in str(error).lower(),
                    retry_limit=1, retry_started=False)
                if binding:
                    fact.update(run_id=binding["run_id"], session_id=flow.session["session_id"],
                                artifact_directory=str(flow.services.run_root / binding["run_id"]))
                online.emit(tracking, "budget.failed", {k: fact[k] for k in ("round_id", "model_id", "budget", "status", "failure_reason", "cost")})
            finally:
                flow.close()
            atomic_json(record_path, fact)
            results.append(fact)
        result = dict(job_id=job["job_id"], status="completed" if all(f["status"] == "completed" for f in results) else "failed",
            work_units=work, records=[str(driver.directory / "records" / (identity("record", row["task_id"], row["round_id"], model, f["budget"]) + ".json")) for f in results])
        atomic_json(result_path, result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Approved AutoAI benchmark controller")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    preparing = sub.add_parser("prepare")
    preparing.add_argument("--rounds-only", action="store_true")
    preparing.add_argument("--workers", type=int, default=1, help="parallel round Session-freezing workers (1..32)")
    preparing.add_argument("--acceptance-only", action="store_true")
    for name in ("run", "resume", "status", "report"):
        entry = sub.add_parser(name)
        if name == "report":
            entry.add_argument("--development-models", action="store_true")
        if name in ("run", "resume"):
            entry.add_argument("--acceptance-only", action="store_true")
            entry.add_argument("--role", choices=("development", "locked", "all"), default="development")
    internal = sub.add_parser("_worker", help=argparse.SUPPRESS)
    internal.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ["AUTOAI_BENCHMARK_ROOT"] = str(args.root.resolve())
    if args.command == "_worker":
        result = worker(args.root, args.job)
    else:
        driver = BenchmarkDriver(args.root)
        if args.command == "prepare":
            if args.acceptance_only and args.workers != 1:
                parser.error("parallel workers apply to full benchmark preparation")
            result = driver.prepare_acceptance() if args.acceptance_only else driver.prepare(rounds_only=args.rounds_only, workers=args.workers)
        elif args.command in {"run", "resume"}:
            result = driver.run(resume=args.command == "resume", acceptance_only=args.acceptance_only, role=args.role)
        elif args.command == "status":
            result = driver.status()
        else:
            result = driver.development_fixed_models() if args.development_models else driver.report()
    print(json.dumps(result if args.command == "status" else {"command": args.command,
        "status": result.get("preparation_status", result.get("status", "completed")),
        "pair_count": result.get("pair_count"), "run_views": result.get("run_views")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

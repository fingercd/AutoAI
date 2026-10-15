"""Internal benchmark services bound to an approved held-test partition.

The public Agent schemas retain their historical evaluation fields. The actual
partition is this server-owned EvaluationPlan; only select_train feeds evidence.
No fitting, worker startup, or LLM request occurs here.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import fcntl
import hashlib
from functools import lru_cache
from pathlib import Path

from .agent.contracts import AgentDomainError, V2
from .agent.repository import AgentSessionRepository
from .agent.service import AgentService
from .datasets.repository import DatasetRepository
from .evaluation_plan import (EvaluationPlan, digest, evaluation_config,
    load_dataset_view, validate_plan, PreparationResourceExhausted)
from .runs.contracts import Principal
from .runs.repository import RunRepository
from .runs.status_projection import project_status
from .runs.submission import RunSubmissionService, RunSubmissionError

WIRE_EVALUATION = dict(split_mode='stratified_holdout', split_train=8, split_valid=1, split_test=1)
SERVER_FIELDS = frozenset(('execution_device', 'cpu_threads', 'benchmark_context'))


_EVIDENCE_CACHE = {}

@lru_cache(maxsize=32)
def _loaded_frozen_view(path, sha):
    view = load_dataset_view(path, sha)
    view.x.flags.writeable = False
    return view

def _frozen_view(path, sha, limits=None):
    with Path(path).open('rb') as handle:
        if hashlib.file_digest(handle, 'sha256').hexdigest() != sha:
            raise ValueError('registered benchmark data changed')
    view = _loaded_frozen_view(Path(path), sha)
    if limits is not None:
        limits.check(view.x.nbytes * 4)
    return view

class FrozenBenchmarkSubmission(RunSubmissionService):
    """Accept only the registered round dataset and exact trusted partition."""
    def __init__(self, *, trusted_plan, dataset_id, principal, execution, **kwargs):
        super().__init__(**kwargs)
        self.trusted_plan = trusted_plan
        self.dataset_id = dataset_id
        self.principal = principal
        self.execution = deepcopy(execution)

    def prepare_evaluation(self, *, dataset_id, raw_config, principal,
                           expected_sha256=None, legacy_path=None, limits=None):
        if (principal != self.principal or dataset_id != self.dataset_id or legacy_path is not None
                or expected_sha256 not in (None, self.trusted_plan.dataset_sha256)
                or raw_config.get('evaluation_plan_digest') not in (None, self.trusted_plan.plan_digest)
                or {'indices', 'groups', 'plan', 'evaluation_plan'} & raw_config.keys()):
            raise RunSubmissionError('benchmark_binding_mismatch', 'Benchmark partition binding differs', status_code=409)
        try:
            source = self.datasets.resolve(dataset_id, principal=principal)
            sha = self.datasets.verify_integrity(source, expected_sha256=self.trusted_plan.dataset_sha256)
            view = _frozen_view(source.path, sha, limits)
            validate_plan(self.trusted_plan, view, evaluation_config(raw_config), raw_config.get('seed', 42))
            self.runs.save_evaluation_plan(self.trusted_plan, principal=principal)
            return view, self.trusted_plan
        except PreparationResourceExhausted:
            raise RunSubmissionError('preparation_resource_exhausted', 'Preparation resource limit exceeded') from None
        except (FileNotFoundError, PermissionError, ValueError):
            raise RunSubmissionError('evaluation_plan_invalid', 'Frozen benchmark evaluation is invalid') from None

    def transform_config(self, session, config, *, replay=False):
        if (session.dataset_id != self.dataset_id or session.dataset_sha256 != self.trusted_plan.dataset_sha256
                or session.seed != self.trusted_plan.seed
                or config.get('evaluation_plan_digest') != self.trusted_plan.plan_digest):
            raise AgentDomainError('benchmark_binding_mismatch', 'Benchmark session binding differs', status_code=409)
        if replay:
            if any(config.get(key) != value for key, value in self.execution.items()):
                raise AgentDomainError('benchmark_execution_conflict', 'Frozen benchmark execution differs', status_code=409)
            return config
        if SERVER_FIELDS & config.keys():
            raise AgentDomainError('benchmark_server_owned', 'Benchmark execution fields are server-owned', status_code=422)
        return {**config, **deepcopy(self.execution)}

    def submit_prepared(self, *, reference, raw_config, principal, submission_source, **kwargs):
        if (reference.dataset_id != self.dataset_id or reference.legacy_path is not None
                or reference.test_dataset_id is not None or reference.test_legacy_path is not None):
            raise RunSubmissionError('benchmark_binding_mismatch', 'Benchmark dataset binding differs', status_code=409)
        owned = SERVER_FIELDS & raw_config.keys()
        if (submission_source != 'agent' and owned
                or owned and any(raw_config.get(key) != value for key, value in self.execution.items())
                or owned - self.execution.keys()):
            raise RunSubmissionError('benchmark_server_owned', 'Benchmark execution fields are server-owned')
        config = {**raw_config, **deepcopy(self.execution)}
        self.prepare_evaluation(dataset_id=reference.dataset_id, raw_config=config, principal=principal,
                                expected_sha256=kwargs.get('expected_dataset_sha256'))
        config['evaluation_plan_digest'] = self.trusted_plan.plan_digest
        return super().submit_prepared(reference=reference, raw_config=config, principal=principal,
                                       submission_source=submission_source, **kwargs)


class BenchmarkAgentService(AgentService):
    """Bind even direct-action Sessions before they acquire a frozen identity."""
    def create_session(self, payload, *, principal):
        submissions = self.submissions
        if (principal != submissions.principal or payload.dataset_id != submissions.dataset_id
                or payload.seed != submissions.trusted_plan.seed
                or digest(payload.evaluation.model_dump()) != submissions.trusted_plan.evaluation_config_digest):
            raise AgentDomainError('benchmark_binding_mismatch', 'Benchmark session binding differs', status_code=409)
        return super().create_session(payload, principal=principal)


@dataclass(frozen=True)
class BenchmarkServices:
    sessions: AgentSessionRepository
    runs: RunRepository
    datasets: DatasetRepository
    submissions: FrozenBenchmarkSubmission
    agent: AgentService
    principal: Principal
    run_root: Path
    dataset_id: str


def make_services(root: Path, round_record: dict, execution: dict | None = None) -> BenchmarkServices:
    """Construct isolated services; root also controls the combined CSV location.

    execution is supplied by the scheduler before first submission. A recreated
    service can read prior records, but a request replay cannot change execution.
    Each task/round owns separate databases under root/storage. bundle_id and
    source_digest come from execution.benchmark_context or round_record; neither
    is inferred from mutable source or a client request.
    """
    from .benchmark_protocol import build_benchmark_search_plan, validate_context_reference, runtime_policies
    root = Path(root).resolve()
    path = Path(round_record['data_path']).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError('benchmark data must be inside the isolated service root')
    plan = EvaluationPlan.model_validate(round_record['plan'])
    if plan.split_algorithm_version != 'benchmark-held-test-v1' or plan.seed != 42:
        raise ValueError('benchmark requires the approved held-test plan and seed 42')
    if plan.dataset_sha256 != round_record['dataset_sha256']:
        raise ValueError('benchmark dataset hash differs from plan')
    view = _frozen_view(path, round_record['dataset_sha256'])
    validate_plan(plan, view, WIRE_EVALUATION, 42)
    binding = {key: round_record[key] for key in ('task_id', 'round_id', 'protocol_digest', 'dataset_sha256')}
    binding.update(data_path=str(path), plan_digest=plan.plan_digest, suite=round_record.get('suite'))
    if not isinstance(binding['protocol_digest'], str) or len(binding['protocol_digest']) != 64 or any(c not in '0123456789abcdef' for c in binding['protocol_digest']):
        raise ValueError('invalid benchmark protocol digest')
    execution = deepcopy(execution or {})
    if execution.keys() - SERVER_FIELDS:
        raise ValueError('unknown benchmark execution field')
    rows = {key: len(values) for key, values in plan.indices.items()}
    groups = {key: len(values) for key, values in plan.groups.items()}
    context = deepcopy(execution.get('benchmark_context', {}))
    for key in ('bundle_id', 'source_digest'):
        if key not in context:
            if key not in round_record:
                raise ValueError('benchmark requires bundle_id and source_digest from the trusted scheduler')
            context[key] = round_record[key]
    context.update(task_id=binding['task_id'], round_id=binding['round_id'],
        protocol_digest=binding['protocol_digest'], role=round_record.get('role', 'benchmark'),
        partition_row_counts=rows, partition_group_counts=groups,
        partition_group_ratios={key: value / sum(groups.values()) for key, value in groups.items()},
        evaluation_fields_role='legacy_request_format', split_algorithm_version=plan.split_algorithm_version)
    context['suite_root'] = str(root)
    protocol = validate_context_reference(context)
    policies = runtime_policies(protocol)
    execution['benchmark_context'] = context
    from .execution_runtime import validate_execution_options
    validate_execution_options(execution.get('execution_device', 'cpu'), execution.get('cpu_threads'))
    principal = Principal('benchmark-owner', 'benchmark-v3')
    round_root = root / 'storage' / digest([binding['suite'], str(binding['task_id'])])[:16] / digest(str(binding['round_id']))[:16] / plan.plan_digest[:16]
    storage = round_root
    runs = RunRepository(storage / 'runs.sqlite3'); runs.initialize()
    sessions = AgentSessionRepository(storage / 'agent.sqlite3'); sessions.initialize()
    datasets = DatasetRepository(storage / 'datasets.sqlite3', storage_root=root); datasets.initialize()
    binding_path = round_root / 'benchmark-binding.json'
    with (round_root / 'benchmark-binding.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if binding_path.exists():
            persisted = json.loads(binding_path.read_text(encoding='utf-8'))
            if persisted['binding'] != binding:
                raise ValueError('isolated benchmark root already belongs to another round')
            dataset = datasets.resolve(persisted['dataset_id'], principal=principal)
            if dataset.path != path:
                raise ValueError('registered benchmark path differs')
            datasets.verify_integrity(dataset, expected_sha256=plan.dataset_sha256)
        else:
            dataset = datasets.register(path, original_name=path.name, principal=principal)
            temporary = binding_path.with_suffix('.next.json')
            temporary.write_text(json.dumps({'binding': binding, 'dataset_id': dataset.dataset_id},
                ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(binding_path)
    run_root = storage / 'runs'; run_root.mkdir(parents=True, exist_ok=True)
    submissions = FrozenBenchmarkSubmission(trusted_plan=plan, dataset_id=dataset.dataset_id,
        principal=principal, execution=execution, run_repository=runs, dataset_repository=datasets,
        run_dir=lambda run_id: run_root / run_id, status_projector=project_status)
    runs.save_evaluation_plan(plan, principal=principal)
    def evidence_builder(train, limits):
        from .train_evidence import compute_train_evidence
        key = (plan.dataset_sha256, plan.plan_digest)
        if key not in _EVIDENCE_CACHE:
            _EVIDENCE_CACHE[key] = compute_train_evidence(train, limits)
        limits.check()
        return _EVIDENCE_CACHE[key].model_copy(deep=True)

    def search_plan_builder(**kwargs):
        return build_benchmark_search_plan(evaluation_version=policies['evaluation_version'], **kwargs)

    agent = BenchmarkAgentService(session_repository=sessions, run_repository=runs, dataset_repository=datasets,
        submission_service=submissions, run_root=run_root, contract_version=V2,
        search_plan_builder=search_plan_builder, compiled_config_transform=submissions.transform_config,
        evidence_builder=evidence_builder)
    return BenchmarkServices(sessions, runs, datasets, submissions, agent, principal, run_root, dataset.dataset_id)

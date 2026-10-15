"""Offline partition integrity and submission replay tests; never run fitting."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from backend.app.agent.contracts import (AgentDomainError, CreateAgentSessionRequestV2,
    CreateAgentExperimentRequestV2, CreateKnowledgeExperimentRequest)
from backend.app.agent.service import AgentService
from backend.app.benchmark_adapter import make_services, WIRE_EVALUATION
from backend.app.evaluation_plan import (digest, EvaluationPlan, load_dataset_view,
    select_train, validate_plan)
from backend.app.runs.submission import RunSubmissionError, RunSubmissionRequest
from backend.tests.modeling_data_factory import write_grouped_classification_csv


@pytest.fixture
def round_record(tmp_path, monkeypatch):
    from backend.app.benchmark_protocol import PROTOCOL_ID
    protocol = dict(protocol_id=PROTOCOL_ID, governance=dict(user_approved=True),
                    hpo=dict(candidate_budgets=[1, 3]), split_seed=42)
    approval_digest = digest(protocol)
    approval = tmp_path / "approval"
    approval.mkdir()
    (approval / "protocol.json").write_text(json.dumps({**protocol, "approval_digest": approval_digest}))
    monkeypatch.setenv("AUTOAI_BENCHMARK_ROOT", str(tmp_path))
    path = tmp_path / 'prepared' / 'combined.csv'
    path.parent.mkdir()
    write_grouped_classification_csv(path, groups_per_class=6, repeats=1, feature_count=8)
    # Six groups per class: two Train, one Valid, three official held Test.
    indices = dict(train=[0, 1, 6, 7], valid=[2, 8], test=[3, 4, 5, 9, 10, 11])
    frame = pd.read_csv(path)
    frame.iloc[indices['test'], 3:] = 999.0
    frame.to_csv(path, index=False)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    view = load_dataset_view(path, sha)
    groups = {key: sorted({view.sample_ids[i] for i in values}) for key, values in indices.items()}
    body = dict(schema_version='evaluation-plan-v1', split_algorithm_version='benchmark-held-test-v1',
        dataset_sha256=sha, axis_digest=digest(view.axis), evaluation_config_digest=digest(WIRE_EVALUATION),
        seed=42, indices=indices, groups=groups, partition_digest=digest(dict(indices=indices, groups=groups)))
    plan = EvaluationPlan(**body, plan_digest=digest(body))
    validate_plan(plan, view, WIRE_EVALUATION, 42)
    from backend.app.search_policy import _digest_source
    return dict(bundle_id='logistic_regression:held-0', source_digest=_digest_source(), data_path=str(path), dataset_sha256=sha, plan=plan.model_dump(mode='json'),
                task_id='fixture', round_id='held-0', protocol_digest=approval_digest, role='development')


def execution_options(device, threads):
    from backend.app.search_policy import _digest_source
    return dict(execution_device=device, cpu_threads=threads,
        benchmark_context=dict(bundle_id='logistic_regression:held-0', source_digest=_digest_source()))


def create_session(services, **extra):
    payload = CreateAgentSessionRequestV2(dataset_id=services.dataset_id, selection_metric='macro_f1',
        allowed_models=['logistic_regression'], max_runs=1, seed=42,
        client_request_id='session', **extra)
    return services.agent.create_session(payload, principal=services.principal)


def submit(services, created):
    return services.agent.create_experiment(session_id=created['session_id'],
        payload=CreateAgentExperimentRequestV2(model_type='logistic_regression', client_request_id='experiment'),
        principal=services.principal)


def test_non_native_plan_saved_and_train_evidence_excludes_test(tmp_path, round_record):
    services = make_services(tmp_path, round_record)
    view, plan = services.submissions.prepare_evaluation(dataset_id=services.dataset_id,
        raw_config={**WIRE_EVALUATION, 'seed': 42}, principal=services.principal)
    persisted = services.runs.get_evaluation_plan(plan.plan_digest, principal=services.principal)
    assert persisted.model_dump() == round_record['plan']
    assert len(select_train(view, plan).x) == 4
    assert np.max(select_train(view, plan).x) < 999
    created = create_session(services, execution_profile='train-evidence-recipes-v1',
        protocol_revision='agent-recipes-revision-v1', modules=['train_evidence', 'legal_recipes'])
    preparation = created['locked_config']['preparation']
    assert preparation['evidence']['statistics']['observation_count'] == 4
    assert 'indices' not in preparation['evaluation_plan']
    assert 'groups' not in preparation['evaluation_plan']
    assert services.submissions.execution['benchmark_context']['partition_group_ratios']['test'] == .5
    assert services.submissions.execution['benchmark_context']['evaluation_fields_role'] == 'legacy_request_format'


@pytest.mark.parametrize('change', ['indices', 'digest', 'seed', 'partition'])
def test_plan_tampering_rejected(tmp_path, round_record, change):
    record = deepcopy(round_record)
    if change == 'indices': record['plan']['indices']['train'][0] = 3
    elif change == 'digest': record['plan']['plan_digest'] = 'b' * 64
    elif change == 'seed': record['plan']['seed'] = 7
    else:
        record['plan']['indices']['train'][0] = 3
        body = {key: value for key, value in record['plan'].items() if key != 'plan_digest'}
        body['partition_digest'] = digest(dict(indices=body['indices'], groups=body['groups']))
        record['plan'] = {**body, 'plan_digest': digest(body)}
    with pytest.raises(ValueError): make_services(tmp_path, record)


@pytest.mark.parametrize('change', [{'seed': 7}, {'evaluation_plan_digest': 'b' * 64}, {'split_train': 6},
                                  {'indices': {'test': [0]}}])
def test_client_cannot_replace_partition(tmp_path, round_record, change):
    services = make_services(tmp_path, round_record)
    with pytest.raises(RunSubmissionError):
        services.submissions.prepare_evaluation(dataset_id=services.dataset_id,
            raw_config={**WIRE_EVALUATION, 'seed': 42, **change}, principal=services.principal)


def test_dataset_tampering_rejected(tmp_path, round_record):
    services = make_services(tmp_path, round_record)
    with open(round_record['data_path'], 'a') as stream: stream.write('\n')
    with pytest.raises(RunSubmissionError):
        services.submissions.prepare_evaluation(dataset_id=services.dataset_id,
            raw_config={**WIRE_EVALUATION, 'seed': 42}, principal=services.principal)


def test_execution_overlay_is_frozen_before_reservation_and_replay(tmp_path, round_record):
    services = make_services(tmp_path, round_record, execution_options('cpu', 2))
    created = create_session(services)
    first = submit(services, created)
    run = services.runs.get(first['run_id'])
    reservation = services.sessions.list_experiments_scoped(created['session_id'], principal=services.principal)[0]
    assert run.config['execution_device'] == reservation.compiled_config['execution_device'] == 'cpu'
    assert run.config['cpu_threads'] == reservation.compiled_config['cpu_threads'] == 2
    assert run.config['evaluation_plan_digest'] == round_record['plan']['plan_digest']
    same = make_services(tmp_path, round_record, execution_options('cpu', 2))
    assert same.dataset_id == services.dataset_id
    assert submit(same, created)['run_id'] == first['run_id']
    changed = make_services(tmp_path, round_record, execution_options('cuda', 4))
    with pytest.raises(AgentDomainError, match='execution differs'):
        submit(changed, created)
    assert len(services.runs.list()) == 1
    assert services.runs.get(first['run_id']).config['cpu_threads'] == 2


def test_public_training_cannot_send_server_owned_fields(tmp_path, round_record):
    services = make_services(tmp_path, round_record)
    request = RunSubmissionRequest(dataset_id=services.dataset_id, legacy_data_path=None,
        test_dataset_id=None, test_legacy_data_path=None,
        raw_config={**WIRE_EVALUATION, 'model_type': 'logistic_regression', 'benchmark_context': {}},
        principal=services.principal, submission_source='human')
    with pytest.raises(RunSubmissionError, match='server-owned'):
        services.submissions.submit(request)
    assert services.runs.list() == []


def test_round_database_isolation(tmp_path, round_record):
    first = make_services(tmp_path, round_record)
    second = make_services(tmp_path, {**round_record, 'round_id': 'held-1'})
    assert first.sessions.database_path != second.sessions.database_path
    assert first.run_root != second.run_root
    assert first.dataset_id != second.dataset_id


def test_search_builder_is_used_by_recipe_session(tmp_path, round_record):
    services = make_services(tmp_path, round_record)
    created = create_session(services, execution_profile='train-evidence-recipes-v1',
        protocol_revision='agent-recipes-revision-v4', modules=['train_evidence', 'legal_recipes', 'bounded_hpo'],
        search_mode='bounded', max_trials=3)
    session = services.sessions.get_session_scoped(created['session_id'], principal=services.principal)
    plans = session.frozen_preparation['preparation']['search_plans']
    assert plans
    assert all(plan['policy_version'] == 'benchmark-three-config-v1' for plan in plans.values())
    assert all(plan['effective_trials'] <= 3 for plan in plans.values())
    catalog = created['locked_config']['preparation']['catalog']
    recipe = catalog['recipes'][0]
    submitted = services.agent.create_experiment(session_id=created['session_id'],
        payload=CreateKnowledgeExperimentRequest(recipe_id=recipe['recipe_id'], recipe_digest=recipe['recipe_digest'],
            catalog_digest=catalog['catalog_digest'], knowledge_refs=[], client_request_id='recipe-submission'),
        principal=services.principal)
    run = services.runs.get(submitted['run_id'])
    assert run.config['execution_search_plan']['policy_version'] == 'benchmark-three-config-v1'
    assert run.config['evaluation_plan_digest'] == round_record['plan']['plan_digest']



def test_ordinary_service_has_no_internal_hooks(tmp_path, round_record):
    services = make_services(tmp_path, round_record)
    ordinary = AgentService(session_repository=services.sessions, run_repository=services.runs,
        dataset_repository=services.datasets, submission_service=services.submissions, run_root=services.run_root)
    assert ordinary.search_plan_builder is None
    assert ordinary.compiled_config_transform is None


def test_parallel_service_construction_reuses_one_dataset(tmp_path, round_record):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        services = list(pool.map(lambda _: make_services(tmp_path, round_record), range(4)))
    assert len({item.dataset_id for item in services}) == 1


def test_finalize_rejects_run_from_another_session(tmp_path, round_record):
    from backend.app.agent.repository import AgentExperimentNotFound
    services = make_services(tmp_path, round_record, execution_options('cpu', 2))
    first = create_session(services)
    run_id = submit(services, first)['run_id']
    second = services.agent.create_session(CreateAgentSessionRequestV2(dataset_id=services.dataset_id,
        selection_metric='macro_f1', allowed_models=['logistic_regression'], max_runs=1, seed=42,
        client_request_id='other-session'), principal=services.principal)
    with pytest.raises(AgentExperimentNotFound):
        services.agent.finalize_session(session_id=second['session_id'], selected_run_id=run_id,
                                        principal=services.principal)


@pytest.mark.parametrize('change', [{'seed': 7}, {'dataset_id': 'another-dataset'}])
def test_session_cannot_change_round_seed_or_dataset(tmp_path, round_record, change):
    services = make_services(tmp_path, round_record)
    request = CreateAgentSessionRequestV2(**{**dict(dataset_id=services.dataset_id,
        selection_metric='macro_f1', allowed_models=['logistic_regression'], max_runs=1, seed=42), **change})
    with pytest.raises(AgentDomainError, match='binding differs'):
        services.agent.create_session(request, principal=services.principal)
    assert services.runs.list() == []

"""Offline regressions for the approved objective prompt and grouped selection."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from backend.app.benchmark_llm import (BenchmarkLLMAdapter, LEGACY_PROMPT_VERSION,
    OBJECTIVE_PROMPT_VERSION, OBJECTIVE_INSTRUCTION)
from backend.app.benchmark_protocol import (OBJECTIVE_PROTOCOL_ID, PROTOCOL_ID,
    FULL_PROTOCOL_ID, runtime_policies, build_benchmark_search_plan)
from backend.app.probability_evaluation import grouped_evaluation_payload, EVALUATION_VERSION
from backend.tests.test_benchmark_llm import context, final_context, adapter
from backend.tests.test_benchmark_adapter import round_record


def test_objective_wire_changes_no_statistics_or_candidate_details():
    current, _ = adapter()
    legacy, _ = adapter()
    legacy.decision_prompt_version = LEGACY_PROMPT_VERSION
    source = context()
    before = copy.deepcopy(source)
    new_request = current._build_request('submit', source)[0]
    old_request = legacy._build_request('submit', source)[0]
    assert source == before
    assert new_request['messages'][0]['content'] == old_request['messages'][0]['content'] + OBJECTIVE_INSTRUCTION
    actual = json.loads(new_request['messages'][1]['content'])
    expected = json.loads(old_request['messages'][1]['content'])
    expected['task'].pop('evaluation_config')
    assert actual == expected
    assert current.last_semantic_trace['decision_prompt_version'] == OBJECTIVE_PROMPT_VERSION
    assert 'structure' not in json.dumps(actual)


def test_finalization_does_not_receive_model_selection_objective():
    current, _ = adapter()
    request = current._build_request('finalize', final_context())[0]
    assert OBJECTIVE_INSTRUCTION not in request['messages'][0]['content']


@pytest.mark.parametrize('protocol_id', [PROTOCOL_ID, FULL_PROTOCOL_ID])
def test_old_protocols_remain_legacy(protocol_id):
    assert runtime_policies({'protocol_id': protocol_id}) == dict(
        decision_prompt_version=LEGACY_PROMPT_VERSION, evaluation_version=None)


def test_new_protocol_binds_prompt_and_metric_together():
    assert runtime_policies({'protocol_id': OBJECTIVE_PROTOCOL_ID}) == dict(
        decision_prompt_version=OBJECTIVE_PROMPT_VERSION, evaluation_version=EVALUATION_VERSION)


def test_new_session_search_plan_receives_metric_and_cannot_be_forged(tmp_path, round_record):
    from backend.app.evaluation_plan import digest
    from backend.app.benchmark_adapter import make_services
    from backend.app.agent.contracts import CreateAgentSessionRequestV2, CreateKnowledgeExperimentRequest
    from backend.app.search_policy import validate_search_plan
    from backend.app.models.registry import ARCHITECTURE_VERSION
    from backend.app.processing_policy import PROCESSING_POLICY_VERSION
    protocol_path = tmp_path / 'approval/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    protocol.pop('approval_digest')
    protocol['protocol_id'] = OBJECTIVE_PROTOCOL_ID
    protocol['approval_digest'] = digest(protocol)
    protocol_path.write_text(json.dumps(protocol))
    row = {**round_record, 'protocol_digest': protocol['approval_digest']}
    services = make_services(tmp_path, row)
    from backend.app.benchmark_driver import make_session_request
    request_body = make_session_request(tmp_path / 'session-v5', dataset_id=services.dataset_id,
        models=['logistic_regression'], budget=3, role='development', session_key='v5-fixture')
    created = services.agent.create_session(CreateAgentSessionRequestV2.model_validate(request_body),
                                           principal=services.principal)
    locked = created['locked_config']
    plans = locked['preparation']['search_plans']
    assert plans and all(p['evaluation_version'] == EVALUATION_VERSION for p in plans.values())
    catalog = locked['preparation']['catalog']
    recipe = catalog['recipes'][0]
    submitted = services.agent.create_experiment(session_id=created['session_id'],
        payload=CreateKnowledgeExperimentRequest(recipe_id=recipe['recipe_id'], recipe_digest=recipe['recipe_digest'],
            catalog_digest=catalog['catalog_digest'], knowledge_refs=[], client_request_id='v5-submit'),
        principal=services.principal)
    run = services.runs.get(submitted['run_id'])
    plan = run.config['execution_search_plan']
    assert plan['evaluation_version'] == EVALUATION_VERSION
    bindings = dict(model_id='logistic_regression', config=run.config, train_count=4,
        feature_count=8, class_count=2, dataset_sha256=row['dataset_sha256'],
        evaluation_plan_digest=row['plan']['plan_digest'], architecture_version=ARCHITECTURE_VERSION,
        processing_policy_version=PROCESSING_POLICY_VERSION)
    validate_search_plan(plan, **bindings)
    forged = copy.deepcopy(plan)
    forged.pop('evaluation_version')
    forged['plan_digest'] = digest({k:v for k,v in forged.items() if k != 'plan_digest'})
    with pytest.raises(ValueError, match='binding differs'):
        validate_search_plan(forged, **bindings)


def row_evaluation():
    return dict(true=[0, 0, 0, 1], pred=[0, 0, 0, 1],
        probabilities=[[.9,.1],[.9,.1],[.1,.9],[.9,.1]],
        balanced_accuracy=1.0, accuracy=1.0, macro_f1=1.0)


def test_group_payload_uses_probability_and_counts_groups_not_rows():
    raw = row_evaluation(); original = copy.deepcopy(raw)
    grouped = grouped_evaluation_payload(raw, ['a','a','b','c'], ['0','1'])
    assert raw == original
    assert grouped['sample_ids'] == ['a','b','c']
    assert grouped['pred'] == [0,1,0]
    assert grouped['balanced_accuracy'] == .25
    assert grouped['accuracy'] == pytest.approx(1/3)


def test_candidate_selection_changes_winner_without_changing_raw_audit(monkeypatch):
    from backend.app import training
    class FakeModel:
        def __init__(self, c): self.c = c
        def fit(self, x, y): return self
        def predict(self, x): return np.array([0,0,0,1] if self.c == 1 else [1,1,1,0])
        def predict_proba(self, x):
            return np.array(row_evaluation()['probabilities'] if self.c == 1 else [[.9,.1],[.9,.1],[.9,.1],[.1,.9]])
    monkeypatch.setattr(training, 'build_traditional_model', lambda c, *a: FakeModel(c.svm_c))
    candidates = [{'index':0,'params':{'svm_c':1.0}}, {'index':1,'params':{'svm_c':.1}}]
    args = (training.TrainConfig(model_type='svm'), 'svm', np.zeros((4,2)), np.array([0,0,1,1]),
            np.zeros((4,2)), np.array([0,0,0,1]), ['0','1'])
    old = training._select_traditional_config(*args, planned_candidates=candidates)
    new = training._select_traditional_config(*args, planned_candidates=candidates,
                                               valid_sample_ids=['a','a','b','c'])
    assert old.config.svm_c == 1.0
    assert new.config.svm_c == .1
    assert new.valid_balanced_accuracy == 1.0
    assert new.valid_eval['balanced_accuracy'] == 0.0  # intact native-row audit
    assert new.search_rows[0]['selection_score'] == .25
    assert new.search_rows[1]['selection_score'] == 1.0


def test_candidate_probabilities_are_saved_for_future_recomputation(tmp_path):
    from backend.app.training import _candidate_validation_payload
    raw = row_evaluation()
    result = _candidate_validation_payload(raw, ['a','a','b','c'], ['0','1'],
                                           SimpleNamespace(run_dir=tmp_path), 1, 2)
    saved = json.loads((tmp_path/'candidate_validation/1_2.json').read_text())
    assert saved['row_evaluation'] == raw
    assert saved['grouped_evaluation'] == result


def test_group_payload_drives_public_metrics_for_both_partitions():
    from backend.app.training import _build_metrics_payload
    grouped = grouped_evaluation_payload(row_evaluation(), ['a','a','b','c'], ['0','1'])
    metrics, summary = _build_metrics_payload([dict(train=grouped,valid=grouped,test=grouped)],
                                             ['0','1'], 'stratified_holdout')
    assert metrics['valid']['balanced_accuracy'] == metrics['test']['balanced_accuracy'] == .25
    assert metrics['test']['confusion_matrix'] == grouped['confusion_matrix']


@pytest.mark.parametrize('protocol_id,expected', [(FULL_PROTOCOL_ID, LEGACY_PROMPT_VERSION),
                                                (OBJECTIVE_PROTOCOL_ID, OBJECTIVE_PROMPT_VERSION)])
def test_real_driver_adapter_loads_approved_protocol(tmp_path, round_record, protocol_id, expected):
    from backend.app.benchmark_driver import BenchmarkDriver
    from backend.app.evaluation_plan import digest
    path = tmp_path / 'approval/protocol.json'
    protocol = json.loads(path.read_text())
    protocol.pop('approval_digest')
    protocol['protocol_id'] = protocol_id
    protocol['approval_digest'] = digest(protocol)
    path.write_text(json.dumps(protocol))
    actual = BenchmarkDriver._adapter(SimpleNamespace(root=tmp_path))
    assert actual.decision_prompt_version == expected

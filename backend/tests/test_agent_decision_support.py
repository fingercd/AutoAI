from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from backend.app.agent.decision_support import (
    VALIDATION_UNCERTAINTY_METHOD,
    build_decision_support,
    estimate_validation_metric_std,
)
from backend.app.main import app
from backend.app.paths import RUNS_DATABASE, RUNS_DIR
from backend.app.runs.repository import RunRepository, _timestamp
from backend.app.agent.service import _scrub
from backend.tests.modeling_data_factory import write_grouped_classification_csv


def _experiment(run_id, attempt, score, gap, fits):
    return {
        'run_id': run_id,
        'attempt': attempt,
        'state': 'succeeded',
        'validation_score': score,
        'diagnosis': {'generalization_gap': gap},
        'model_fit_count': fits,
        'effective_action': {'proposal_id': f'p_{attempt:016x}'},
    }


def test_recommendation_penalizes_gap_and_cost_not_just_highest_score():
    support = build_decision_support(
        [
            _experiment('high-unstable', 1, 0.91, 0.30, 12),
            _experiment('stable', 2, 0.89, 0.02, 3),
        ],
        remaining_runs=1,
        proposal_catalog={
            'proposals': [
                {'proposal_id': 'p_0000000000000001'},
                {'proposal_id': 'p_0000000000000002'},
                {'proposal_id': 'p_0000000000000003'},
            ]
        },
    )
    assert support['recommended_run_id'] == 'stable'
    assert support['uncertainty']['status'] == 'insufficient_evidence'
    assert support['uncertainty']['comparative_score_range'] == 0.02


def test_single_result_does_not_fabricate_uncertainty_or_auto_stop():
    support = build_decision_support(
        [_experiment('best', 1, 0.95, 0.01, 3)],
        remaining_runs=2,
        proposal_catalog={'proposals': [
            {'proposal_id': 'p_0000000000000001'},
            {'proposal_id': 'p_0000000000000002'},
        ]},
    )
    assert support['uncertainty'] == {
        'status': 'insufficient_evidence',
        'observation_count': 1,
        'candidate_std_available': 0,
        'comparative_score_range': None,
    }
    assert support['stop_recommendation']['action'] == 'continue'


def test_candidate_uncertainty_enables_conservative_target_stop():
    experiment = _experiment('best', 1, 0.96, 0.01, 3)
    experiment['validation_std'] = 0.02
    support = build_decision_support(
        [experiment],
        remaining_runs=2,
        proposal_catalog={
            'proposals': [
                {'proposal_id': 'p_0000000000000001'},
                {'proposal_id': 'p_0000000000000002'},
            ]
        },
    )
    assert support['uncertainty']['status'] == 'ready'
    assert support['candidate_assessments'][0]['conservative_score'] == 0.9208
    assert support['stop_recommendation']['action'] == 'finalize'


def test_validation_jackknife_std_is_derived_from_validation_counts():
    estimate = estimate_validation_metric_std(
        [[48, 2], [2, 48]],
        'macro_f1',
        independent_group_count=100,
        minimum_group_count_per_class=50,
    )
    assert estimate is not None
    validation_std, validation_count = estimate
    assert validation_count == 100
    assert 0.0 < validation_std < 0.1


def test_small_or_replicated_validation_groups_cannot_claim_certainty():
    assert estimate_validation_metric_std(
        [[2, 0], [0, 2]],
        'macro_f1',
        independent_group_count=4,
        minimum_group_count_per_class=2,
    ) is None
    assert estimate_validation_metric_std(
        [[98, 2], [2, 98]],
        'macro_f1',
        independent_group_count=4,
        minimum_group_count_per_class=2,
    ) is None
    assert estimate_validation_metric_std(
        [[50, 0], [0, 50]],
        'macro_f1',
        independent_group_count=12,
        minimum_group_count_per_class=2,
    ) is None
    perfect = estimate_validation_metric_std(
        [[5, 0], [0, 5]],
        'macro_f1',
        independent_group_count=10,
        minimum_group_count_per_class=5,
    )
    assert perfect is not None
    assert perfect[0] > 0.0


def test_agent_projection_removes_sample_identifiers():
    visible = _scrub({
        'progress': {
            'current_fold_sample_id': 'held-out-secret',
            'fold_index': 2,
        },
        'error': 'Sample_ID=patient-42 内存在多个 Label，无法按组划分',
    })
    assert visible['progress'] == {'fold_index': 2}
    rendered = json.dumps(visible, ensure_ascii=False).lower()
    assert 'held-out-secret' not in rendered
    assert 'patient-42' not in rendered
    assert 'sample_id=' not in rendered


def test_uncertainty_module_requires_diagnosis_and_locks_policy(tmp_path):
    client = TestClient(app)
    source = tmp_path / 'selection.csv'
    write_grouped_classification_csv(
        source, groups_per_class=6, repeats=2, feature_count=8
    )
    with source.open('rb') as handle:
        dataset_id = client.post(
            '/api/datasets/upload',
            files={'file': (source.name, handle, 'text/csv')},
        ).json()['dataset_id']
    base = {
        'dataset_id': dataset_id,
        'selection_metric': 'macro_f1',
        'allowed_models': ['logistic_regression'],
        'max_runs': 2,
        'seed': 42,
        'evaluation': {
            'split_mode': 'stratified_holdout',
            'split_train': 8,
            'split_valid': 1,
            'split_test': 1,
        },
    }
    rejected = client.post(
        '/api/agent/sessions',
        json={**base, 'modules': {'uncertainty_selection': True}},
    )
    assert rejected.status_code == 422
    enabled = client.post(
        '/api/agent/sessions',
        json={
            **base,
            'modules': {
                'fail_fast_guard': True,
                'feedback_diagnosis': True,
                'uncertainty_selection': True,
            },
        },
    )
    assert enabled.status_code == 201, enabled.text
    context = enabled.json()['context']
    assert context['status'] == 'ready'
    assert context['selection_policy']['schema_version'] == (
        'agent-decision-support-v1'
    )
    assert context['selection_policy']['uncertainty_multiplier'] == 1.96
    assert context['selection_policy']['minimum_independent_validation_groups'] == 10


def test_session_api_projects_real_validation_uncertainty_and_can_stop(tmp_path):
    client = TestClient(app)
    source = tmp_path / 'uncertainty-api.csv'
    write_grouped_classification_csv(
        source, groups_per_class=6, repeats=2, feature_count=8
    )
    with source.open('rb') as handle:
        dataset_id = client.post(
            '/api/datasets/upload',
            files={'file': (source.name, handle, 'text/csv')},
        ).json()['dataset_id']
    created = client.post('/api/agent/sessions', json={
        'dataset_id': dataset_id,
        'selection_metric': 'macro_f1',
        'allowed_models': ['logistic_regression'],
        'max_runs': 2,
        'seed': 42,
        'evaluation': {
            'split_mode': 'stratified_holdout',
            'split_train': 8,
            'split_valid': 1,
            'split_test': 1,
        },
        'modules': {
            'fail_fast_guard': True,
            'feedback_diagnosis': True,
            'uncertainty_selection': True,
        },
    })
    assert created.status_code == 201, created.text
    session_id = created.json()['session_id']
    experiment = client.post(
        f'/api/agent/sessions/{session_id}/experiments',
        json={
            'model_type': 'logistic_regression',
            'normalization': 'zscore',
            'class_balance': 'none',
        },
    )
    assert experiment.status_code == 202, experiment.text
    run_id = experiment.json()['run_id']
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    scalar_keys = (
        'accuracy', 'balanced_accuracy', 'macro_precision',
        'macro_recall', 'macro_f1', 'weighted_f1',
    )
    train = {key: 0.98 for key in scalar_keys}
    valid = {key: 0.96 for key in scalar_keys}
    train['confusion_matrix'] = [[49, 1], [1, 49]]
    valid.update({
        'aggregation': 'direct_holdout',
        'confusion_matrix': [[48, 2], [2, 48]],
        'independent_group_count': 100,
        'minimum_group_count_per_class': 50,
    })
    (run_dir / 'metrics.json').write_text(
        json.dumps({'train': train, 'valid': valid, 'test': {'macro_f1': 1.0}}),
        encoding='utf-8',
    )
    repository = RunRepository(RUNS_DATABASE)
    repository.initialize()
    now_text = _timestamp(datetime.now(timezone.utc))
    with repository._connection() as connection:  # type: ignore[attr-defined]
        connection.execute('BEGIN IMMEDIATE')
        connection.execute(
            '''
            UPDATE runs
            SET state = 'succeeded', version = version + 1,
                manifest_name = 'manifest.json', updated_at = ?, finished_at = ?
            WHERE run_id = ? AND state = 'queued'
            ''',
            (now_text, now_text, run_id),
        )
        connection.commit()

    detail = client.get(f'/api/agent/sessions/{session_id}')
    assert detail.status_code == 200, detail.text
    payload = detail.json()
    summary = payload['experiments'][0]
    assert 0.0 < summary['validation_std'] < 0.1
    assert summary['validation_uncertainty'] == {
        'method': VALIDATION_UNCERTAINTY_METHOD,
        'independent_group_count': 100,
        'minimum_group_count_per_class': 50,
    }
    support = payload['decision_support']
    assert support['uncertainty']['status'] == 'ready'
    assert support['stop_recommendation'] == {
        'action': 'finalize',
        'reason': 'target_reached',
    }
    flat = json.dumps(
        {
            'experiments': payload['experiments'],
            'decision_support': support,
        },
        ensure_ascii=False,
    ).lower()
    assert 'confusion' not in flat
    assert '"test"' not in flat

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.tests.modeling_data_factory import write_grouped_classification_csv


def test_code_evolution_policy_requires_guard_and_is_locked(tmp_path):
    client = TestClient(app)
    source = tmp_path / 'evolution.csv'
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
        'max_runs': 1,
        'seed': 42,
        'evaluation': {
            'split_mode': 'stratified_holdout',
            'split_train': 8,
            'split_valid': 1,
            'split_test': 1,
        },
    }
    missing_guard = client.post(
        '/api/agent/sessions',
        json={
            **base,
            'modules': {'constrained_code_evolution': True},
        },
    )
    assert missing_guard.status_code == 422
    enabled = client.post(
        '/api/agent/sessions',
        json={
            **base,
            'modules': {
                'fail_fast_guard': True,
                'constrained_code_evolution': True,
            },
        },
    )
    assert enabled.status_code == 201, enabled.text
    context = enabled.json()['context']
    assert context['status'] == 'pending'
    assert context['module_status']['constrained_code_evolution'] == 'experimental'
    assert context['code_evolution_policy']['status'] == 'experimental'
    assert (
        context['code_evolution_policy']['mode']
        == 'candidate_generation_and_smoke_only'
    )
    assert context['code_evolution_policy']['requires_guard'] is True

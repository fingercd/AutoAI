from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app.agent import policy
from backend.app.agent.policy import compile_proposal_catalog, resolve_proposal
from backend.app.agent.service import _compute_config_hash
from backend.app.main import app
from backend.tests.modeling_data_factory import write_grouped_classification_csv


def _dataset(client, tmp_path):
    source = tmp_path / 'policy.csv'
    write_grouped_classification_csv(
        source,
        groups_per_class=8,
        repeats=2,
        feature_count=12,
    )
    with source.open('rb') as handle:
        response = client.post(
            '/api/datasets/upload',
            files={'file': (source.name, handle, 'text/csv')},
        )
    assert response.status_code == 200
    return response.json()['dataset_id']


def _session_body(dataset_id):
    return {
        'dataset_id': dataset_id,
        'selection_metric': 'macro_f1',
        'allowed_models': ['logistic_regression', 'svm'],
        'max_runs': 2,
        'seed': 42,
        'evaluation': {
            'split_mode': 'stratified_holdout',
            'split_train': 8,
            'split_valid': 1,
            'split_test': 1,
        },
        'modules': {
            'evidence_card': True,
            'restricted_strategy_pool': True,
            'dynamic_preprocessing': True,
        },
    }


def test_catalog_is_deterministic_and_bounded():
    evidence = {'statistics': {'risk_codes': ['class_imbalance']}}
    first = compile_proposal_catalog(
        allowed_models=['svm', 'logistic_regression'],
        evidence_card=evidence,
        dynamic_preprocessing=True,
    )
    second = compile_proposal_catalog(
        allowed_models=['logistic_regression', 'svm'],
        evidence_card=evidence,
        dynamic_preprocessing=True,
    )
    assert first == second
    assert first['proposal_count'] == 8
    assert all(item['normalization'] != 'area' for item in first['proposals'])


def test_restricted_session_requires_exact_canonical_recipe(tmp_path):
    client = TestClient(app)
    dataset_id = _dataset(client, tmp_path)
    created = client.post('/api/agent/sessions', json=_session_body(dataset_id))
    assert created.status_code == 201, created.text
    session = created.json()
    catalog = session['context']['proposal_catalog']
    assert session['context']['status'] == 'ready'
    recipe = catalog['proposals'][0]
    malicious = (
        'Bearer abc password=hunter2 https://internal.example '
        '/users/private/model C:\\private\\token.txt'
    )
    accepted = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments",
        json={**recipe, 'rationale': malicious},
    )
    assert accepted.status_code == 202, accepted.text
    for forbidden in (
        'bearer abc', 'hunter2', 'internal.example',
        '/users/private', 'c:\\private',
    ):
        assert forbidden not in accepted.text.lower()
    observed = client.get(
        f"/api/agent/sessions/{session['session_id']}"
    )
    for forbidden in ('hunter2', '/users/private', 'internal.example'):
        assert forbidden not in observed.text.lower()
    mismatch = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments",
        json={
            **recipe,
            'normalization': 'none',
            'rationale': 'tamper',
        },
    )
    assert mismatch.status_code == 422


def test_dynamic_preprocessing_requires_evidence_and_restricted_pool(tmp_path):
    client = TestClient(app)
    dataset_id = _dataset(client, tmp_path)
    body = _session_body(dataset_id)
    body['modules'] = {'dynamic_preprocessing': True}
    response = client.post('/api/agent/sessions', json=body)
    assert response.status_code == 422


def test_backend_effective_hash_ignores_parent_and_proposal_identity():
    session = SimpleNamespace(
        dataset_id='dataset-1',
        selection_metric='macro_f1',
        seed=42,
        evaluation_config={'split_mode': 'stratified_holdout'},
    )
    base = {
        'model_type': 'svm',
        'normalization': 'zscore',
        'class_balance': 'none',
    }
    first = _compute_config_hash(
        session=session,
        action={
            **base,
            'proposal_id': 'p_0123456789abcdef',
            'parent_run_id': 'run-a',
        },
    )
    second = _compute_config_hash(
        session=session,
        action={
            **base,
            'proposal_id': 'p_fedcba9876543210',
            'parent_run_id': 'run-b',
        },
    )
    assert first == second


def test_catalog_rejects_id_collision_and_digest_tampering(monkeypatch):
    monkeypatch.setattr(policy, '_proposal_id', lambda _: 'p_0000000000000000')
    with pytest.raises(ValueError, match='collision'):
        compile_proposal_catalog(
            allowed_models=['logistic_regression', 'svm'],
            evidence_card=None,
            dynamic_preprocessing=True,
        )
    monkeypatch.undo()
    catalog = compile_proposal_catalog(
        allowed_models=['logistic_regression'],
        evidence_card=None,
        dynamic_preprocessing=False,
    )
    catalog['catalog_digest'] = '0' * 64
    with pytest.raises(ValueError, match='integrity'):
        resolve_proposal(
            {'proposal_catalog': catalog},
            catalog['proposals'][0]['proposal_id'],
        )

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.agent.evidence import (
    build_dataset_evidence_card,
    build_train_evidence_card,
)
from backend.app.classification_split import stratified_group_holdout_indices
from backend.app.main import app
from backend.app.parsers import load_modeling_csv
from backend.tests.modeling_data_factory import write_grouped_classification_csv


@pytest.fixture
def client():
    return TestClient(app)


def _upload(client, path):
    with path.open('rb') as handle:
        response = client.post(
            '/api/datasets/upload',
            files={'file': (path.name, handle, 'text/csv')},
        )
    assert response.status_code == 200
    return response.json()['dataset_id']


def test_evidence_card_contains_only_anonymous_train_statistics():
    x = np.asarray([[0, 1], [0, 1], [2, 2], [3, 3]], dtype=np.float64)
    y = np.asarray([0, 0, 1, 1], dtype=np.int64)
    groups = np.asarray(['secret-a', 'secret-a', 'secret-b', 'secret-b'])
    card = build_train_evidence_card(
        x,
        y,
        groups,
        split_fingerprint='b' * 64,
        data_format='wide-feature-v2',
    )
    assert card['scope'] == 'train_only'
    assert card['statistics']['duplicate_observation_count'] == 1
    assert 'very_small_train_partition' in card['statistics']['risk_codes']
    flat = json.dumps(card, ensure_ascii=False).lower()
    for forbidden in ('secret-a', 'secret-b', 'sample_id', 'label', '/users/', 'test'):
        assert forbidden not in flat


def test_enabled_session_persists_train_only_evidence(client, tmp_path):
    source = tmp_path / 'evidence.csv'
    write_grouped_classification_csv(
        source,
        groups_per_class=8,
        repeats=2,
        feature_count=40,
    )
    dataset_id = _upload(client, source)
    response = client.post(
        '/api/agent/sessions',
        json={
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
            'modules': {'evidence_card': True},
        },
    )
    assert response.status_code == 201, response.text
    card = response.json()['context']['evidence_card']
    assert card['scope'] == 'train_only'
    assert card['statistics']['predictor_count'] == 40
    assert 'high_dimension' in card['statistics']['risk_codes']
    flat = json.dumps(card, ensure_ascii=False).lower()
    for forbidden in ('sample_id', 'name', '/users/', 'test'):
        assert forbidden not in flat


def test_default_session_does_not_build_evidence(monkeypatch, client, tmp_path):
    source = tmp_path / 'legacy.csv'
    write_grouped_classification_csv(
        source,
        groups_per_class=6,
        repeats=2,
        feature_count=8,
    )
    dataset_id = _upload(client, source)

    def unexpected(*args, **kwargs):
        raise AssertionError('evidence should be disabled')

    monkeypatch.setattr(
        'backend.app.agent.service.build_dataset_evidence_card',
        unexpected,
    )
    response = client.post(
        '/api/agent/sessions',
        json={
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
        },
    )
    assert response.status_code == 201
    assert response.json()['context']['status'] == 'disabled'


def test_held_out_feature_changes_do_not_change_decision_evidence(tmp_path):
    source = tmp_path / 'source.csv'
    modified = tmp_path / 'modified.csv'
    write_grouped_classification_csv(
        source,
        groups_per_class=8,
        repeats=2,
        feature_count=12,
    )
    dataset = load_modeling_csv(source)
    label_names = sorted(set(dataset.labels))
    label_to_id = {label: index for index, label in enumerate(label_names)}
    y = np.asarray([label_to_id[label] for label in dataset.labels])
    sample_ids = dataset.frame['Sample_ID'].astype(str).to_numpy()
    splits = stratified_group_holdout_indices(
        y,
        sample_ids,
        seed=42,
        split_train=8,
        split_valid=1,
        split_test=1,
        label_names=label_names,
    )
    held_out = splits['valid'] + splits['test']
    frame = pd.read_csv(source)
    metadata_count = 4 if frame.columns[3] == 'Name' else 3
    frame.iloc[held_out, metadata_count:] += 1000.0
    frame.to_csv(modified, index=False)
    evaluation = {'split_train': 8, 'split_valid': 1, 'split_test': 1}
    original_card = build_dataset_evidence_card(
        source,
        dataset_digest='a' * 64,
        seed=42,
        evaluation_config=evaluation,
    )
    modified_card = build_dataset_evidence_card(
        modified,
        dataset_digest='b' * 64,
        seed=42,
        evaluation_config=evaluation,
    )
    assert original_card == modified_card


def test_evidence_resource_limits_fail_fast(monkeypatch, tmp_path):
    source = tmp_path / 'too-large.csv'
    source.write_text('0123456789', encoding='utf-8')
    monkeypatch.setattr(
        'backend.app.agent.evidence.MAX_EVIDENCE_FILE_BYTES',
        5,
    )
    with pytest.raises(ValueError, match='file limit'):
        build_dataset_evidence_card(
            source,
            dataset_digest='a' * 64,
            seed=42,
            evaluation_config={'split_train': 8, 'split_valid': 1, 'split_test': 1},
        )


def test_train_builder_enforces_matrix_cell_limit(monkeypatch):
    monkeypatch.setattr(
        'backend.app.agent.evidence.MAX_EVIDENCE_CELLS',
        3,
    )
    with pytest.raises(ValueError, match='matrix limit'):
        build_train_evidence_card(
            np.ones((2, 2), dtype=np.float32),
            np.asarray([0, 1]),
            np.asarray(['a', 'b']),
            split_fingerprint='b' * 64,
            data_format='wide-feature-v2',
        )

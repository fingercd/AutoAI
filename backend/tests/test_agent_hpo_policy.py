from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.app import training
from backend.app.main import app
from backend.app.paths import RUNS_DATABASE
from backend.app.runs.repository import RunRepository
from backend.app.training import (
    TrainConfig,
    _random_forest_oob_metrics,
    _select_random_forest_config,
    _select_traditional_config,
    _traditional_candidate_configs,
)
from backend.tests.modeling_data_factory import write_grouped_classification_csv


def test_hpo_profiles_bound_candidate_counts_deterministically():
    y = np.asarray([0, 1] * 10)
    standard = _traditional_candidate_configs(
        TrainConfig(hpo_profile='standard'), 'svm', 10, y
    )
    tiny = _traditional_candidate_configs(
        TrainConfig(hpo_profile='tiny'), 'svm', 10, y
    )
    off = _traditional_candidate_configs(
        TrainConfig(hpo_profile='off'), 'svm', 10, y
    )
    assert len(standard) == 5
    assert len(tiny) == 3
    assert len(off) == 1
    assert [item.svm_c for item in tiny] == [0.01, 0.1, 1.0]


def test_selector_uses_session_metric_and_marks_same_winner(monkeypatch):
    class FakeModel:
        def __init__(self, c):
            self.c = c

        def fit(self, x, y):
            return self

    monkeypatch.setattr(
        training,
        'build_traditional_model',
        lambda config, *_: FakeModel(config.logistic_c),
    )

    def evaluate(model, *_):
        if model.c == 0.1:
            return {'accuracy': 0.9, 'balanced_accuracy': 0.9, 'macro_f1': 0.5}
        return {'accuracy': 0.8, 'balanced_accuracy': 0.8, 'macro_f1': 0.7}

    monkeypatch.setattr(training, '_evaluate_traditional_model', evaluate)
    selected = _select_traditional_config(
        TrainConfig(hpo_profile='tiny', hpo_selection_metric='macro_f1'),
        'logistic_regression',
        np.ones((4, 2)),
        np.asarray([0, 0, 1, 1]),
        np.ones((2, 2)),
        np.asarray([0, 1]),
        ['a', 'b'],
    )
    assert selected.config.logistic_c == 1.0
    selected_row = next(row for row in selected.search_rows if row['is_selected'])
    assert selected_row['selection_metric'] == 'macro_f1'
    assert selected_row['selection_source'] == 'validation'
    assert selected_row['selection_score'] == 0.7


def test_random_forest_oob_metrics_include_macro_f1():
    model = SimpleNamespace(
        oob_decision_function_=np.asarray([
            [0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.7, 0.3],
        ]),
        classes_=np.asarray([0, 1]),
    )
    accuracy, balanced, macro_f1 = _random_forest_oob_metrics(
        model,
        np.asarray([0, 0, 1, 1]),
    )
    assert accuracy == 0.75
    assert 0.0 < balanced < 1.0
    assert 0.0 < macro_f1 < 1.0


def test_random_forest_macro_f1_selection_uses_oob_macro(monkeypatch):
    candidates = [
        TrainConfig(random_forest_max_depth=3),
        TrainConfig(random_forest_max_depth=5),
    ]

    class FakeModel:
        def __init__(self, depth):
            self.depth = depth

        def fit(self, x, y):
            return self

    monkeypatch.setattr(
        training,
        '_traditional_candidate_configs',
        lambda *args, **kwargs: candidates,
    )
    monkeypatch.setattr(
        training,
        'build_traditional_model',
        lambda config, *_: FakeModel(config.random_forest_max_depth),
    )
    monkeypatch.setattr(
        training,
        '_random_forest_oob_metrics',
        lambda model, y: (
            0.9 if model.depth == 3 else 0.8,
            0.9 if model.depth == 3 else 0.8,
            0.4 if model.depth == 3 else 0.7,
        ),
    )
    monkeypatch.setattr(
        training,
        '_evaluate_traditional_model',
        lambda *args, **kwargs: {
            'accuracy': 0.75,
            'balanced_accuracy': 0.75,
            'macro_f1': 0.75,
        },
    )
    selected = _select_random_forest_config(
        TrainConfig(hpo_selection_metric='macro_f1'),
        np.ones((4, 2)),
        np.asarray([0, 0, 1, 1]),
        np.ones((2, 2)),
        np.asarray([0, 1]),
        ['a', 'b'],
    )
    assert selected.config.random_forest_max_depth == 5
    row = next(item for item in selected.search_rows if item['is_selected'])
    assert row['selection_metric'] == 'macro_f1'
    assert row['selection_source'] == 'oob'
    assert row['selection_score'] == 0.7


def test_random_forest_off_profile_uses_validation_without_oob(monkeypatch):
    class FakeModel:
        def fit(self, x, y):
            return self

    monkeypatch.setattr(
        training,
        'build_traditional_model',
        lambda *args, **kwargs: FakeModel(),
    )
    monkeypatch.setattr(
        training,
        '_evaluate_traditional_model',
        lambda *args, **kwargs: {
            'accuracy': 0.8,
            'balanced_accuracy': 0.8,
            'macro_f1': 0.8,
        },
    )
    selected = _select_traditional_config(
        TrainConfig(
            model_type='random_forest',
            hpo_profile='off',
            hpo_selection_metric='macro_f1',
        ),
        'random_forest',
        np.ones((4, 2)),
        np.asarray([0, 0, 1, 1]),
        np.ones((2, 2)),
        np.asarray([0, 1]),
        ['a', 'b'],
    )
    assert selected.search_rows[0]['selection_source'] == 'validation'


def test_bounded_hpo_session_locks_tiny_policy_and_run_config(tmp_path):
    client = TestClient(app)
    source = tmp_path / 'hpo.csv'
    write_grouped_classification_csv(
        source, groups_per_class=6, repeats=2, feature_count=8
    )
    with source.open('rb') as handle:
        uploaded = client.post(
            '/api/datasets/upload',
            files={'file': (source.name, handle, 'text/csv')},
        )
    dataset_id = uploaded.json()['dataset_id']
    created = client.post(
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
            'modules': {'bounded_hpo': True},
        },
    )
    assert created.status_code == 201, created.text
    session = created.json()
    assert session['context']['status'] == 'ready'
    assert session['context']['hpo_policy'] == {
        'schema_version': 'bounded-hpo-v1',
        'profile': 'tiny',
        'selection_metric': 'macro_f1',
        'max_candidates': 3,
    }
    submitted = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments",
        json={
            'model_type': 'logistic_regression',
            'normalization': 'zscore',
            'class_balance': 'none',
            'rationale': 'baseline',
        },
    )
    assert submitted.status_code == 202, submitted.text
    repository = RunRepository(RUNS_DATABASE)
    repository.initialize()
    record = repository.get(submitted.json()['run_id'])
    assert record.config['hpo_profile'] == 'tiny'
    assert record.config['hpo_selection_metric'] == 'macro_f1'
    rejected = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments",
        json={'model_type': 'logistic_regression', 'hpo_profile': 'standard'},
    )
    assert rejected.status_code == 422


def test_hpo_artifacts_record_exact_model_fit_count(tmp_path, monkeypatch):
    source = tmp_path / 'train.csv'
    write_grouped_classification_csv(
        source, groups_per_class=6, repeats=2, feature_count=8
    )
    runs_dir = tmp_path / 'runs'
    monkeypatch.setattr(training, 'RUNS_DIR', runs_dir)
    result = training.train_model(
        source,
        {
            'model_type': 'logistic_regression',
            'hpo_profile': 'tiny',
            'hpo_selection_metric': 'macro_f1',
            'feature_selection_enabled': False,
        },
        run_id='hpo-artifact-test',
    )
    run_dir = runs_dir / 'hpo-artifact-test'
    search = pd.read_csv(run_dir / 'hyperparameter_search.csv')
    metadata = json.loads((run_dir / 'model_metadata.json').read_text())
    config = json.loads((run_dir / 'config.json').read_text())
    cv_metrics = json.loads((run_dir / 'cv_metrics.json').read_text())
    expected = len(search) + result['fold_count']
    assert metadata['model_fit_count'] == expected
    assert config['model_fit_count'] == expected
    assert cv_metrics['model_fit_count'] == expected
    assert set(search['hpo_profile']) == {'tiny'}
    assert set(search['selection_metric']) == {'macro_f1'}


def test_agent_metric_is_forwarded_when_bounded_hpo_is_disabled(tmp_path):
    client = TestClient(app)
    source = tmp_path / 'legacy-agent.csv'
    write_grouped_classification_csv(
        source, groups_per_class=6, repeats=2, feature_count=8
    )
    with source.open('rb') as handle:
        dataset_id = client.post(
            '/api/datasets/upload',
            files={'file': (source.name, handle, 'text/csv')},
        ).json()['dataset_id']
    session = client.post(
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
    ).json()
    submitted = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments",
        json={'model_type': 'logistic_regression', 'rationale': 'baseline'},
    )
    repository = RunRepository(RUNS_DATABASE)
    repository.initialize()
    record = repository.get(submitted.json()['run_id'])
    assert record.config['hpo_selection_metric'] == 'macro_f1'
    assert 'hpo_profile' not in record.config

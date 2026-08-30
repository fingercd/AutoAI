from __future__ import annotations

import csv
import json

from backend.app.runs.status_projection import build_training_status_projection


def _metadata():
    return {'model_type': 'random_forest', 'model_family': 'traditional_ml'}


def test_cv_metrics_projection_preserves_oob_selection_source(tmp_path):
    (tmp_path / 'cv_metrics.json').write_text(
        json.dumps({
            'strategy': 'leave_one_sample_id_cv',
            'folds': [
                {
                    'fold_index': 1,
                    'best_params': {'random_forest_max_depth': 5},
                    'selection_metric': 'macro_f1',
                    'selection_source': 'oob',
                    'selection_score': 0.7,
                    'split_metrics': {'valid': {'balanced_accuracy': 0.6}},
                }
            ],
        }),
        encoding='utf-8',
    )
    projection = build_training_status_projection(
        tmp_path,
        config={'evaluation_strategy': 'leave_one_sample_id_cv'},
        model_metadata=_metadata(),
        read_status_file=False,
    )
    entry = projection['traditional']['best_params_by_fold'][0]
    assert entry['selection_metric'] == 'macro_f1'
    assert entry['selection_source'] == 'oob'


def test_csv_fallback_projection_preserves_selection_source(tmp_path):
    with (tmp_path / 'hyperparameter_search.csv').open(
        'w', encoding='utf-8', newline=''
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'fold_index', 'is_selected', 'selection_metric',
                'selection_source', 'selection_score',
                'valid_balanced_accuracy', 'params_json',
            ],
        )
        writer.writeheader()
        writer.writerow({
            'fold_index': 1,
            'is_selected': 'true',
            'selection_metric': 'macro_f1',
            'selection_source': 'oob',
            'selection_score': 0.71,
            'valid_balanced_accuracy': 0.62,
            'params_json': '{"random_forest_max_depth": 5}',
        })
    projection = build_training_status_projection(
        tmp_path,
        config={'evaluation_strategy': 'external_test_holdout'},
        model_metadata=_metadata(),
        read_status_file=False,
    )
    traditional = projection['traditional']
    assert traditional['selection_metric'] == 'macro_f1'
    assert traditional['selection_source'] == 'oob'

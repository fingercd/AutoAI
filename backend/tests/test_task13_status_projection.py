from __future__ import annotations

import json
from pathlib import Path

from backend.app.runs.status_projection import build_training_status_projection


def _write_json(run_dir: Path, name: str, payload: dict) -> None:
    (run_dir / name).write_text(json.dumps(payload), encoding="utf-8")


def test_projects_traditional_selection_by_fold_and_search_csv(tmp_path: Path) -> None:
    _write_json(
        tmp_path,
        "config.json",
        {"evaluation_strategy": "leave_one_repeat_index_cv", "model_type": "pca_lda"},
    )
    _write_json(
        tmp_path,
        "status.json",
        {
            "model_family": "traditional_ml",
            "feature_importance": {
                "method": "interval_permutation_importance",
                "importance_metric": "baseline_macro_f1_minus_perturbed_macro_f1",
            },
        },
    )
    _write_json(
        tmp_path,
        "model_metadata.json",
        {
            "architecture_version": "docx-classification-v2",
            "explainability_method": "window_permutation",
            "artifact_explainability_method": "interval_permutation_importance",
        },
    )
    _write_json(
        tmp_path,
        "cv_metrics.json",
        {
            "folds": [
                {
                    "fold_index": 0,
                    "best_params": {"pca_components": 8},
                    "selection_metric": "balanced_accuracy",
                    "selection_score": 0.81,
                },
                {
                    "fold_index": 1,
                    "best_params": {"pca_components": 10},
                    "selection_metric": "balanced_accuracy",
                    "selection_score": 0.91,
                },
            ]
        },
    )
    (tmp_path / "hyperparameter_search.csv").write_text(
        "fold_index,is_selected,valid_balanced_accuracy,params_json\n"
        '0,True,0.81,"{""pca_components"": 8}"\n',
        encoding="utf-8",
    )

    projection = build_training_status_projection(tmp_path)

    assert projection["traditional"]["best_params_by_fold"] == [
        {"fold_index": 0, "params": {"pca_components": 8}, "valid_balanced_accuracy": 0.81},
        {"fold_index": 1, "params": {"pca_components": 10}, "valid_balanced_accuracy": 0.91},
    ]
    assert projection["traditional"]["hyperparameter_search_csv"] == {
        "artifact": "hyperparameter_search.csv",
        "available": True,
    }
    assert projection["explainability"] == {
        "declared_method": "window_permutation",
        "artifact_method": "interval_permutation_importance",
        "importance_metric": "baseline_macro_f1_minus_perturbed_macro_f1",
    }


def test_projects_deep_audit_values_from_status_history(tmp_path: Path) -> None:
    _write_json(
        tmp_path,
        "config.json",
        {"evaluation_strategy": "stratified_holdout", "model_type": "cnn1d"},
    )
    _write_json(
        tmp_path,
        "status.json",
        {
            "model_family": "deep_learning",
            "actual_epochs": 3,
            "history": [
                {"best_valid_loss": 0.9, "learning_rate": 0.001},
                {"best_valid_loss": 0.4, "learning_rate": 0.0005},
                {"best_valid_loss": 0.4, "learning_rate": 0.0001},
            ],
            "sample_feature_importance": {
                "method": "gradcam_1d",
                "importance_metric": "gradcam_activation",
            },
        },
    )
    _write_json(
        tmp_path,
        "model_metadata.json",
        {
            "explainability_method": "gradcam_1d",
            "artifact_explainability_method": "gradcam_1d",
        },
    )

    projection = build_training_status_projection(tmp_path)

    assert projection["deep_training"] == {
        "best_valid_loss": 0.4,
        "actual_epochs": 3,
        "min_learning_rate": 0.0001,
    }
    assert projection["explainability"]["importance_metric"] == "gradcam_activation"


def test_external_projection_never_exposes_cv_or_fold_fields_and_legacy_runs_are_safe(tmp_path: Path) -> None:
    _write_json(
        tmp_path,
        "config.json",
        {"evaluation_strategy": "external_test_holdout", "model_type": "svm"},
    )
    _write_json(tmp_path, "status.json", {"model_family": "traditional_ml"})
    _write_json(
        tmp_path,
        "cv_metrics.json",
        {
            "folds": [
                {
                    "fold_index": 0,
                    "best_params": {"svm_c": 1.0},
                    "selection_metric": "balanced_accuracy",
                    "selection_score": 0.75,
                }
            ]
        },
    )

    projection = build_training_status_projection(tmp_path)

    assert projection["traditional"]["best_params"] == {"svm_c": 1.0}
    assert projection["traditional"]["valid_balanced_accuracy"] == 0.75
    assert "best_params_by_fold" not in projection["traditional"]
    assert "fold" not in json.dumps(projection).lower()
    legacy_projection = build_training_status_projection(tmp_path / "missing")
    assert legacy_projection["model_type"] is None
    assert legacy_projection["explainability"] == {}

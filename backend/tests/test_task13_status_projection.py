from __future__ import annotations

import json
from pathlib import Path

from backend.app.runs.contracts import RunRecord
from backend.app.runs.status_projection import build_training_status_projection, project_status


def _write_json(run_dir: Path, name: str, payload: dict) -> None:
    (run_dir / name).write_text(json.dumps(payload), encoding="utf-8")


def test_projects_traditional_selection_by_fold_and_search_csv(tmp_path: Path) -> None:
    _write_json(
        tmp_path,
        "config.json",
        {"evaluation_strategy": "leave_one_sample_id_cv", "model_type": "pca_lda"},
    )
    _write_json(
        tmp_path,
        "status.json",
        {
            "model_family": "traditional_ml",
            "sample_feature_importance": {
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
        {
            "fold_index": 0,
            "params": {"pca_components": 8},
            "selection_metric": "balanced_accuracy",
            "selection_score": 0.81,
            "valid_balanced_accuracy": 0.81,
        },
        {
            "fold_index": 1,
            "params": {"pca_components": 10},
            "selection_metric": "balanced_accuracy",
            "selection_score": 0.91,
            "valid_balanced_accuracy": 0.91,
        },
    ]
    assert projection["traditional"]["hyperparameter_search_csv"] == {
        "artifact": "hyperparameter_search.csv",
        "available": True,
    }
    assert projection["explainability"] == {"status": "temporarily_hidden"}


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
    assert projection["explainability"] == {"status": "temporarily_hidden"}


def test_projects_log_loss_occlusion_contract(tmp_path: Path) -> None:
    _write_json(
        tmp_path,
        "status.json",
        {
            "sample_feature_importance": {
                "method": "sample_occlusion_log_loss",
                "importance_metric": "masked_true_class_log_loss_minus_original_true_class_log_loss",
            },
        },
    )
    _write_json(
        tmp_path,
        "model_metadata.json",
        {
            "explainability_method": "window_occlusion_log_loss",
            "artifact_explainability_method": "sample_occlusion_log_loss",
        },
    )

    projection = build_training_status_projection(tmp_path)

    assert projection["explainability"] == {"status": "temporarily_hidden"}


def test_projects_random_forest_oob_selection_separately_from_validation(tmp_path: Path) -> None:
    _write_json(
        tmp_path,
        "config.json",
        {"evaluation_strategy": "stratified_holdout", "model_type": "random_forest"},
    )
    _write_json(tmp_path, "status.json", {"model_family": "traditional_ml"})
    _write_json(
        tmp_path,
        "cv_metrics.json",
        {
            "folds": [
                {
                    "fold_index": 1,
                    "best_params": {"random_forest_max_depth": 5},
                    "selection_metric": "oob_balanced_accuracy",
                    "selection_score": 0.87,
                    "split_metrics": {"valid": {"balanced_accuracy": 0.79}},
                }
            ]
        },
    )

    projection = build_training_status_projection(tmp_path)

    assert projection["traditional"]["best_params_by_fold"] == [
        {
            "fold_index": 1,
            "params": {"random_forest_max_depth": 5},
            "selection_metric": "oob_balanced_accuracy",
            "selection_score": 0.87,
            "valid_balanced_accuracy": 0.79,
        }
    ]


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
    assert legacy_projection["explainability"] == {"status": "temporarily_hidden"}


def test_project_status_attaches_audit_without_reading_its_previous_status(tmp_path: Path) -> None:
    config = {"evaluation_strategy": "external_test_holdout", "model_type": "svm"}
    queued = RunRecord(
        run_id="audit-run",
        state="queued",
        version=1,
        dataset_id=None,
        legacy_data_path=None,
        config=config,
    )

    queued_payload = project_status(tmp_path, queued)

    assert queued_payload["training_audit"]["evaluation_strategy"] == "external_test_holdout"
    assert queued_payload["training_audit"]["explainability"] == {"status": "temporarily_hidden"}

    _write_json(tmp_path, "model_metadata.json", {"model_family": "traditional_ml"})
    _write_json(
        tmp_path,
        "cv_metrics.json",
        {
            "folds": [
                {
                    "fold_index": 0,
                    "best_params": {"svm_c": 1.0},
                    "selection_metric": "balanced_accuracy",
                    "selection_score": 0.88,
                }
            ]
        },
    )
    _write_json(tmp_path, "status.json", {"model_family": "deep_learning", "history": [{"best_valid_loss": 0.1}]})
    succeeded = RunRecord(
        run_id="audit-run",
        state="succeeded",
        version=2,
        dataset_id=None,
        legacy_data_path=None,
        config=config,
    )

    payload = project_status(tmp_path, succeeded)

    assert payload["training_audit"]["traditional"] == {
        "hyperparameter_search_csv": {"artifact": "hyperparameter_search.csv", "available": False},
        "best_params": {"svm_c": 1.0},
        "selection_metric": "balanced_accuracy",
        "selection_score": 0.88,
        "valid_balanced_accuracy": 0.88,
    }
    assert "deep_training" not in payload["training_audit"]
    assert "fold" not in json.dumps(payload["training_audit"]).lower()
    assert json.loads((tmp_path / "status.json").read_text(encoding="utf-8")) == payload

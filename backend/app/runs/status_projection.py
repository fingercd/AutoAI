from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from csv import DictReader
from math import isfinite
from pathlib import Path
from typing import Any

from .contracts import RunRecord


def _atomic_json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False, suffix='.tmp')
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def project_status(run_dir: Path, record: RunRecord, **fields: Any) -> dict[str, Any]:
    payload = {
        'run_id': record.run_id,
        'status': record.legacy_status,
        'state': record.state,
        'version': record.version,
        'dataset_id': record.dataset_id,
        **fields,
    }
    _atomic_json_write(run_dir / 'status.json', payload)
    return payload


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _merged_file_mapping(path: Path, override: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = _read_json_mapping(path)
    payload.update(_mapping(override))
    return payload


def _first_present(*values: object) -> object | None:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _finite_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None


def _history_rows(run_dir: Path, status: Mapping[str, Any]) -> list[dict[str, Any]]:
    history = status.get("history")
    if isinstance(history, list):
        return [_mapping(row) for row in history if isinstance(row, Mapping)]
    try:
        with (run_dir / "history.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            return [_mapping(row) for row in DictReader(handle)]
    except OSError:
        return []


def _selected_search_rows(run_dir: Path) -> list[dict[str, Any]]:
    try:
        with (run_dir / "hyperparameter_search.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [_mapping(row) for row in DictReader(handle)]
    except OSError:
        return []
    return [row for row in rows if str(row.get("is_selected", "")).lower() in {"1", "true", "yes"}]


def _selection_entries(run_dir: Path, status: Mapping[str, Any]) -> list[dict[str, Any]]:
    cv_metrics = _read_json_mapping(run_dir / "cv_metrics.json")
    candidates: list[object] = []
    for source in (cv_metrics.get("folds"), status.get("best_params_by_fold"), _read_json_mapping(run_dir / "split.json")):
        if isinstance(source, list):
            candidates = source
            if candidates:
                break

    entries: list[dict[str, Any]] = []
    for position, raw in enumerate(candidates):
        item = _mapping(raw)
        params = _mapping(item.get("best_params"))
        selection_score = _finite_float(item.get("selection_score"))
        split_metrics = _mapping(item.get("split_metrics"))
        valid_metrics = _mapping(split_metrics.get("valid"))
        valid_balanced_accuracy = _first_present(
            selection_score if item.get("selection_metric") == "balanced_accuracy" else None,
            _finite_float(item.get("valid_balanced_accuracy")),
            _finite_float(valid_metrics.get("balanced_accuracy")),
        )
        if params or valid_balanced_accuracy is not None:
            entries.append(
                {
                    "fold_index": item.get("fold_index", position),
                    "params": params,
                    "valid_balanced_accuracy": valid_balanced_accuracy,
                }
            )

    if entries:
        return entries

    for position, row in enumerate(_selected_search_rows(run_dir)):
        try:
            params = _mapping(json.loads(str(row.get("params_json") or "{}")))
        except json.JSONDecodeError:
            params = {}
        valid_balanced_accuracy = _finite_float(row.get("valid_balanced_accuracy"))
        if params or valid_balanced_accuracy is not None:
            entries.append(
                {
                    "fold_index": row.get("fold_index", position),
                    "params": params,
                    "valid_balanced_accuracy": valid_balanced_accuracy,
                }
            )
    return entries


def _importance_summary(run_dir: Path, status: Mapping[str, Any], name: str) -> dict[str, Any]:
    summary = _mapping(status.get(name))
    if summary:
        return summary
    return _read_json_mapping(run_dir / f"{name}.json")


def _explainability_projection(
    run_dir: Path,
    *,
    config: Mapping[str, Any],
    status: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    feature = _importance_summary(run_dir, status, "feature_importance")
    sample = _importance_summary(run_dir, status, "sample_feature_importance")
    declared_method = _first_present(
        model_metadata.get("explainability_method"),
        status.get("explainability_method"),
        config.get("explainability_method"),
    )
    artifact_method = _first_present(
        model_metadata.get("artifact_explainability_method"),
        status.get("artifact_explainability_method"),
        config.get("artifact_explainability_method"),
        sample.get("method"),
        feature.get("method"),
    )
    importance_metric = _first_present(sample.get("importance_metric"), feature.get("importance_metric"))
    if declared_method is None and artifact_method is None and importance_metric is None:
        return {}
    return {
        "declared_method": declared_method,
        "artifact_method": artifact_method,
        "importance_metric": importance_metric,
    }


def build_training_status_projection(
    run_dir: Path,
    *,
    config: Mapping[str, Any] | None = None,
    status: Mapping[str, Any] | None = None,
    model_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a read-only, UI-ready audit summary for a completed or legacy Run.

    The projection deliberately tolerates partially written and pre-v2 Run directories.
    In particular, external-test holdout results use singular selection fields and never
    expose CV/fold wording to callers.
    """

    run_path = Path(run_dir)
    config_payload = _merged_file_mapping(run_path / "config.json", config)
    status_payload = _merged_file_mapping(run_path / "status.json", status)
    metadata_payload = _merged_file_mapping(run_path / "model_metadata.json", model_metadata)

    evaluation_strategy = _first_present(
        status_payload.get("evaluation_strategy"), config_payload.get("evaluation_strategy")
    )
    model_type = _first_present(
        status_payload.get("model_type"), metadata_payload.get("model_type"), config_payload.get("model_type")
    )
    model_family = _first_present(
        status_payload.get("model_family"), metadata_payload.get("model_family")
    )
    projection: dict[str, Any] = {
        "model_type": model_type,
        "model_family": model_family,
        "architecture_version": _first_present(
            status_payload.get("architecture_version"),
            metadata_payload.get("architecture_version"),
            config_payload.get("architecture_version"),
        ),
        "evaluation_strategy": evaluation_strategy,
        "explainability": _explainability_projection(
            run_path,
            config=config_payload,
            status=status_payload,
            model_metadata=metadata_payload,
        ),
    }

    selection_entries = _selection_entries(run_path, status_payload)
    if model_family == "traditional_ml":
        search_csv = {
            "artifact": "hyperparameter_search.csv",
            "available": (run_path / "hyperparameter_search.csv").is_file(),
        }
        if evaluation_strategy == "external_test_holdout":
            selected = selection_entries[0] if selection_entries else {}
            traditional: dict[str, Any] = {"hyperparameter_search_csv": search_csv}
            if selected:
                traditional["best_params"] = selected["params"]
                traditional["valid_balanced_accuracy"] = selected["valid_balanced_accuracy"]
            projection["traditional"] = traditional
        else:
            projection["traditional"] = {
                "best_params_by_fold": selection_entries,
                "hyperparameter_search_csv": search_csv,
            }

    history = _history_rows(run_path, status_payload)
    deep_loss_values = [
        value
        for row in history
        for value in (_finite_float(row.get("best_valid_loss")), _finite_float(row.get("valid_loss")))
        if value is not None
    ]
    learning_rates = [
        value
        for row in history
        for value in (_finite_float(row.get("learning_rate")),)
        if value is not None
    ]
    is_deep = model_family == "deep_learning" or bool(deep_loss_values)
    if is_deep:
        actual_epochs = status_payload.get("actual_epochs")
        if not isinstance(actual_epochs, int):
            actual_epochs = len(history)
        projection["deep_training"] = {
            "best_valid_loss": min(deep_loss_values) if deep_loss_values else None,
            "actual_epochs": actual_epochs,
            "min_learning_rate": min(learning_rates) if learning_rates else None,
        }
    return projection

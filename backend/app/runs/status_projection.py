from __future__ import annotations

import csv
import json
import math
import os
import tempfile
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
    status_path = run_dir / 'status.json'
    payload: dict[str, Any] = {}
    if status_path.is_file():
        try:
            loaded = json.loads(status_path.read_text(encoding='utf-8'))
            if isinstance(loaded, dict):
                payload.update(loaded)
        except (json.JSONDecodeError, OSError):
            pass
    payload.update({
        'run_id': record.run_id,
        'status': record.legacy_status,
        'state': record.state,
        'version': record.version,
        'dataset_id': record.dataset_id,
    })
    if record.error:
        payload['error'] = record.error
    else:
        payload.pop('error', None)
    if record.manifest_name:
        payload['manifest_name'] = record.manifest_name
    payload.update(fields)
    _atomic_json_write(status_path, payload)
    return payload


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_history_value(value: str | None) -> Any:
    if value is None or value == '':
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as handle:
            return [
                {key: _parse_history_value(value) for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
    except (OSError, csv.Error):
        return []


def _recover_sample_count(config: dict[str, Any]) -> int | None:
    data_path = config.get('data_path')
    if not data_path:
        return None
    path = Path(str(data_path))
    if not path.is_file():
        return None
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            required = ('Index', 'Name', 'Label', 'Repeat_index')
            if not reader.fieldnames or not all(name in reader.fieldnames for name in required):
                return None
            samples = {tuple(row.get(name, '') for name in required) for row in reader}
    except (OSError, csv.Error):
        return None
    return len(samples) or None


def recover_status_from_artifacts(run_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get('status') != 'success' and payload.get('state') != 'succeeded':
        return payload
    recovered = dict(payload)
    changed = False

    metrics = _read_json_object(run_dir / 'metrics.json')
    if metrics and not recovered.get('metrics'):
        recovered['metrics'] = metrics
        changed = True

    history = _read_history(run_dir / 'history.csv')
    if history and not recovered.get('history'):
        recovered['history'] = history
        recovered['actual_epochs'] = len(history)
        changed = True

    config = _read_json_object(run_dir / 'config.json')
    if config:
        for key in ('model_type', 'evaluation_strategy', 'fold_count'):
            if recovered.get(key) is None and config.get(key) is not None:
                recovered[key] = config[key]
                changed = True
        if not recovered.get('config'):
            recovered['config'] = config
            changed = True
        if recovered.get('target_epochs') is None and config.get('epochs') is not None:
            recovered['target_epochs'] = config['epochs']
            changed = True
        if recovered.get('total_target_epochs') is None and config.get('epochs') is not None:
            fold_count = int(config.get('fold_count') or 1)
            recovered['total_target_epochs'] = fold_count * int(config['epochs'])
            changed = True
        if recovered.get('sample_count') is None:
            sample_count = _recover_sample_count(config)
            if sample_count is not None:
                recovered['sample_count'] = sample_count
                changed = True

    manifest = _read_json_object(run_dir / 'manifest.json')
    metadata = manifest.get('metadata') if isinstance(manifest.get('metadata'), dict) else {}
    artifacts = manifest.get('artifacts') if isinstance(manifest.get('artifacts'), dict) else {}
    for key in ('model_type', 'model_family', 'evaluation_strategy', 'fold_count'):
        if recovered.get(key) is None and metadata.get(key) is not None:
            recovered[key] = metadata[key]
            changed = True
    if recovered.get('completed_at') is None and manifest.get('created_at'):
        recovered['completed_at'] = manifest['created_at']
        changed = True
    if recovered.get('model_artifact') is None:
        model_artifact = next((name for name in ('model.pkl', 'model.pt') if name in artifacts), None)
        if model_artifact:
            recovered['model_artifact'] = model_artifact
            changed = True

    for field, json_name, csv_name in (
        ('feature_importance', 'feature_importance.json', 'feature_importance.csv'),
        ('sample_feature_importance', 'sample_feature_importance.json', 'sample_feature_importance.csv'),
    ):
        if recovered.get(field) is None and json_name in artifacts:
            summary: dict[str, Any] = {'status': 'ready', 'artifact': json_name}
            if csv_name in artifacts:
                summary['csv_artifact'] = csv_name
            recovered[field] = summary
            changed = True

    label_map = _read_json_object(run_dir / 'label_map.json')
    if recovered.get('label_names') is None and label_map:
        try:
            recovered['label_names'] = [label_map[key] for key in sorted(label_map, key=lambda item: int(item))]
            changed = True
        except (TypeError, ValueError, KeyError):
            pass

    if changed:
        _atomic_json_write(run_dir / 'status.json', recovered)
    return recovered

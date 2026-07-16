"""把规范 Run、真实训练产物和 Manifest 投影为稳定的结果页契约。"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .artifacts import ManifestCorruptError, RunArtifactWriter
from .contracts import RunRecord, public_error_message
from .status_projection import build_training_status_projection


RESULT_SCHEMA_VERSION = 'run-result-v1'
SCALAR_METRIC_KEYS = (
    'accuracy',
    'balanced_accuracy',
    'macro_precision',
    'macro_recall',
    'macro_f1',
    'weighted_f1',
)


def _read_json_object(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        warnings.append(f'{path.name} 无法解析')
        return {}
    if not isinstance(payload, dict):
        warnings.append(f'{path.name} 不是对象')
        return {}
    return payload


def _without_paths(value: Any) -> Any:
    """递归移除服务器绝对路径字段，避免结果 API 暴露部署目录。"""
    if isinstance(value, dict):
        return {
            key: _without_paths(item)
            for key, item in value.items()
            if key not in {'data_path', 'test_data_path', 'run_dir', 'path'}
            and not key.endswith('_path')
        }
    if isinstance(value, list):
        return [_without_paths(item) for item in value]
    return value


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None


def _duration_seconds(record: RunRecord) -> float | None:
    started = _parse_datetime(record.started_at)
    finished = _parse_datetime(record.finished_at)
    if started is None or finished is None:
        return None
    return max(0.0, (finished - started).total_seconds())


def _duration_between(started_value: str | None, finished_value: str | None) -> float | None:
    started = _parse_datetime(started_value)
    finished = _parse_datetime(finished_value)
    if started is None or finished is None:
        return None
    return max(0.0, (finished - started).total_seconds())


def _coerce_csv(value: str | None) -> Any:
    if value is None or value == '':
        return None
    lowered = value.casefold()
    if lowered in {'true', 'false'}:
        return lowered == 'true'
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _read_history(path: Path, warnings: list[str], *, limit: int = 1000) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        return {
            'available': False,
            'rows': [],
            'columns': [],
            'truncated': False,
            'reason': '本次训练未生成训练过程数据',
        }
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            rows: list[dict[str, Any]] = []
            truncated = False
            for index, row in enumerate(reader):
                if index >= limit:
                    truncated = True
                    break
                rows.append({key: _coerce_csv(value) for key, value in row.items() if key is not None})
            columns = list(reader.fieldnames or [])
    except (OSError, UnicodeDecodeError, csv.Error):
        warnings.append('history.csv 无法解析')
        return {
            'available': False,
            'rows': [],
            'columns': [],
            'truncated': False,
            'reason': '训练过程文件损坏或无法读取',
        }
    return {
        'available': bool(rows),
        'rows': rows,
        'columns': columns,
        'truncated': truncated,
        'reason': None if rows else '该模型没有可展示的训练过程曲线',
    }


def _metric_scalars(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {key: payload.get(key) for key in SCALAR_METRIC_KEYS if payload.get(key) is not None}


def _prediction_distribution(confusion: Any, labels: list[str]) -> dict[str, Any] | None:
    if not isinstance(confusion, list) or not confusion:
        return None
    try:
        rows = [[int(value) for value in row] for row in confusion]
        width = len(rows[0])
        if width == 0 or any(len(row) != width for row in rows):
            return None
        true_counts = [sum(row) for row in rows]
        predicted_counts = [sum(row[index] for row in rows) for index in range(width)]
    except (TypeError, ValueError, IndexError):
        return None
    resolved_labels = labels if len(labels) == len(true_counts) else [str(index) for index in range(len(true_counts))]
    return {
        'labels': resolved_labels,
        'true_counts': true_counts,
        'predicted_counts': predicted_counts,
    }


def _result_state(
    record: RunRecord,
    *,
    manifest: dict[str, Any] | None,
    descriptors: list[dict[str, Any]],
    manifest_error: str | None,
) -> str:
    if record.state == 'queued':
        return 'pending'
    if record.state == 'running':
        return 'running'
    if record.state == 'failed':
        return 'failed'
    if record.state == 'cancelled':
        return 'cancelled'
    if manifest_error == 'missing':
        return 'missing_manifest'
    if manifest_error == 'corrupt':
        return 'corrupt_manifest'
    if manifest is None:
        return 'missing_manifest'
    damaged = any(
        item['integrity'] in {'missing', 'corrupt'}
        and (item['required'] or item['applicable'])
        for item in descriptors
    )
    return 'partial' if damaged else 'ready'


def project_run_result(record: RunRecord, *, run_dir: Path, dataset_name: str | None = None) -> dict[str, Any]:
    """生成可刷新、可降级、不会泄露服务器路径的结果页响应。"""
    run_dir = Path(run_dir)
    warnings: list[str] = []
    status = _read_json_object(run_dir / 'status.json', warnings)
    is_succeeded = record.state == 'succeeded'
    metrics_file = _read_json_object(run_dir / 'metrics.json', warnings) if is_succeeded else {}
    cv_file = _read_json_object(run_dir / 'cv_metrics.json', warnings) if is_succeeded else {}
    model_metadata = _read_json_object(run_dir / 'model_metadata.json', warnings) if is_succeeded else {}
    label_map = _read_json_object(run_dir / 'label_map.json', warnings) if is_succeeded else {}

    manifest: dict[str, Any] | None = None
    descriptors: list[dict[str, Any]] = []
    manifest_error: str | None = None
    artifact_writer = RunArtifactWriter(run_dir)
    try:
        manifest, descriptors = artifact_writer.descriptors(run_id=record.run_id)
        if manifest.get('schema_version') != 'run-artifact-manifest-v2':
            warnings.append('该任务使用历史 Manifest，结果完整性按兼容模式解释')
    except FileNotFoundError:
        manifest_error = 'missing'
        unavailable_reason = (
            '训练尚未成功完成，结果文件未发布'
            if not is_succeeded
            else 'Manifest 缺失，无法验证或下载文件'
        )
        descriptors = artifact_writer.unavailable_descriptors(
            run_id=record.run_id,
            reason=unavailable_reason,
        )
        if record.state == 'succeeded':
            warnings.append('Manifest 缺失，下载功能不可用')
    except ManifestCorruptError:
        manifest_error = 'corrupt'
        descriptors = artifact_writer.unavailable_descriptors(
            run_id=record.run_id,
            reason='Manifest 损坏，无法验证或下载文件',
        )
        warnings.append('Manifest 损坏，下载功能不可用')

    if not is_succeeded:
        for descriptor in descriptors:
            descriptor['downloadable'] = False
            descriptor['download_url'] = None
            descriptor['reason'] = '训练尚未成功完成，结果文件不可下载'

    result_state = _result_state(
        record,
        manifest=manifest,
        descriptors=descriptors,
        manifest_error=manifest_error,
    )
    if result_state == 'partial':
        warnings.append('部分结果文件缺失或完整性校验失败')

    strategy = str(
        status.get('evaluation_strategy')
        or record.config.get('evaluation_strategy')
        or cv_file.get('strategy')
        or 'stratified_holdout'
    )
    fold_count = status.get('fold_count') or cv_file.get('fold_count')
    cv_summary = cv_file.get('cv_summary') if isinstance(cv_file.get('cv_summary'), dict) else {}
    if is_succeeded and not cv_summary and isinstance(status.get('cv_summary'), dict):
        cv_summary = status['cv_summary']

    raw_metrics = metrics_file or (
        status.get('metrics')
        if is_succeeded and isinstance(status.get('metrics'), dict)
        else {}
    )
    raw_split_metrics = {
        name: _without_paths(raw_metrics.get(name, {}))
        for name in ('train', 'valid', 'test')
        if isinstance(raw_metrics.get(name), dict)
    }
    is_cv = strategy == 'leave_one_sample_id_cv'
    pooled_oof = cv_summary.get('pooled_test') if is_cv and isinstance(cv_summary.get('pooled_test'), dict) else None
    fold_mean = cv_summary.get('fold_mean') if isinstance(cv_summary.get('fold_mean'), dict) else {}
    fold_std = cv_summary.get('fold_std') if isinstance(cv_summary.get('fold_std'), dict) else {}
    if is_cv:
        split_metrics = {
            name: {
                'aggregation': 'pooled_oof' if name == 'test' else 'fold_mean',
                'values': (
                    _without_paths(pooled_oof)
                    if name == 'test'
                    else _without_paths(fold_mean.get(name, {}))
                ),
                'fold_std': _without_paths(fold_std.get(name, {})),
                'pooled': _without_paths(
                    pooled_oof if name == 'test' else raw_split_metrics.get(name, {})
                ),
            }
            for name in ('train', 'valid', 'test')
        }
        direct = None
        primary = pooled_oof or raw_split_metrics.get('test') or raw_metrics
    else:
        split_metrics = {
            name: {
                'aggregation': 'direct',
                'values': payload,
                'fold_std': None,
                'pooled': None,
            }
            for name, payload in raw_split_metrics.items()
        }
        direct = raw_split_metrics
        primary = raw_split_metrics.get('test') or raw_metrics

    labels: list[str] = []
    if label_map:
        try:
            labels = [str(label_map[key]) for key in sorted(label_map, key=lambda key: int(key))]
        except (TypeError, ValueError):
            labels = [str(value) for value in label_map.values()]
    if not labels and is_succeeded and isinstance(status.get('label_names'), list):
        labels = [str(value) for value in status['label_names']]

    snapshot = dict(record.dataset_snapshot)
    dataset_payload = {
        'dataset_id': record.dataset_id,
        'name': snapshot.get('name') or dataset_name or record.config.get('dataset_name'),
        'sha256': snapshot.get('sha256'),
        'curve_count': snapshot.get('curve_count') or status.get('curve_count') or status.get('sample_count'),
        'sample_id_count': snapshot.get('sample_id_count') or status.get('sample_id_count'),
        'class_count': snapshot.get('class_count') or status.get('class_count') or (len(labels) or None),
        'feature_count': snapshot.get('feature_count') or status.get('feature_count') or model_metadata.get('feature_count'),
        'test_curve_count': snapshot.get('test_curve_count') or status.get('test_sample_count'),
    }

    error = None
    if record.error or record.error_details:
        error = {
            'code': record.error_details.get('code') or 'training_failed',
            'stage': record.error_details.get('stage') or 'training',
            'message': public_error_message(record.error_details.get('message') or record.error),
            'retryable': bool(record.error_details.get('retryable', False)),
            'type': record.error_details.get('type'),
        }

    confusion = primary.get('confusion_matrix') if isinstance(primary, dict) else None
    classification_report = primary.get('classification_report') if isinstance(primary, dict) else None
    model_type = status.get('model_type') or record.config.get('model_type') or model_metadata.get('model_type')
    model_family = status.get('model_family') or model_metadata.get('model_family')
    if is_succeeded and model_family == 'deep_learning':
        history = _read_history(run_dir / 'history.csv', warnings)
    else:
        history = {
            'available': False,
            'rows': [],
            'columns': [],
            'truncated': False,
            'reason': (
                '训练尚未成功完成，暂无训练过程结果'
                if not is_succeeded
                else '传统模型没有 epoch 训练曲线，请查看参数选择审计'
            ),
        }
    training_audit = (
        build_training_status_projection(
            run_dir,
            config=record.config,
            status=status,
            model_metadata=model_metadata,
            read_status_file=False,
        )
        if is_succeeded
        else {}
    )
    started_at = record.started_at or status.get('started_at')
    finished_at = record.finished_at or (
        status.get('completed_at')
        if record.state in {'succeeded', 'failed', 'cancelled'}
        else None
    )
    duration_seconds = _duration_seconds(record)
    if duration_seconds is None:
        duration_seconds = _duration_between(started_at, finished_at)
        if duration_seconds is not None:
            warnings.append('训练耗时来自历史 status.json 时间，数据库未记录完整终态时间')

    return {
        'schema_version': RESULT_SCHEMA_VERSION,
        'run': {
            'run_id': record.run_id,
            'state': record.state,
            'status': record.legacy_status,
            'result_state': result_state,
            'created_at': record.created_at or status.get('created_at'),
            'started_at': started_at,
            'finished_at': finished_at,
            'duration_seconds': duration_seconds,
            'progress': _without_paths(record.progress),
            'error': error,
        },
        'dataset': _without_paths(dataset_payload),
        'model': {
            'type': model_type,
            'family': model_family,
            'architecture_version': status.get('architecture_version') or model_metadata.get('architecture_version'),
            'artifact_note': status.get('model_artifact_note'),
        },
        'evaluation': {
            'strategy': strategy,
            'fold_count': int(fold_count) if fold_count is not None else None,
            'primary_split': 'test',
            'primary_aggregation': 'pooled_oof' if is_cv else 'direct',
        },
        'metrics': {
            'primary': _without_paths(primary),
            'splits': split_metrics,
            'direct': direct,
            'fold_mean': _without_paths(fold_mean),
            'fold_std': _without_paths(fold_std),
            'pooled_oof': _without_paths(pooled_oof),
            'cv': {
                'pooled_oof': _without_paths(pooled_oof),
                'fold_mean': _without_paths(fold_mean),
                'fold_std': _without_paths(fold_std),
            },
        },
        'analysis': {
            'confusion_matrix': confusion,
            'classification_report': _without_paths(classification_report),
            'prediction_distribution': _prediction_distribution(confusion, labels),
            'history': history,
            'training_audit': _without_paths(training_audit),
            'roc': {'available': False, 'reason': '当前训练产物未计算 ROC 曲线或 ROC-AUC'},
            'precision_recall': {'available': False, 'reason': '当前训练产物未计算 Precision-Recall 曲线'},
        },
        'explainability': {
            'global': _without_paths(status.get('feature_importance')) if is_succeeded else None,
            'samples': _without_paths(status.get('sample_feature_importance')) if is_succeeded else None,
        },
        'artifacts': descriptors,
        'warnings': list(dict.fromkeys(warnings)),
    }

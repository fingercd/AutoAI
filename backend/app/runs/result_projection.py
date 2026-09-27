"""把规范 Run、真实训练产物和 Manifest 投影为稳定的结果页契约（run-result-v1）。

系统位置：runs 包的“结果页投影层”，对应 GET /api/training/runs/{run_id}/result。
与 status_projection 的单向写投影不同，本模块是只读投影：每次请求实时从
status.json / metrics.json / cv_metrics.json / manifest 等产物组装响应。

关键设计约束：

- 可刷新、可降级：产物缺失/损坏时降级为 warnings + unavailable，而不是 500；
- 不泄露服务器路径：所有外挂 payload 递归经 _without_paths 过滤；
- CV 口径严格区分 pooled OOF（主指标）、fold_mean 与 fold_std；
- 当前没有 ROC / Precision-Recall 产物，固定返回 available=False，前端不得伪造空图；
- 未成功 Run 不读取任何训练产物，artifact 一律不可下载。
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..version import ARTIFACT_MANIFEST_CONTRACT_VERSION, RUN_RESULT_CONTRACT_VERSION
from .artifacts import ManifestCorruptError, RunArtifactWriter
from .contracts import RunRecord, public_error_message
from .status_projection import build_training_status_projection


# schema_version 直接复用全局契约常量，保证结果页契约只有一处定义。
RESULT_SCHEMA_VERSION = RUN_RESULT_CONTRACT_VERSION
# 六个主指标键：投影只暴露这些标量，防止内部中间指标外泄。
SCALAR_METRIC_KEYS = (
    'accuracy',
    'balanced_accuracy',
    'macro_precision',
    'macro_recall',
    'macro_f1',
    'weighted_f1',
)


def _read_json_object(path: Path, warnings: list[str]) -> dict[str, Any]:
    """读取 JSON 对象；解析失败记录 warning 并返回 {}（结果页降级：能展示多少展示多少）。"""
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


def _read_json_list(path: Path, warnings: list[str]) -> list[Any]:
    """读取 JSON 数组；失败时记录 warning 并返回 []。"""
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        warnings.append(f'{path.name} 无法解析')
        return []
    if not isinstance(payload, list):
        warnings.append(f'{path.name} 不是数组')
        return []
    return payload


def _without_paths(value: Any) -> Any:
    """递归移除服务器绝对路径字段，避免结果 API 暴露部署目录。

    同时剔除字面键 data_path/test_data_path/run_dir/path 以及所有 *_path 结尾的键；
    列表与字典递归处理，标量原样返回。这是 server 模式安全契约的一部分。
    """
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
    """解析 ISO 时间戳，兼容结尾 Z；非法输入返回 None 而非抛异常（历史数据可能脏）。"""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None


def _duration_seconds(record: RunRecord) -> float | None:
    """由规范 RunRecord 的起止时间算耗时（秒），钳制为非负；缺任一时间返回 None。"""
    started = _parse_datetime(record.started_at)
    finished = _parse_datetime(record.finished_at)
    if started is None or finished is None:
        return None
    return max(0.0, (finished - started).total_seconds())


def _duration_between(started_value: str | None, finished_value: str | None) -> float | None:
    """同上，但接受原始字符串——用于数据库时间缺失时回退 status.json 的历史时间。"""
    started = _parse_datetime(started_value)
    finished = _parse_datetime(finished_value)
    if started is None or finished is None:
        return None
    return max(0.0, (finished - started).total_seconds())


def _coerce_csv(value: str | None) -> Any:
    """把 CSV 单元格文本尽力转成 bool/int/float，转不动就保留原字符串。"""
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
    """读取 history.csv 为前端可直接画曲线的结构。

    limit 截断防止超大文件拖垮结果页；文件缺失/空文件与解析失败给出不同的
    reason 文案；available=False 时 rows 一定为空。
    """
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
    """从任意 metrics payload 抽取白名单标量指标；非 dict 或缺键安全跳过。"""
    if not isinstance(payload, dict):
        return {}
    return {key: payload.get(key) for key in SCALAR_METRIC_KEYS if payload.get(key) is not None}


def _prediction_distribution(confusion: Any, labels: list[str]) -> dict[str, Any] | None:
    """由混淆矩阵推导真实/预测类别计数，供前端画分布图。

    矩阵非方阵或含非整数值时返回 None（不展示比展示错数据好）；labels 数量
    对不上时回退为数字序号，保证图形始终可渲染。
    """
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


def _sample_id_count_from_splits(
    folds: list[Any],
    *,
    evaluation_strategy: str,
) -> int | None:
    """从公开的划分 artifact 恢复主数据集 Sample_ID 数，不读取原始 CSV。

    external_test_holdout 的 test_sample_ids 属于独立测试集，不计入主数据；
    空集合返回 None 表示“未知”，避免与真实的 0 混淆。
    """
    sample_ids: set[str] = set()
    allowed_fields = {'train_sample_ids', 'valid_sample_ids'}
    if evaluation_strategy != 'external_test_holdout':
        allowed_fields.add('test_sample_ids')
    for raw_fold in folds:
        if not isinstance(raw_fold, dict):
            continue
        for field in allowed_fields:
            values = raw_fold.get(field)
            if isinstance(values, list):
                sample_ids.update(str(value) for value in values if value is not None)
    return len(sample_ids) or None


def _analysis_split(
    payload: Any,
    *,
    labels: list[str],
    aggregation: str,
) -> dict[str, Any]:
    """组装单个划分（train/valid/test）的图表数据：混淆矩阵、分类报告、预测分布。

    aggregation 标明口径：direct（单次划分）、pooled_oof（CV 合并 OOF）、
    pooled_cross_fold（CV 跨折合并的 train/valid）。
    """
    split_payload = payload if isinstance(payload, dict) else {}
    confusion = split_payload.get('confusion_matrix')
    return {
        'aggregation': aggregation,
        'confusion_matrix': _without_paths(confusion),
        'classification_report': _without_paths(split_payload.get('classification_report')),
        'prediction_distribution': _prediction_distribution(confusion, labels),
    }


def _sample_explainability_summary(
    status: dict[str, Any],
    descriptors: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """汇总单样品特征重要性的可用状态。

    优先取 status.json 内嵌摘要；否则看 manifest 描述符：json 存在且完整性
    ok/volatile 视为 ready，并附带同名 csv 下载线索；都没有则返回 None。
    """
    summary = status.get('sample_feature_importance')
    if isinstance(summary, dict):
        return _without_paths(summary)
    json_descriptor = next(
        (
            item
            for item in descriptors
            if item.get('name') == 'sample_feature_importance.json'
        ),
        None,
    )
    if not isinstance(json_descriptor, dict) or not json_descriptor.get('exists'):
        return None
    csv_descriptor = next(
        (
            item
            for item in descriptors
            if item.get('name') == 'sample_feature_importance.csv'
            and item.get('exists')
        ),
        None,
    )
    integrity = str(json_descriptor.get('integrity') or '')
    return {
        'status': 'ready' if integrity in {'ok', 'volatile'} else 'unavailable',
        'reason': json_descriptor.get('reason'),
        'artifact': 'sample_feature_importance.json',
        'csv_artifact': (
            'sample_feature_importance.csv'
            if isinstance(csv_descriptor, dict)
            else None
        ),
    }


def _result_state(
    record: RunRecord,
    *,
    manifest: dict[str, Any] | None,
    descriptors: list[dict[str, Any]],
    manifest_error: str | None,
) -> str:
    """把规范 state + manifest 健康状况折叠成结果页 result_state。

    进行中：pending/running/failed/cancelled 直接映射；成功后再看 manifest：
    缺失→missing_manifest，损坏→corrupt_manifest；必需/适用 artifact 有
    missing/corrupt → partial，否则 ready。
    """
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
    """生成可刷新、可降级、不会泄露服务器路径的结果页响应。

    组装顺序：读状态与产物（仅成功 Run）→ 校验 manifest → 定 result_state →
    按评估口径（CV 用 pooled OOF 作主指标）整理 metrics → 数据集快照 →
    分析图表 → 训练审计 → 时间/耗时。全程收集 warnings 并在末尾去重返回。
    """
    run_dir = Path(run_dir)
    warnings: list[str] = []
    status = _read_json_object(run_dir / 'status.json', warnings)
    # 未成功 Run 一律不读训练产物：半成品不具备展示资格，也避免读到写一半的文件。
    is_succeeded = record.state == 'succeeded'

    manifest: dict[str, Any] | None = None
    descriptors: list[dict[str, Any]] = []
    manifest_error: str | None = None
    artifact_writer = RunArtifactWriter(run_dir)
    try:
        manifest, descriptors = artifact_writer.descriptors(run_id=record.run_id)
        if manifest.get('schema_version') != ARTIFACT_MANIFEST_CONTRACT_VERSION:
            warnings.append('该 Run 的产物清单由旧版 Worker 生成，已按兼容策略读取')
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

    def verified_json(name, expected_type):
        if is_succeeded and manifest is not None:
            try:
                value = artifact_writer.read_verified_json(manifest, name)
                if isinstance(value, expected_type):
                    return value
            except (OSError, ValueError, UnicodeDecodeError):
                warnings.append(f'{name} 无法通过完整性校验或解析')
        return expected_type()

    metrics_file = verified_json('metrics.json', dict)
    cv_file = verified_json('cv_metrics.json', dict)
    model_metadata = verified_json('model_metadata.json', dict)
    label_map = verified_json('label_map.json', dict)
    split_file = verified_json('split.json', list)

    result_state = _result_state(
        record,
        manifest=manifest,
        descriptors=descriptors,
        manifest_error=manifest_error,
    )
    # partial 说明有必需产物缺失或完整性校验失败，提示用户结果不完整。
    if result_state == 'partial':
        warnings.append('部分结果文件缺失或完整性校验失败')

    # 评估策略多来源择优：status → 训练 config → cv_metrics → 默认 stratified_holdout。
    strategy = str(
        status.get('evaluation_strategy')
        or record.config.get('evaluation_strategy')
        or cv_file.get('strategy')
        or 'stratified_holdout'
    )
    fold_count = status.get('fold_count') or cv_file.get('fold_count')
    cv_summary = cv_file.get('cv_summary') if isinstance(cv_file.get('cv_summary'), dict) else {}
    # Mutable status projections are not authority for scientific measurements.
    raw_metrics = metrics_file
    raw_split_metrics = {
        name: _without_paths(raw_metrics.get(name, {}))
        for name in ('train', 'valid', 'test')
        if isinstance(raw_metrics.get(name), dict)
    }
    is_cv = strategy == 'leave_one_sample_id_cv'
    pooled_oof = cv_summary.get('pooled_test') if is_cv and isinstance(cv_summary.get('pooled_test'), dict) else None
    fold_mean = cv_summary.get('fold_mean') if isinstance(cv_summary.get('fold_mean'), dict) else {}
    fold_std = cv_summary.get('fold_std') if isinstance(cv_summary.get('fold_std'), dict) else {}
    # CV 口径：test 主指标必须是 pooled OOF（所有折合并计算），不能取 fold mean。
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
    # label_map 键是字符串化的整数编码，按数值排序还原类别名顺序。
    if label_map:
        try:
            labels = [str(label_map[key]) for key in sorted(label_map, key=lambda key: int(key))]
        except (TypeError, ValueError):
            labels = [str(value) for value in label_map.values()]
    if not labels and is_succeeded and isinstance(status.get('label_names'), list):
        labels = [str(value) for value in status['label_names']]

    snapshot = dict(record.dataset_snapshot)
    recovered_sample_id_count = _sample_id_count_from_splits(
        split_file,
        evaluation_strategy=strategy,
    )
    dataset_payload = {
        'dataset_id': record.dataset_id,
        'name': snapshot.get('name') or dataset_name or record.config.get('dataset_name'),
        'sha256': snapshot.get('sha256'),
        'curve_count': snapshot.get('curve_count') or status.get('curve_count') or status.get('sample_count'),
        'sample_id_count': (
            snapshot.get('sample_id_count')
            or status.get('sample_id_count')
            or recovered_sample_id_count
        ),
        'class_count': snapshot.get('class_count') or status.get('class_count') or (len(labels) or None),
        'feature_count': (
            snapshot.get('feature_count')
            or status.get('feature_count')
            or model_metadata.get('feature_count')
            or model_metadata.get('L')
        ),
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

    analysis_splits: dict[str, dict[str, Any]] = {}
    for split_name in ('train', 'valid', 'test'):
        chart_payload = raw_split_metrics.get(split_name, {})
        if not chart_payload and split_name == 'test' and isinstance(pooled_oof, dict):
            chart_payload = pooled_oof
        if is_cv:
            aggregation = 'pooled_oof' if split_name == 'test' else 'pooled_cross_fold'
        else:
            aggregation = 'direct'
        analysis_splits[split_name] = _analysis_split(
            chart_payload,
            labels=labels,
            aggregation=aggregation,
        )
    primary_analysis = analysis_splits['test']
    model_type = status.get('model_type') or record.config.get('model_type') or model_metadata.get('model_type')
    model_family = status.get('model_family') or model_metadata.get('model_family')
    # 只有深度模型有 epoch 曲线；传统模型固定返回 available=False 并提示看参数选择审计。
    verified_names = {item['name'] for item in descriptors if item['integrity'] == 'ok'}
    if is_succeeded and model_family == 'deep_learning' and 'history.csv' in verified_names:
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
        if is_succeeded and result_state == 'ready'
        else {}
    )
    started_at = record.started_at or status.get('started_at')
    finished_at = record.finished_at or (
        status.get('completed_at')
        if record.state in {'succeeded', 'failed', 'cancelled'}
        else None
    )
    duration_seconds = _duration_seconds(record)
    # 数据库缺终态时间时回退 status.json，并显式 warning 说明耗时口径不严格。
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
            'search': _without_paths(model_metadata.get('search_summary')) if is_succeeded else None,
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
            'splits': analysis_splits,
            # Test 顶层字段保留一个兼容周期；新页面应读取 analysis.splits。
            'confusion_matrix': primary_analysis['confusion_matrix'],
            'classification_report': primary_analysis['classification_report'],
            'prediction_distribution': primary_analysis['prediction_distribution'],
            'history': history,
            'training_audit': _without_paths(training_audit),
            'roc': {'available': False, 'reason': '当前训练产物未计算 ROC 曲线或 ROC-AUC'},
            'precision_recall': {'available': False, 'reason': '当前训练产物未计算 Precision-Recall 曲线'},
        },
        'explainability': {
            'samples': (
                _sample_explainability_summary(status, descriptors)
                if is_succeeded
                else None
            ),
        },
        'artifacts': descriptors,
        'warnings': list(dict.fromkeys(warnings)),
    }

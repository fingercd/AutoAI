"""把规范 RunRecord 投影成旧前端可读取的状态 JSON。

系统位置：runs 包的“兼容投影层”。训练 Worker 与 SQLite 状态机只维护规范
RunRecord（见 contracts.py），而旧前端、训练记录列表与部分结果页仍读取每个
Run 目录下的 status.json。本模块负责把规范记录单向投影（写）成 status.json，
并在历史目录缺字段时从 artifact 只读恢复（补齐）可展示信息。

协作模块：

- contracts.RunRecord / public_error_message：规范快照与错误脱敏；
- result_projection：结果页投影复用本模块的 build_training_status_projection；
- artifacts / store：提供 manifest、metrics、history 等原始产物。

关键设计约束：

- 投影可以合并训练结果与 artifact 摘要，但不会反向驱动 SQLite 状态机（单向写）。
- 恢复函数只在兼容目录缺少数据库字段时提取可展示信息，不把不完整文件推断成成功 Run。
- 所有写盘经 _atomic_json_write 原子替换，读者不会看到写一半的 JSON。
- 错误字符串一律经 public_error_message 脱敏，禁止泄露服务器绝对路径。
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections.abc import Mapping
from csv import DictReader
from math import isfinite
from pathlib import Path
from typing import Any

from .contracts import RunRecord, public_error_message


def _atomic_json_write(path: Path, payload: object) -> None:
    """原子地把 payload 写成 JSON 文件。

    采用“同目录临时文件 + flush/fsync + os.replace”三段式：临时文件必须与
    目标同目录，rename 才是同分区原子操作；读者任一时刻要么看到旧文件、
    要么看到完整的新文件，杜绝截断 JSON。
    """
    # 先确保目标目录存在（新 Run 目录可能尚未创建）。
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False, suffix='.tmp')
    temporary = Path(handle.name)
    try:
        with handle:
            # ensure_ascii=False 保留中文等字符，indent=2 便于人工排查产物。
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            # fsync 落盘后再替换，避免掉电后目标文件成为空洞。
            os.fsync(handle.fileno())
        # 同目录 rename 在 Windows/POSIX 上都是原子操作。
        os.replace(temporary, path)
    finally:
        # 替换成功后临时文件已不存在；异常路径下兜底删除，避免残留 .tmp。
        if temporary.exists():
            temporary.unlink()


def project_status(run_dir: Path, record: RunRecord, **fields: Any) -> dict[str, Any]:
    """原子写入一个由规范 RunRecord 派生的 status.json 兼容投影。

    参数：
        run_dir: Run 产物目录，status.json 写入其中。
        record: SQLite 中的规范 RunRecord（唯一事实来源）。
        **fields: 调用方追加/覆盖的展示字段（如训练进度），最后合并。

    返回：合并后的完整 payload（同时已落盘）。

    逻辑要点：先读已有 status.json 作底保留历史展示字段；再用规范记录覆盖
    核心字段；dataset_snapshot 只放行白名单统计键；错误信息脱敏写入、无错误时
    显式清除旧 error 字段；最后附加 training_audit 审计投影并原子写盘。
    """
    status_path = run_dir / 'status.json'
    payload: dict[str, Any] = {}
    if status_path.is_file():
        try:
            loaded = json.loads(status_path.read_text(encoding='utf-8'))
            if isinstance(loaded, dict):
                payload.update(loaded)
        except (json.JSONDecodeError, OSError):
            pass
    # 规范记录覆盖核心字段：status/state 双写，同时满足新旧前端读取习惯。
    payload.update({
        'run_id': record.run_id,
        'status': record.legacy_status,
        'state': record.state,
        'version': record.version,
        'dataset_id': record.dataset_id,
        'created_at': record.created_at,
        'started_at': record.started_at,
        'completed_at': record.finished_at,
    })
    # dataset_snapshot 可能含内部字段，只透传白名单统计键。
    if record.dataset_snapshot:
        payload.update(
            {
                key: value
                for key, value in record.dataset_snapshot.items()
                if key in {'curve_count', 'sample_id_count', 'class_count', 'feature_count', 'test_curve_count'}
            }
        )
    # 有错误：message 一律脱敏后写入，error_details.message 同样处理。
    if record.error:
        payload['error'] = public_error_message(record.error)
        if record.error_details:
            payload['error_details'] = {
                **record.error_details,
                'message': public_error_message(record.error_details.get('message') or record.error),
            }
    else:
    # 无错误：清掉旧投影里残留的 error 字段，避免“已成功却仍显示错误”。
        payload.pop('error', None)
        payload.pop('error_details', None)
    if record.manifest_name:
        payload['manifest_name'] = record.manifest_name
    # 仓库中的进度是规范来源；停止原因也保存在这里。调用方 fields 仍可覆盖
    # 同名展示字段，便于训练中的阶段更新。
    payload.update(record.progress)
    payload.update(fields)
    # 新投影不再公开全局重要性；历史 artifact 文件仍由下载层单独兼容。
    payload.pop('feature_importance', None)
    # training_audit 是结果页“训练审计”区块的只读摘要，
    # 与结果页复用同一构造函数以保证口径一致。
    payload["training_audit"] = build_training_status_projection(
        run_dir,
        config=record.config,
        status=payload,
        read_status_file=False,
    )
    _atomic_json_write(status_path, payload)
    return payload


def _read_json_object(path: Path) -> dict[str, Any]:
    """读取 JSON 对象；文件缺失、损坏或不是 dict 时一律返回 {}——不让坏文件炸掉投影。"""
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_json_list(path: Path) -> list[Any]:
    """读取 JSON 数组；缺失/损坏/非 list 时返回 []（与 _read_json_object 同策略）。"""
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return []
    return payload if isinstance(payload, list) else []


def _recover_sample_id_count(folds: list[Any], evaluation_strategy: str) -> int | None:
    """从 split.json 的 fold 列表统计不同 Sample_ID 数。

    external_test_holdout 的 test 来自独立测试集，不属于主数据，故不计入；
    空集合返回 None 而不是 0，避免前端把“未知”显示成“0 个样本”。
    """
    sample_ids: set[str] = set()
    fields = {'train_sample_ids', 'valid_sample_ids'}
    # 独立测试集的 Sample_ID 不属于主数据集统计口径。
    if evaluation_strategy != 'external_test_holdout':
        fields.add('test_sample_ids')
    for raw_fold in folds:
        if not isinstance(raw_fold, dict):
            continue
        for field in fields:
            values = raw_fold.get(field)
            if isinstance(values, list):
                sample_ids.update(str(value) for value in values if value is not None)
    return len(sample_ids) or None


def _parse_history_value(value: str | None) -> Any:
    """把 history.csv 单元格文本转成 int/float；空串与非有限值（NaN/inf）归一为 None。"""
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
    """读取 history.csv 并逐格数值化；utf-8-sig 兼容带 BOM 的 CSV。"""
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
    """直接扫描原始数据 CSV 统计样本（曲线）数，仅在旧目录缺 sample_count 时兜底。

    按已知宽表格式做元组去重：wide-feature-v2 用 (Index, Label, Sample_ID)，
    v1/历史格式回退 (Index, Name, Label, group_field)；表头两种都不符合时返回
    None，不臆造数字。
    """
    data_path = config.get('data_path')
    if not data_path:
        return None
    path = Path(str(data_path))
    if not path.is_file():
        return None
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            group_field = 'Sample_ID' if 'Sample_ID' in fieldnames else 'Repeat_index'
            # wide-feature-v2 直接包含 Name；v1 以 Index 回退，历史六列仍按原字段恢复。
            required = (
                ('Index', 'Label', group_field)
                if tuple(fieldnames[:3]) == ('Index', 'Label', 'Sample_ID')
                else ('Index', 'Name', 'Label', group_field)
            )
            if not all(name in fieldnames for name in required):
                return None
            samples = {tuple(row.get(name, '') for name in required) for row in reader}
    except (OSError, csv.Error):
        return None
    return len(samples) or None


def recover_status_from_artifacts(run_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """从已有 artifact 补齐旧投影缺失的展示字段，不改变规范状态。

    只对“已成功”的 Run 生效（status==success 或 state==succeeded），其余原样返回。
    恢复来源：metrics.json / history.csv / config.json / manifest.json /
    label_map.json / model_metadata.json / split.json。只有确实补齐了字段
    （changed）才回写 status.json，且仍走原子写。注意：这是展示层兜底，
    不会把不完整产物推断成成功，也不会写回数据库。
    """
    # 非成功 Run 不做恢复——失败/进行中的目录不允许被“脑补”成完整结果。
    if payload.get('status') != 'success' and payload.get('state') != 'succeeded':
        return payload
    recovered = dict(payload)
    recovered.pop('feature_importance', None)
    changed = False

    metrics = _read_json_object(run_dir / 'metrics.json')
    if metrics and not recovered.get('metrics'):
        recovered['metrics'] = metrics
        changed = True

    history = _read_history(run_dir / 'history.csv')
    if history:
        if not recovered.get('history'):
            recovered['history'] = history
            changed = True
        if recovered.get('actual_epochs') is None:
            recovered['actual_epochs'] = len(recovered.get('history') or history)
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

    if recovered.get('sample_feature_importance') is None and 'sample_feature_importance.json' in artifacts:
        summary: dict[str, Any] = {
            'status': 'ready',
            'artifact': 'sample_feature_importance.json',
        }
        if 'sample_feature_importance.csv' in artifacts:
            summary['csv_artifact'] = 'sample_feature_importance.csv'
        recovered['sample_feature_importance'] = summary
        changed = True

    label_map = _read_json_object(run_dir / 'label_map.json')
    if recovered.get('label_names') is None and label_map:
        try:
            recovered['label_names'] = [label_map[key] for key in sorted(label_map, key=lambda item: int(item))]
            changed = True
        except (TypeError, ValueError, KeyError):
            pass
    if recovered.get('class_count') is None and label_map:
        recovered['class_count'] = len(label_map)
        changed = True

    model_metadata = _read_json_object(run_dir / 'model_metadata.json')
    if recovered.get('feature_count') is None:
        feature_count = model_metadata.get('feature_count') or model_metadata.get('L')
        if feature_count is not None:
            recovered['feature_count'] = feature_count
            changed = True
    if recovered.get('model_family') is None and model_metadata.get('model_family') is not None:
        recovered['model_family'] = model_metadata['model_family']
        changed = True

    if recovered.get('curve_count') is None and recovered.get('sample_count') is not None:
        recovered['curve_count'] = recovered['sample_count']
        changed = True
    if recovered.get('sample_id_count') is None:
        evaluation_strategy = str(
            recovered.get('evaluation_strategy')
            or config.get('evaluation_strategy')
            or metadata.get('evaluation_strategy')
            or 'stratified_holdout'
        )
        sample_id_count = _recover_sample_id_count(
            _read_json_list(run_dir / 'split.json'),
            evaluation_strategy,
        )
        if sample_id_count is not None:
            recovered['sample_id_count'] = sample_id_count
            changed = True

    if changed:
        _atomic_json_write(run_dir / 'status.json', recovered)
    return recovered


def _mapping(value: object) -> dict[str, Any]:
    """把任意值安全转成 dict：仅 Mapping 被复制，其余返回 {}（投影层统一容错基元）。"""
    return dict(value) if isinstance(value, Mapping) else {}


def _read_json_mapping(path: Path) -> dict[str, Any]:
    """读取 JSON 文件为 dict；任何 IO/解析/编码错误都降级为 {}。"""
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _merged_file_mapping(
    path: Path,
    override: Mapping[str, Any] | None,
    *,
    read_file: bool = True,
) -> dict[str, Any]:
    """合并“磁盘文件 + 调用方覆盖值”。

    read_file=False 用于调用方已持有内容的场景（如写投影途中），避免读到
    自己尚未写完的旧文件造成字段回退；override 永远覆盖文件内容。
    """
    payload = _read_json_mapping(path) if read_file else {}
    payload.update(_mapping(override))
    return payload


def _first_present(*values: object) -> object | None:
    """返回第一个“有意义”的值（非 None 且非空串）；全空则 None。用于多来源字段择优。"""
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _finite_float(value: object) -> float | None:
    """宽松转 float；不可转或非有限（NaN/inf）返回 None，保证投影 JSON 合法且可比较。"""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None


def _canonical_evaluation_strategy(value: object) -> object:
    """把历史评估策略名归一到现行契约名。

    leave_one_repeat_index_cv / outer_leave_one_repeat_index_cv 是
    leave_one_sample_id_cv 的旧称（Repeat_index 字段已更名为 Sample_ID），
    对外只暴露新名字。
    """
    if str(value or "").strip().lower() in {
        "leave_one_repeat_index_cv",
        "outer_leave_one_repeat_index_cv",
    }:
        return "leave_one_sample_id_cv"
    return value


def _history_rows(run_dir: Path, status: Mapping[str, Any]) -> list[dict[str, Any]]:
    """优先用 status 内嵌的 history；否则回退读 history.csv（值为字符串，由调用方再数值化）。"""
    history = status.get("history")
    if isinstance(history, list):
        return [_mapping(row) for row in history if isinstance(row, Mapping)]
    try:
        with (run_dir / "history.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            return [_mapping(row) for row in DictReader(handle)]
    except OSError:
        return []


def _selected_search_rows(run_dir: Path) -> list[dict[str, Any]]:
    """从 hyperparameter_search.csv 筛出 is_selected 为真的行（每折最终选中的超参）。"""
    try:
        with (run_dir / "hyperparameter_search.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [_mapping(row) for row in DictReader(handle)]
    except OSError:
        return []
    return [row for row in rows if str(row.get("is_selected", "")).lower() in {"1", "true", "yes"}]


def _selection_entries(run_dir: Path, status: Mapping[str, Any]) -> list[dict[str, Any]]:
    """汇总每折的超参选择结果（best params + 选择指标 + valid balanced_accuracy）。

    数据来源按优先级：cv_metrics.json 的 folds → status.best_params_by_fold →
    split.json → hyperparameter_search.csv 的 selected 行。字段缺失时交叉回填
    （例如 selection_metric 为 balanced_accuracy 时可用 valid 指标补 score）。
    所有来源都没有有效内容时返回空列表，由调用方决定缺省展示。
    """
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
        selection_metric = str(item.get("selection_metric") or "balanced_accuracy")
        selection_score = _finite_float(item.get("selection_score"))
        split_metrics = _mapping(item.get("split_metrics"))
        valid_metrics = _mapping(split_metrics.get("valid"))
        # valid balanced_accuracy 三来源择优：顶层字段 → split_metrics.valid → 选择分数本身。
        valid_balanced_accuracy = _first_present(
            _finite_float(item.get("valid_balanced_accuracy")),
            _finite_float(valid_metrics.get("balanced_accuracy")),
            selection_score if selection_metric == "balanced_accuracy" else None,
        )
        if selection_score is None and selection_metric == "balanced_accuracy":
            selection_score = _finite_float(valid_balanced_accuracy)
        if params or selection_score is not None or valid_balanced_accuracy is not None:
            entries.append(
                {
                    "fold_index": item.get("fold_index", position),
                    "params": params,
                    "selection_metric": selection_metric,
                    "selection_score": selection_score,
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
        selection_metric = str(row.get("selection_metric") or "balanced_accuracy")
        selection_score = _finite_float(row.get("selection_score"))
        valid_balanced_accuracy = _finite_float(row.get("valid_balanced_accuracy"))
        if selection_score is None and selection_metric == "balanced_accuracy":
            selection_score = valid_balanced_accuracy
        if params or selection_score is not None or valid_balanced_accuracy is not None:
            entries.append(
                {
                    "fold_index": row.get("fold_index", position),
                    "params": params,
                    "selection_metric": selection_metric,
                    "selection_score": selection_score,
                    "valid_balanced_accuracy": valid_balanced_accuracy,
                }
            )
    return entries


def _importance_summary(run_dir: Path, status: Mapping[str, Any], name: str) -> dict[str, Any]:
    """读取重要性摘要：优先 status 内嵌，缺了再读同名 .json 产物文件。"""
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
    """汇总可解释性方法声明（declared/artifact）与重要性口径。

    declared_method 是训练配置声明的方法，artifact_method 是产物实际使用的方法
    （两者可能不同，例如 DSCARNet 使用 AggMap/PCA 双通路 2D Grad-CAM 回投 1D）。
    三者全空时返回 {}，表示该 Run 没有可解释性信息，前端不渲染对应区块。
    """
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
    )
    importance_metric = sample.get("importance_metric")
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
    read_status_file: bool = True,
) -> dict[str, Any]:
    """为已完成或历史 Run 生成只读、UI 就绪的训练审计摘要。

    刻意容忍写了一半的目录和 v2 之前的旧目录：所有读取都经 _merged_file_mapping
    降级，缺字段就置 None 而不报错。external_test_holdout 使用单数字段
    （best_params/selection_score），绝不向调用方暴露 CV/折相关措辞。

    参数：
        run_dir: Run 产物目录。
        config/status/model_metadata: 调用方已持有的覆盖值（优先于磁盘文件）。
        read_status_file: False 时不读 status.json——写投影途中避免读到旧文件。

    返回字段：model_type/model_family/architecture_version/evaluation_strategy/
    explainability，以及按模型族附加的 traditional（超参选择）或 deep_training
    （loss/epoch/lr 摘要）区块。
    """

    run_path = Path(run_dir)
    config_payload = _merged_file_mapping(run_path / "config.json", config)
    status_payload = _merged_file_mapping(
        run_path / "status.json",
        status,
        read_file=read_status_file,
    )
    metadata_payload = _merged_file_mapping(run_path / "model_metadata.json", model_metadata)

    evaluation_strategy = _canonical_evaluation_strategy(
        _first_present(status_payload.get("evaluation_strategy"), config_payload.get("evaluation_strategy"))
    )
    model_type = _first_present(
        status_payload.get("model_type"), metadata_payload.get("model_type"), config_payload.get("model_type")
    )
    model_family = _first_present(
        metadata_payload.get("model_family"), status_payload.get("model_family")
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
                traditional["selection_metric"] = selected["selection_metric"]
                traditional["selection_score"] = selected["selection_score"]
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
    # 旧目录没有 model_family 时，用 history 中是否存在 valid loss 推断“这是深度模型”。
    is_deep = model_family == "deep_learning" or (model_family is None and bool(deep_loss_values))
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

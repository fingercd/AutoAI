"""训练 Run 的创建、查询、取消、删除与 artifact 下载路由。

SQLite RunRecord 是规范状态；status.json 只补充旧前端字段。创建接口只写入 queued
Run，取消和删除均通过仓库状态机校验。成功 Run 的下载必须通过 manifest 条目，
不会把整个 storage/runs 目录暴露给通用文件接口。
"""

# ---------------------------------------------------------------------------
# 模块说明（教学注释）
#
# 本文件是训练 Run 生命周期管理的 HTTP 路由层，位于 FastAPI 后端的最外层：
# 前端（static/index.html 与 static/v2）通过 /api/training/runs 系列接口
# 创建、查询、取消、删除训练 Run，并下载训练产物（artifact）。
#
# 在系统中的位置与协作关系：
#   - contracts（TrainingRunRequest/TrainingSpec）：校验浏览器提交的训练配置；
#   - deps（get_run_repository / get_run_dir / resolve_training_data_reference）：
#     提供 Run 仓库、Run 目录和数据引用解析（server 模式强制 dataset_id）；
#   - runs.repository（RunRecord 状态机）：SQLite 中的 RunRecord 是唯一规范状态，
#     状态迁移（queued/running/succeeded/failed/cancelled）只能由仓库完成；
#   - runs.artifacts（RunArtifactWriter/Manifest）：成功 Run 的产物以 Manifest
#     登记大小和 SHA-256，下载与指标读取都走 Manifest 校验，不直接信任文件；
#   - runs.status_projection / result_projection：把规范状态投影成旧前端兼容的
#     status.json 字段（full 投影）或 run-result-v1 结果页契约；
#   - http.principal（Principal）：所有读取/取消/删除/下载都按 Principal
#     做 owner/tenant 作用域隔离，请求体绝不接受 owner_id/tenant_id。
#
# 关键设计约束（与 AGENTS.md 中的项目规则对应）：
#   - HTTP 创建接口只写入 queued Run，绝不在请求线程里直接启动训练；
#     真正的执行由独立 Worker 按 WORKER_CONTRACT_VERSION 契约消费队列。
#   - 任何返回给浏览器的 payload 都要剥掉服务器本地路径（data_path 等），
#     防止把主机目录结构泄露到前端或结果页。
#   - 列表页 summary 投影中的 test_macro_f1 只在“成功且 Manifest 完整”时读取；
#     交叉验证（CV）只取 pooled OOF 指标，绝不取 fold mean 冒充测试主指标。
# ---------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import json
import math
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from ..contracts import TrainingConfigValidationError, TrainingRunRequest, TrainingSpec
from ..datasets.repository import DatasetRepository
from ..http.principal import get_principal
from ..paths import DATASETS_DATABASE, STORAGE_DIR
from ..runs.artifacts import (
    ArtifactIntegrityError,
    ManifestCorruptError,
    RunArtifactWriter,
    discard_run_artifacts,
)
from ..runs.contracts import Principal, RunRecord, public_error_message
from ..runs.repository import InvalidRunTransition, RunNotFound
from ..runs.result_projection import project_run_result
from ..runs.status_projection import project_status, recover_status_from_artifacts
from ..version import WORKER_CONTRACT_VERSION
from .deps import get_run_dir, get_run_repository, resolve_training_data_reference

router = APIRouter()


def _discard_stopped_run_artifacts(run_id: str) -> None:
    """尽力清除 STOP Run 的半成品；失败时由下次查询/worker 启动重试。"""
    try:
        discard_run_artifacts(get_run_dir(run_id))
    except OSError:
        # Windows 上 worker 尚未完全退出时文件可能短暂被占用。记录状态已经是
        # cancelled，下载接口仍会拒绝访问；后续轮询会再次执行清理。
        pass


def _reconcile_interrupted_runs(repository: Any) -> list[str]:
    """把租约已失效的僵尸 running Run 收敛为 STOP，并清掉半成品。"""
    stopped = repository.stop_expired(now=datetime.now(timezone.utc))
    for run_id in stopped:
        _discard_stopped_run_artifacts(run_id)
    return stopped


def _read_status_payload(run_id: str) -> dict[str, Any]:
    """只读解析兼容 status.json；损坏或缺失时返回空对象。"""
    # status.json 只是“旧前端兼容快照”，不是规范状态，因此这里刻意宽容：
    # 文件缺失、JSON 损坏、编码错误、甚至内容不是对象时，都退化为空 dict，
    # 由规范侧的 RunRecord 兜底，而不是让整个接口失败。
    status_file = get_run_dir(run_id) / 'status.json'
    if not status_file.is_file():
        return {}
    try:
        loaded = json.loads(status_file.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _timestamp_value(value: Any) -> str | None:
    # 只接受“非空白字符串”作为时间戳；数字、None、空串一律视为无记录，
    # 避免把脏数据透传到前端的 ISO 时间字段。
    return str(value) if isinstance(value, str) and value.strip() else None


def _record_times(record: RunRecord, status: dict[str, Any] | None = None) -> tuple[str | None, str | None, str | None]:
    """优先使用数据库时间，并只从历史状态文件恢复确有记录的时间。"""
    # 规范来源是 RunRecord（SQLite）；status.json 只用于补齐升级前旧 Run
    # 缺失的时间字段。finished_at 额外兼容旧字段名 completed_at。
    payload = status if status is not None else _read_status_payload(record.run_id)
    created_at = record.created_at or _timestamp_value(payload.get('created_at'))
    started_at = record.started_at or _timestamp_value(payload.get('started_at'))
    finished_at = (
        record.finished_at
        or _timestamp_value(payload.get('finished_at'))
        or _timestamp_value(payload.get('completed_at'))
    )
    return created_at, started_at, finished_at


def _duration_seconds(started_at: str | None, finished_at: str | None) -> float | None:
    # 计算训练耗时（秒）。两个时间缺任一就无法计算，返回 None 而不是 0，
    # 让前端区分“未知”与“耗时 0 秒”。
    if not started_at or not finished_at:
        return None
    try:
        # 兼容以 'Z' 结尾的 UTC 写法；fromisoformat 在旧版本 Python 不接受 'Z'。
        started = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
        finished = datetime.fromisoformat(finished_at.replace('Z', '+00:00'))
        # 无时区的旧数据一律按 UTC 解释，避免 naive/aware 混减抛 TypeError。
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        if finished.tzinfo is None:
            finished = finished.replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    # max(0.0, ...)：容忍时钟回拨或脏数据造成的负耗时，不下溢成负数展示。
    return max(0.0, (finished - started).total_seconds())


def _without_server_paths(value: Any) -> Any:
    # 递归剥离任何可能泄露服务器文件系统布局的字段：显式的
    # data_path / test_data_path / run_dir / path，以及所有 *_path 命名的键。
    # 这是“浏览器永不见主机路径”安全约束的最后一道防线，作用于每个对外 payload。
    if isinstance(value, dict):
        return {
            key: _without_server_paths(item)
            for key, item in value.items()
            if key not in {'data_path', 'test_data_path', 'run_dir', 'path'}
            and not key.endswith('_path')
        }
    if isinstance(value, list):
        return [_without_server_paths(item) for item in value]
    return value


def _dataset_name_for_record(record: RunRecord) -> str | None:
    """优先读取 Run 快照，并为升级前的 Run 回查原始上传文件名。"""
    # 正常路径：创建 Run 时已把数据集名写入 config 快照，直接取用。
    configured = record.config.get('dataset_name')
    if configured:
        return str(configured)
    # 兼容路径 1：老 Run 没存名字但有 dataset_id，回查数据集仓库拿原始上传文件名。
    # 回查失败（库缺失/权限/IO 错误）不视为致命，继续往下退。
    if record.dataset_id:
        try:
            repository = DatasetRepository(DATASETS_DATABASE, storage_root=STORAGE_DIR)
            repository.initialize()
            return repository.resolve_system(record.dataset_id, legacy_path=None).original_name
        except (FileNotFoundError, PermissionError, OSError):
            pass
    # 兼容路径 2：更早的 Run 只有 legacy_data_path，取其文件名部分展示。
    if record.legacy_data_path:
        return Path(record.legacy_data_path).name
    return None


def _bounded_macro_f1(value: Any) -> float | None:
    """只接受 JSON 数值中的有限 Macro F1；bool 和字符串不做隐式转换。"""
    # bool 是 int 的子类，必须显式排除，否则 True 会被当成 1.0 的假指标；
    # 字符串数字也拒绝，防止脏 Manifest 把 "0.95" 混进指标展示。
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        metric = float(value)
    except OverflowError:
        return None
    # Macro F1 的合法域是 [0, 1]；NaN/Inf/越界值一律视为不可展示。
    return metric if math.isfinite(metric) and 0.0 <= metric <= 1.0 else None


def _read_manifest_verified_json(
    writer: RunArtifactWriter,
    manifest: dict[str, Any],
    name: str,
) -> dict[str, Any] | None:
    """读取一个由 Manifest 同时登记了大小和 SHA-256 的 JSON 对象。"""
    # 读取任何指标文件前都先过完整性校验，流程分四步：
    # 1) Manifest 条目本身必须合法（size 为非负 int、sha256 为 64 位十六进制）；
    # 2) 解析出的路径必须正好落在 run_dir 内，防止 '../' 之类的目录逃逸；
    # 3) 实际字节数与登记大小一致；4) 实际 SHA-256 与登记值一致。
    # 任何一步失败都返回 None（视为“不可信/不存在”），而不是读半个坏文件。
    try:
        entry = manifest['artifacts'].get(name)
        if not isinstance(entry, dict):
            return None
        expected_size = entry.get('size_bytes')
        expected_sha256 = entry.get('sha256')
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in '0123456789abcdefABCDEF' for character in expected_sha256)
        ):
            return None
        # resolve() 后比较父目录，等价于“禁止符号链接/相对路径逃出 run 目录”。
        path = (writer.run_dir / name).resolve()
        if path.parent != writer.run_dir or not path.is_file():
            return None
        encoded = path.read_bytes()
        if len(encoded) != expected_size:
            return None
        if hashlib.sha256(encoded).hexdigest() != expected_sha256.lower():
            return None
        # utf-8-sig 兼容带 BOM 的 JSON 输出（部分 Windows 工具会写入 BOM）。
        loaded = json.loads(encoded.decode('utf-8-sig'))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _summary_test_macro_f1(
    record: RunRecord,
    writer: RunArtifactWriter,
    manifest: dict[str, Any],
) -> float | None:
    """读取列表页唯一需要的测试主指标，不构造完整 run-result-v1。"""
    # 评估口径分两大类，各自有多种历史别名：
    #   - cv：交叉验证（leave_one_sample_id_cv 及其历史命名）；
    #   - direct：直接划分（stratified_holdout / external_test_holdout 等）。
    cv_aliases = {
        'leave_one_sample_id_cv',
        'leave_one_repeat_index_cv',
        'outer_leave_one_repeat_index_cv',
        'loocv',
        'loo',
    }
    direct_aliases = {
        'stratified_holdout',
        'stratified',
        'holdout',
        'external_test_holdout',
    }

    def evaluation_mode(value: Any) -> str | None:
        # 把任意来源的口径字符串归一化成 'cv' / 'direct' / None（未知）。
        normalized = str(value or '').strip().lower()
        if normalized in cv_aliases:
            return 'cv'
        if normalized in direct_aliases:
            return 'direct'
        return None

    metadata = manifest.get('metadata')
    metadata = metadata if isinstance(metadata, dict) else {}
    # 口径可以出现在三个位置：DB 里的 config.evaluation_strategy、
    # config.split_mode、Manifest metadata.evaluation_strategy。全部收集后
    # 取“声明过的不同口径集合”。
    declared_modes = {
        mode
        for mode in (
            evaluation_mode(record.config.get('evaluation_strategy')),
            evaluation_mode(record.config.get('split_mode')),
            evaluation_mode(metadata.get('evaluation_strategy')),
        )
        if mode is not None
    }
    # 数据库配置与已发布 Manifest 对评估口径有冲突时，宁可不展示也不猜测。
    if len(declared_modes) > 1:
        return None
    mode = next(iter(declared_modes), None)
    cv_metrics: dict[str, Any] | None = None
    if mode is None:
        # 只为没有口径快照的旧 Run 读取已校验的 cv_metrics.strategy；未知仍返回空值。
        cv_metrics = _read_manifest_verified_json(writer, manifest, 'cv_metrics.json')
        mode = evaluation_mode(cv_metrics.get('strategy')) if cv_metrics is not None else None

    if mode == 'cv':
        if cv_metrics is None:
            cv_metrics = _read_manifest_verified_json(writer, manifest, 'cv_metrics.json')
        if cv_metrics is None:
            return None
        # 文件内自报的 strategy 若与 cv 矛盾（例如写成 holdout），视为不可信。
        artifact_mode = evaluation_mode(cv_metrics.get('strategy'))
        if artifact_mode is not None and artifact_mode != 'cv':
            return None
        # CV 的主指标必须取自 cv_summary.pooled_test（所有折 OOF 预测合并
        # 计算的 pooled 指标），这是项目契约硬性规定。
        cv_summary = cv_metrics.get('cv_summary')
        pooled_test = cv_summary.get('pooled_test') if isinstance(cv_summary, dict) else None
        if isinstance(pooled_test, dict) and 'macro_f1' in pooled_test:
            return _bounded_macro_f1(pooled_test.get('macro_f1'))

        # 兼容少量没有 cv_summary 的旧 Run：只有明确标注为 pooled OOF 的
        # test 节点才可作为回退，绝不读取 fold_mean 或不明顶层指标。
        metrics = _read_manifest_verified_json(writer, manifest, 'metrics.json')
        test_metrics = metrics.get('test') if isinstance(metrics, dict) else None
        if isinstance(test_metrics, dict) and test_metrics.get('aggregation') in {
            'pooled_out_of_fold',
            'pooled_oof',
        }:
            return _bounded_macro_f1(test_metrics.get('macro_f1'))
        return None

    # 既不是 cv 也不是 direct 的未知口径，不给列表页提供指标。
    if mode != 'direct':
        return None
    # direct 口径直接读 metrics.json 里的 test.macro_f1。
    metrics = _read_manifest_verified_json(writer, manifest, 'metrics.json')
    test_metrics = metrics.get('test') if isinstance(metrics, dict) else None
    if not isinstance(test_metrics, dict):
        return None
    return _bounded_macro_f1(test_metrics.get('macro_f1'))


def _projection(record: RunRecord) -> dict[str, Any]:
    """合并规范 RunRecord、兼容 status.json 和可恢复的 artifact 摘要。"""
    # full 投影的装配顺序：
    #   1) 以 status.json 兼容快照打底（提供旧前端认识的字段）；
    #   2) 用 artifact 目录里可恢复的信息补充（例如训练中途 Web 重启过）；
    #   3) 再用规范 RunRecord 覆盖关键字段——DB 永远赢过快照。
    status_file = get_run_dir(record.run_id) / 'status.json'
    payload = _read_status_payload(record.run_id)
    payload = recover_status_from_artifacts(status_file.parent, payload)
    dataset_name = _dataset_name_for_record(record)
    created_at, started_at, finished_at = _record_times(record, payload)
    payload.update(
        {
            'run_id': record.run_id,
            'status': record.legacy_status,
            'state': record.state,
            'version': record.version,
            'dataset_id': record.dataset_id,
            'dataset_name': dataset_name,
            'config': record.config,
            'created_at': created_at,
            'started_at': started_at,
            'completed_at': finished_at,
            **record.progress,
        }
    )
    # 错误信息对外一律经过 public_error_message 脱敏（去掉堆栈/内部路径），
    # error_details 里的 message 同样替换，避免泄露实现细节。
    if record.error:
        payload['error'] = public_error_message(record.error)
        if record.error_details:
            payload['error_details'] = {
                **record.error_details,
                'message': public_error_message(record.error_details.get('message') or record.error),
            }
    else:
        payload.pop('error', None)
    if record.manifest_name:
        payload['manifest_name'] = record.manifest_name
    # 出口前再过一次 artifact 恢复并剥离服务器路径，作为对外安全边界。
    return _without_server_paths(recover_status_from_artifacts(status_file.parent, payload))


def _dataset_snapshot_for_reference(
    *,
    dataset_id: str | None,
    legacy_path: str | None,
    dataset_name: str | None,
) -> dict[str, Any]:
    # 创建 Run 时对数据集做一次“指纹快照”：解析引用并重新计算 SHA-256。
    # 这样即使之后数据集文件被替换，也能在结果页/审计时发现训练时使用的
    # 数据与当前数据不一致。关键字-only 参数（*）强制调用方写清语义。
    repository = DatasetRepository(DATASETS_DATABASE, storage_root=STORAGE_DIR)
    repository.initialize()
    record = repository.resolve_system(dataset_id, legacy_path=legacy_path)
    actual_sha256 = repository.verify_integrity(record)
    return {
        'dataset_id': dataset_id,
        'name': dataset_name or record.original_name,
        'sha256': actual_sha256,
    }


def _summary_projection(record: RunRecord) -> dict[str, Any]:
    # 训练记录列表页的轻量投影：不构造完整 run-result-v1，只拼出表格所需字段。
    snapshot = record.dataset_snapshot
    created_at, started_at, finished_at = _record_times(record)
    # result_state 描述“产物可用性”，与 Run 自身的生命周期 state 正交：
    # 非终态直接映射为 pending/running/failed/cancelled；终态（succeeded）
    # 则要进一步检查 Manifest 与 artifact 完整性。
    result_state = {
        'queued': 'pending',
        'running': 'running',
        'failed': 'failed',
        'cancelled': 'cancelled',
    }.get(record.state)
    test_macro_f1: float | None = None
    if result_state is None:
        # 只有 succeeded 会走到这里：逐个 descriptor 校验 artifact 完整性。
        run_dir = get_run_dir(record.run_id)
        artifact_writer = RunArtifactWriter(run_dir)
        try:
            manifest, descriptors = artifact_writer.descriptors(run_id=record.run_id)
        except FileNotFoundError:
            result_state = 'missing_manifest'
        except ManifestCorruptError:
            result_state = 'corrupt_manifest'
        else:
            # 任一“必需或适用”的 artifact 缺失/损坏都降级为 partial，
            # 提示结果可能不完整，但不隐瞒 Run 本身已成功。
            damaged = any(
                item['integrity'] in {'missing', 'corrupt'}
                and (item['required'] or item['applicable'])
                for item in descriptors
            )
            result_state = 'partial' if damaged else 'ready'
            # 只有完全 ready 的成功 Run 才读取测试主指标，
            # 避免把残缺结果的数字展示给列表页。
            if record.state == 'succeeded' and result_state == 'ready':
                test_macro_f1 = _summary_test_macro_f1(record, artifact_writer, manifest)
    return {
        'run_id': record.run_id,
        'state': record.state,
        'status': record.legacy_status,
        'result_state': result_state,
        'model_type': record.config.get('model_type'),
        'dataset_id': record.dataset_id,
        'dataset_name': snapshot.get('name') or _dataset_name_for_record(record),
        'created_at': created_at,
        'started_at': started_at,
        'finished_at': finished_at,
        'duration_seconds': _duration_seconds(started_at, finished_at),
        'test_macro_f1': test_macro_f1,
        'progress': _without_server_paths(record.progress),
        'error': public_error_message(record.error) if record.error else None,
    }


def _assert_worker_contract_compatible(repository: Any) -> None:
    """已发现活跃旧 Worker 时拒绝创建 Run，避免它抢占并生成旧格式产物。"""
    # Web 与训练 Worker 通过队列解耦，二者必须跑同一个 WORKER_CONTRACT_VERSION。
    # 如果心跳显示有活跃 Worker 但契约版本不一致，新入队的 Run 会被旧 Worker
    # 抢走并产出旧格式结果，因此这里直接 503 拒绝创建，宁缺毋滥。
    health = repository.worker_health(
        now=datetime.now(timezone.utc),
        expected_contract_version=WORKER_CONTRACT_VERSION,
    )
    if health.get('available') and not health.get('compatible'):
        raise HTTPException(
            status_code=503,
            detail={
                'code': 'worker_contract_mismatch',
                'message': '训练 Worker 版本与当前 Web 不兼容，请同时重启 Web 和 Worker',
                'expected_contract_version': WORKER_CONTRACT_VERSION,
                'actual_contract_version': health.get('contract_version'),
            },
        )


@router.post('/api/training/runs', status_code=202)
@router.post('/api/train', status_code=202)
def create_run(payload: TrainingRunRequest, principal: Principal = Depends(get_principal)) -> dict[str, object]:
    """持久化 queued Run；本请求绝不直接调用训练器。"""
    # 流程总览：解析数据引用 → 校验训练配置 → 检查 Worker 契约 → 组装 config
    # → 数据集指纹快照 → 写入 queued Run → 生成兼容 status.json → 返回 202。
    # 注意 202（Accepted）语义：请求被接受并入队，不代表训练已开始。
    # resolve_training_data_reference 内部强制 server 模式只接受 dataset_id，
    # 拒绝浏览器传入的任意 data_path（防目录穿越/任意文件读）。
    data_ref = resolve_training_data_reference(payload, principal=principal)
    try:
        # TrainingSpec.validated 会按是否携带独立测试集校验评估口径等约束
        # （例如 external_test_holdout 必须有独立测试集）。
        spec = TrainingSpec.from_legacy(payload.config).validated(
            has_external_test=bool(data_ref.test_dataset_id or data_ref.test_legacy_path)
        )
    except TrainingConfigValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={'code': 'invalid_training_config', 'message': str(exc)},
        ) from exc
    repository = get_run_repository()
    _assert_worker_contract_compatible(repository)
    # config 落库时附带数据集名/测试集引用等快照信息，保证历史 Run 在数据集
    # 被清理后仍能展示“当时用的是什么数据”。
    config = spec.to_legacy_dict()
    config['dataset_name'] = data_ref.dataset_name
    if data_ref.test_dataset_id:
        config['test_dataset_id'] = data_ref.test_dataset_id
    if data_ref.test_legacy_path:
        config['test_data_path'] = data_ref.test_legacy_path
    if data_ref.test_dataset_name:
        config['test_dataset_name'] = data_ref.test_dataset_name
    dataset_snapshot = _dataset_snapshot_for_reference(
        dataset_id=data_ref.dataset_id,
        legacy_path=data_ref.legacy_path,
        dataset_name=data_ref.dataset_name,
    )
    if data_ref.test_dataset_id or data_ref.test_legacy_path:
        # 独立测试集同样记录 SHA-256 指纹，防止测试集被替换后结果被误读。
        test_snapshot = _dataset_snapshot_for_reference(
            dataset_id=data_ref.test_dataset_id,
            legacy_path=data_ref.test_legacy_path,
            dataset_name=data_ref.test_dataset_name,
        )
        config['test_dataset_sha256'] = test_snapshot['sha256']
    # create_queued 只做状态机允许的“创建 queued”迁移；身份由服务端
    # Principal 注入，请求体里的 owner_id/tenant_id 不会被采信。
    record = repository.create_queued(
        dataset_id=data_ref.dataset_id,
        legacy_data_path=data_ref.legacy_path,
        config=config,
        dataset_snapshot=dataset_snapshot,
        principal=principal,
    )
    # 同步写一份兼容 status.json，让旧前端在 Worker 尚未拾取时也能看到进度壳。
    project_status(get_run_dir(record.run_id), record, config=config)
    return {
        'run_id': record.run_id,
        'status': record.legacy_status,
        'state': record.state,
        'warnings': list(spec.warnings),
    }


@router.get('/api/training/runs')
def get_runs(
    projection: str | None = Query(default=None),
    limit: int | None = Query(default=None, ge=1, le=100),
    cursor: str | None = Query(default=None),
    principal: Principal = Depends(get_principal),
) -> Any:
    """默认保持旧列表；projection=summary 返回轻量、可游标分页的结果。"""
    # 两种投影并存是为了兼容：full（默认）保持旧前端 list 形状；summary 是
    # 训练记录页的轻量分页形态，只带列表所需字段和可空 test_macro_f1。
    if projection not in {None, 'full', 'summary'}:
        raise HTTPException(status_code=422, detail='projection 仅支持 full 或 summary')
    repository = get_run_repository()
    _reconcile_interrupted_runs(repository)
    try:
        # list_scoped 按 Principal 过滤 owner/tenant：server 模式下历史
        # owner 为空的 Run 对普通用户不可见。
        # summary 分页用“多取 1 条”的经典技巧判断是否还有下一页（has_more）。
        records = repository.list_scoped(
            principal=principal,
            limit=(limit or 20) + 1 if projection == 'summary' else limit,
            cursor=cursor,
        )
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail='cursor 对应的 run 不存在') from exc
    for record in records:
        if record.state == 'cancelled':
            _discard_stopped_run_artifacts(record.run_id)
    if projection == 'summary':
        page_limit = limit or 20
        has_more = len(records) > page_limit
        page = records[:page_limit]
        # 游标取本页最后一条的 run_id；无更多数据时显式返回 None，
        # 让前端可以停用“加载更多”。
        return {
            'items': [_summary_projection(record) for record in page],
            'next_cursor': page[-1].run_id if has_more and page else None,
        }
    return [_projection(record) for record in records]


@router.get('/api/training/runs/{run_id}')
def get_run(run_id: str, principal: Principal = Depends(get_principal)) -> dict[str, Any]:
    """返回单个 Run 的规范状态与兼容训练结果投影。"""
    # get_scoped 同时完成“存在性 + 归属”检查：不属于当前 Principal 的 Run
    # 与不存在一样返回 404，避免通过响应差异探测他人 run_id。
    repository = get_run_repository()
    _reconcile_interrupted_runs(repository)
    try:
        record = repository.get_scoped(run_id, principal=principal)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail='run 不存在') from exc
    if record.state == 'cancelled':
        _discard_stopped_run_artifacts(record.run_id)
    return _projection(record)


@router.get('/api/training/runs/{run_id}/result')
def get_run_result(run_id: str, principal: Principal = Depends(get_principal)) -> dict[str, Any]:
    """返回版本化、状态感知且可刷新恢复的建模结果页契约。"""
    # 结果页契约 run-result-v1（见 docs/run_result_contract.md）：queued/running
    # 返回进度壳，succeeded 返回完整指标与可解释性产物索引，failed 返回脱敏错误；
    # 前端刷新专属 URL（#/results?run_id=...）后靠它恢复页面状态。
    repository = get_run_repository()
    _reconcile_interrupted_runs(repository)
    try:
        record = repository.get_scoped(run_id, principal=principal)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail='run 不存在') from exc
    if record.state == 'cancelled':
        _discard_stopped_run_artifacts(record.run_id)
    return project_run_result(
        record,
        run_dir=get_run_dir(run_id),
        dataset_name=_dataset_name_for_record(record),
    )


@router.post('/api/training/runs/{run_id}/stop')
def stop_run(run_id: str, principal: Principal = Depends(get_principal)) -> dict[str, Any]:
    """事务性停止 queued/running Run，只保留 SQLite 中的 STOP 记录。"""
    repository = get_run_repository()
    try:
        record = repository.cancel_scoped(
            run_id,
            now=datetime.now(timezone.utc),
            principal=principal,
            reason='user_requested',
            message='用户已停止训练',
        )
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail='run 不存在') from exc
    except InvalidRunTransition as exc:
        raise HTTPException(status_code=409, detail='run 当前状态不能停止') from exc
    _discard_stopped_run_artifacts(run_id)
    # 不再生成 status.json：STOP Run 不保留本次训练的任何磁盘结果。
    return _projection(record)


@router.post('/api/training/runs/{run_id}/cancel')
def cancel_run(run_id: str, principal: Principal = Depends(get_principal)) -> dict[str, Any]:
    """旧取消接口兼容别名；新前端统一使用 /stop。"""
    return stop_run(run_id, principal)


@router.delete('/api/training/runs/{run_id}')
def delete_run(run_id: str, principal: Principal = Depends(get_principal)) -> dict[str, object]:
    """只删除终态 Run；目录先原子移入隔离区，DB 失败时可恢复。"""
    repository = get_run_repository()
    _reconcile_interrupted_runs(repository)
    try:
        record = repository.get_scoped(run_id, principal=principal)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail='run 不存在') from exc
    # 只允许删除终态 Run；queued/running 必须先取消，否则可能删掉 Worker
    # 正在写入的目录，留下半拉子产物与状态不一致。
    if record.state in {'queued', 'running'}:
        raise HTTPException(status_code=409, detail='run 当前状态不能删除')

    # 删除采用“两阶段 + 隔离区”设计，保证任何一步失败都可回滚：
    #   阶段 1：把 run 目录原子 rename 到同级隔离目录（不真正删数据）；
    #   阶段 2：删除 DB 记录；成功后才真正 rmtree 隔离目录。
    # DB 删除失败时把隔离目录 rename 回原路径，做到“记录与产物同在或同不在”。
    run_dir = get_run_dir(run_id)
    quarantine_dir: Path | None = None
    try:
        if run_dir.exists():
            # 先验证清理能力，避免目录已移走后才发现清理函数不可用。
            probe = run_dir.parent / f'.delete-probe-{uuid.uuid4().hex}'
            probe.mkdir(parents=True, exist_ok=False)
            try:
                shutil.rmtree(probe)
            finally:
                if probe.exists():
                    probe.rmdir()
            # replace() 在同分区内是原子 rename，Worker 或其他请求不会看到
            # 半个被删除的目录。
            quarantine_dir = run_dir.parent / f'.{run_id}.deleting-{uuid.uuid4().hex}'
            run_dir.replace(quarantine_dir)
    except OSError as exc:
        raise HTTPException(status_code=500, detail='run 产物删除失败，训练记录已保留') from exc
    try:
        repository.delete_terminal_scoped(run_id, principal=principal)
    except RunNotFound as exc:
        # 三个 except 分支做同一件事：把隔离目录搬回原位再报错，即回滚阶段 1。
        if quarantine_dir is not None and quarantine_dir.exists() and not run_dir.exists():
            quarantine_dir.replace(run_dir)
        raise HTTPException(status_code=404, detail='run 不存在') from exc
    except InvalidRunTransition as exc:
        # 并发下 Run 可能刚被 Worker 改回非终态，此时拒绝删除并恢复目录。
        if quarantine_dir is not None and quarantine_dir.exists() and not run_dir.exists():
            quarantine_dir.replace(run_dir)
        raise HTTPException(status_code=409, detail='run 当前状态不能删除') from exc
    except Exception as exc:
        if quarantine_dir is not None and quarantine_dir.exists() and not run_dir.exists():
            quarantine_dir.replace(run_dir)
        raise HTTPException(status_code=500, detail='run 记录删除失败，产物目录已恢复') from exc
    if quarantine_dir is not None and quarantine_dir.exists():
        try:
            shutil.rmtree(quarantine_dir)
        except OSError as exc:
            # 最隐蔽的失败：DB 记录已删但目录删不掉。此时用 import_legacy 把
            # 原记录完整写回 DB（连同 owner/tenant 与时间戳），再把目录搬回，
            # 使整个删除操作对外表现为“从未发生”。
            repository.import_legacy(
                run_id=record.run_id,
                state=record.state,
                config=record.config,
                dataset_id=record.dataset_id,
                legacy_data_path=record.legacy_data_path,
                principal=Principal(owner_id=record.owner_id, tenant_id=record.tenant_id),
                dataset_snapshot=record.dataset_snapshot,
                created_at=record.created_at,
                started_at=record.started_at,
                finished_at=record.finished_at,
                progress=record.progress,
                manifest_name=record.manifest_name,
                error=record.error,
                error_details=record.error_details,
            )
            if quarantine_dir.exists() and not run_dir.exists():
                quarantine_dir.replace(run_dir)
            raise HTTPException(status_code=500, detail='run 清理失败，训练记录和剩余产物已恢复') from exc
    return {'run_id': run_id, 'deleted': True}


@router.get('/api/training/runs/{run_id}/artifact/{name}')
def get_run_artifact(
    run_id: str,
    name: str,
    principal: Principal = Depends(get_principal),
) -> FileResponse:
    """仅返回 Manifest 中存在且标为 downloadable 的单个产物。"""
    # 下载白名单由 Manifest 驱动：model.pkl / model.pt / joblib / status.json
    # 等不在 downloadable 之列（见 AGENTS.md 的 artifact 规则），resolve_download
    # 内部会对不在白名单或完整性不符的名字抛错。
    run_dir = get_run_dir(run_id)
    try:
        record = get_run_repository().get_scoped(run_id, principal=principal)
    except RunNotFound as exc:
        # 早期直接训练产物可能没有 SQLite 记录，仍保持只读兼容。
        # 正式删除会同时移除 Run 目录，因此不会被此兼容路径恢复访问。
        # 兼容分支只对本机 local 模式（owner/tenant 均为空）开放；server 模式
        # 下任何带身份的请求查不到记录都一律 404。
        if principal.owner_id is not None or principal.tenant_id is not None or not run_dir.is_dir():
            raise HTTPException(status_code=404, detail='run 不存在') from exc
    else:
        # 未成功完成的 Run 不开放下载：失败/进行中的目录里可能有半成品文件，
        # 409 而不是 404，提示“存在但不可用”。
        if record.state != 'succeeded':
            raise HTTPException(status_code=409, detail='run 尚未成功完成，产物不可下载')
    try:
        path = RunArtifactWriter(run_dir).resolve_download(name)
    except PermissionError as exc:
        # 名字合法但不在白名单（例如 model.pt）。
        raise HTTPException(status_code=403, detail='不允许下载该文件') from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail='文件不存在') from exc
    except (ManifestCorruptError, ArtifactIntegrityError) as exc:
        # Manifest 或文件校验和异常：不冒险发不可信字节，用 409 表示冲突状态。
        raise HTTPException(status_code=409, detail='文件完整性校验失败') from exc
    except ValueError as exc:
        # 非法文件名（含路径分隔符、'..' 等）在最早一层挡掉。
        raise HTTPException(status_code=400, detail='文件名无效') from exc
    return FileResponse(path, filename=Path(name).name)

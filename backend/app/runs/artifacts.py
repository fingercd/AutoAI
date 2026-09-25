"""Run artifact 的原子写入、Manifest 发布和安全下载解析。

所有文件先写同目录临时文件、fsync 后再 os.replace。Run 只有在 Manifest 完成后
才能成功；下载时重新检查 Manifest、downloadable 标记、路径边界和文件存在性。
joblib 映射对象默认私有，避免把内部拟合对象当成公共 API。

教学级模块说明：
- 系统位置：这是 Run（一次训练任务）产物层的核心模块，位于 backend/app/runs/ 下。
  训练 worker 通过 RunArtifactWriter 把 metrics.json、predictions.csv 等结果文件写入
  每个 Run 自己的目录（storage/runs/<run_id>/），训练结束后用 finalize() 生成
  manifest.json；HTTP 结果接口（/api/training/runs/{run_id}/result 与 artifact
  下载接口）再通过 load_manifest / descriptors / resolve_download 读取并校验产物。
- 协作模块：Manifest 契约版本来自 ..version（ARTIFACT_MANIFEST_CONTRACT_VERSION）；
  Run 的状态机由 repository.py 的 SQLite 仓库管理，本模块只负责文件层，
  不参与任何状态决策（status.json 只是数据库终态提交后的兼容投影）。
- 关键设计约束：
  1) 原子写入：所有文件先写同目录 .tmp 临时文件并 fsync，再 os.replace 覆盖目标名；
     同目录 rename 在同一文件系统内是原子操作，进程崩溃也不会留下半截文件。
  2) Manifest 信任边界：结果页与下载都以 manifest.json 中的显式 catalog 及
     sha256/大小校验为准；Manifest 缺失或损坏时 Run 不能视为成功，只能给出
     不可下载的描述符（见 unavailable_descriptors）。
  3) 下载白名单：ARTIFACT_CATALOG 是唯一公开下载面；模型对象（model.pkl/model.pt）、
     joblib 拟合对象、含服务器路径字段的 config.json 一律不可下载，避免泄露内部
     路径，也避免把可反序列化的对象暴露成公共 API。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..version import ARTIFACT_MANIFEST_CONTRACT_VERSION


# Manifest 的 schema 版本直接绑定全局契约版本：读写两侧用同一常量判断一份 Manifest
# 是否属于“新版显式 catalog”（v2），从而在 resolve_download / descriptors 中区分新旧兼容路径。
MANIFEST_SCHEMA_VERSION = ARTIFACT_MANIFEST_CONTRACT_VERSION
# 历史 Run 曾把“全局” feature_importance 当作公开下载；当前正式前端只展示单样品解释结果。
# 这两个旧文件名仅保留“已知名字的窄范围直接下载”兼容，不再进入结果页 catalog。
LEGACY_DIRECT_DOWNLOAD_NAMES = frozenset({
    'feature_importance.json',
    'feature_importance.csv',
})

# 新 Run 的公开下载面由这里唯一声明。未知文件和可执行模型对象默认私有；旧
# manifest 的 downloadable 标记仍由 resolve_download 兼容读取。历史全局
# 重要性文件不再出现在结果页 catalog，但保留已知文件名的直接下载兼容。
ARTIFACT_CATALOG: dict[str, dict[str, object]] = {
    'metrics.json': {'label': '总体指标', 'category': 'metrics', 'required': True, 'downloadable': True},
    'cv_metrics.json': {'label': '交叉验证汇总', 'category': 'metrics', 'required': True, 'downloadable': True},
    'fold_metrics.csv': {'label': '分折指标', 'category': 'metrics', 'required': True, 'downloadable': True},
    'predictions.csv': {'label': '预测明细', 'category': 'predictions', 'required': True, 'downloadable': True},
    'cv_predictions.csv': {'label': '交叉验证预测明细', 'category': 'predictions', 'required': False, 'downloadable': True},
    'history.csv': {'label': '训练过程', 'category': 'training', 'required': False, 'downloadable': True},
    'hyperparameter_search.csv': {'label': '参数搜索记录', 'category': 'training', 'required': False, 'downloadable': True},
    'search_plan.json': {'label': '冻结搜索计划', 'category': 'training', 'required': False, 'downloadable': True},
    'search_trials.json': {'label': '逐次搜索记录', 'category': 'training', 'required': False, 'downloadable': True},
    'search_trials.csv': {'label': '逐次搜索表', 'category': 'training', 'required': False, 'downloadable': True},
    'search_summary.json': {'label': '搜索汇总', 'category': 'training', 'required': False, 'downloadable': True},
    'search_timeline.json': {'label': '阶段时间线', 'category': 'training', 'required': False, 'downloadable': True},
    'sample_feature_importance.json': {'label': '单样品解释结果', 'category': 'explainability', 'required': False, 'downloadable': True},
    'sample_feature_importance.csv': {'label': '单样品解释结果', 'category': 'explainability', 'required': False, 'downloadable': True},
    'dscarnet_mapping.json': {'label': 'DSCARNet 映射说明', 'category': 'explainability', 'required': False, 'downloadable': True},
    'model_metadata.json': {'label': '模型元数据', 'category': 'metadata', 'required': True, 'downloadable': True},
    'label_map.json': {'label': '类别映射', 'category': 'metadata', 'required': True, 'downloadable': True},
    'split.json': {'label': '数据划分', 'category': 'metadata', 'required': True, 'downloadable': True},
    'config.json': {'label': '训练配置', 'category': 'metadata', 'required': True, 'downloadable': True},
    # status.json 会在 DB 终态提交后刷新，是兼容投影而非不可变结果文件。
    'status.json': {'label': '兼容状态投影', 'category': 'internal', 'required': False, 'downloadable': False, 'volatile': True},
    'model.pkl': {'label': '模型对象', 'category': 'model', 'required': False, 'downloadable': False},
    'model.pt': {'label': '模型权重', 'category': 'model', 'required': False, 'downloadable': False},
}


# 两种 Manifest 相关异常，语义故意分开：
# - ManifestCorruptError（ValueError）：Manifest 本身“读不懂”（非法 JSON、结构缺失、
#   run_id 与目录不一致、条目字段类型非法）。属于数据损坏/被篡改，上层据此判定
#   结果不可用，而不是当成“文件不存在”。
# - ArtifactIntegrityError（OSError）：Manifest 读得懂，但磁盘上的 artifact 实际大小
#   或 sha256 与登记值不一致。归为 OSError 是为了和真实 I/O 故障走同一处理路径。
class ManifestCorruptError(ValueError):
    """Manifest 不是可解释的对象或 artifact 索引损坏。"""


class ArtifactIntegrityError(OSError):
    """artifact 的实际大小或内容哈希与 Manifest 不一致。"""


# 防目录穿越工具：判断 target 解析后的绝对路径是否仍位于 root 之内。
# 下载 artifact 时以 run_dir 为 root 调用，借道 '../' 构造的逃逸路径会在这里被拦下。
# 用 relative_to 的 ValueError 做判断，是 pathlib 惯用的“包含关系”判定写法。


def _is_within(target: Path, root: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False


def discard_run_artifacts(run_dir: str | Path) -> bool:
    """原子隔离并删除一个未成功 Run 的全部磁盘产物。

    Run 的生命周期记录保存在 SQLite，不依赖 ``status.json``。停止训练时可以
    安全移除整个 Run 目录，只留下数据库中的 STOP 记录。先在同级目录原子改名，
    避免结果接口在清理过程中看到半套文件；目录已不存在时按幂等成功处理。
    """
    path = Path(run_dir)
    for stale in path.parent.glob(f'.{path.name}.discarding-*'):
        if stale.is_dir():
            shutil.rmtree(stale)
    if not path.exists():
        return False
    quarantine = path.parent / f'.{path.name}.discarding-{uuid.uuid4().hex}'
    try:
        path.replace(quarantine)
    except FileNotFoundError:
        return False
    shutil.rmtree(quarantine)
    return True


# 分块（1 MiB）流式计算文件的 sha256，避免把大 artifact 一次性读入内存。
# Manifest 登记的就是这个值；下载前会重新计算并比对，作为内容完整性校验。
def _sha256(path: Path, active=lambda: None) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            active()
            digest.update(chunk)
    return digest.hexdigest()


# 用文件名后缀推断 format 字段（如 'json'、'csv'）；无后缀时归为 'binary'。
# Manifest 的 format/media_type 只是给前端展示和下载响应头用的元数据，不影响校验逻辑。
def _format_for_name(name: str) -> str:
    suffix = Path(name).suffix.lower().lstrip('.')
    return suffix or 'binary'


# 后缀 → HTTP Content-Type 的映射；未知类型一律 application/octet-stream，
# 让浏览器按二进制下载处理，避免未知文件被当作可渲染内容打开。
def _media_type(name: str) -> str:
    return {
        'json': 'application/json',
        'csv': 'text/csv',
        'txt': 'text/plain',
    }.get(_format_for_name(name), 'application/octet-stream')


# 单个文件名的 catalog 策略查询，是“默认私有、显式公开”安全模型的核心：
# - 命中 ARTIFACT_CATALOG：返回声明的公开策略（注意返回副本 dict(...)，防止调用方
#   意外改到全局常量）。
# - *.joblib：DSCARNet 等模型的内部拟合对象，强制 internal 且不可下载——joblib
#   反序列化可执行任意代码，绝不能成为公共下载面。
# - 其余未知文件名：一律 internal/不可下载。新文件要公开必须先登记进 CATALOG。
def _catalog_policy(name: str) -> dict[str, object]:
    if name in ARTIFACT_CATALOG:
        return dict(ARTIFACT_CATALOG[name])
    if name.endswith('.joblib'):
        return {
            'label': '内部拟合对象',
            'category': 'internal',
            'required': False,
            'downloadable': False,
        }
    return {
        'label': name,
        'category': 'internal',
        'required': False,
        'downloadable': False,
    }


# 判断 CSV 是否至少有一行数据（表头之外）。
# 用于 history.csv / hyperparameter_search.csv 的“适用性”判断：文件存在但为空壳
# （只有表头）时，结果页应显示“不适用”而不是提供空下载。
# 读不出或解码失败一律按“无数据”处理，宁可不展示也不误报。
def _csv_has_data(path: Path) -> bool:
    try:
        with path.open('r', encoding='utf-8-sig') as handle:
            header_seen = False
            for line in handle:
                if not line.strip():
                    continue
                if not header_seen:
                    header_seen = True
                    continue
                return True
    except (OSError, UnicodeDecodeError):
        return False
    return False


# 判断某个 catalog 条目对“这次 Run”是否适用（applicable）。
# 依据 Manifest metadata 里的 model_family / model_type：
# - history.csv 只对 deep_learning 模型有意义；hyperparameter_search.csv 只对
#   traditional_ml 有意义；dscarnet_mapping.json 只对 dscarnet 模型有意义。
# - 其余条目：required 的必填项或磁盘上真实存在的文件才算适用。
# 文件存在但无数据行时也判不适用（借助 _csv_has_data），前端据此灰显而非提供空文件。
def _is_applicable(name: str, *, path: Path, manifest: dict[str, Any], exists: bool, required: bool) -> bool:
    metadata = manifest.get('metadata') if isinstance(manifest.get('metadata'), dict) else {}
    family = str(metadata.get('model_family') or '')
    model_type = str(metadata.get('model_type') or '')
    if name == 'history.csv':
        return family == 'deep_learning' and (not exists or _csv_has_data(path))
    if name == 'hyperparameter_search.csv':
        return family == 'traditional_ml' and (not exists or _csv_has_data(path))
    if name == 'dscarnet_mapping.json':
        return model_type == 'dscarnet'
    return required or exists


# 检查 config.json 是否包含本机/服务器路径字段（path、data_path、test_data_path、
# run_dir 或任何 *_path 键，递归遍历嵌套 dict/list）。
# 这是 finalize() 的内容级闸门：含路径的 config 会泄露部署机器目录结构，违背
# “config.json 只有在不含服务器路径时才可下载”的契约，此时强制不可下载。
# JSON 读不出来时保守地返回 True（当作含路径），宁错杀不放过。
def _json_contains_path_fields(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return True

    def contains(value: Any) -> bool:
        if isinstance(value, dict):
            return any(
                key in {'path', 'data_path', 'test_data_path', 'run_dir'}
                or key.endswith('_path')
                or contains(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return any(contains(item) for item in value)
        return False

    return contains(payload)


# RunArtifactWriter：单个 Run 目录的产物写入器 + Manifest 发布器 + 下载解析器。
# 三个使用阶段：
# 1) 写入阶段（训练 worker）：write_json / write_bytes / write_private_bytes /
#    write_run_result，全部走 _atomic_write；_path() 会拒绝任何含路径分隔符、
#    '.'、'..' 的 artifact 名，artifact 只能是 Run 目录下的扁平文件名。
# 2) 发布阶段：finalize() 扫描目录、按 catalog 策略生成带 sha256/大小的 manifest.json；
#    write_private_bytes 登记的名字与含服务器路径的 config.json 在此被强制不可下载。
# 3) 读取阶段（结果接口）：load_manifest / descriptors / resolve_download，
#    每次读取都重新校验 Manifest 与完整性，不信任任何缓存。
class RunArtifactWriter:
    """在单个 Run 目录内安全写入并最终发布一组产物。"""
    # run_dir 立即 resolve 成绝对路径，后续 _is_within 的包含判断才有稳定基准；
    # _private_names 登记 write_private_bytes 写过的文件，finalize 时强制不可下载。
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir).resolve()
        self._private_names: set[str] = set()

    # artifact 名必须是“扁平文件名”：Path(name).name != name 说明带目录分隔符，
    # '.'/'..' 显式拒绝。所有读写都经此入口拼路径，从源头杜绝目录穿越。
    def _path(self, name: str) -> Path:
        if not name or Path(name).name != name or name in {'.', '..'}:
            raise ValueError(f'invalid artifact name: {name!r}')
        return self.run_dir / name

    # 原子写核心：同目录建 .tmp 临时文件 → 写入 → flush+fsync 落盘 → os.replace
    # 覆盖目标。同文件系统内 replace 是原子操作，读者不会看到写一半的文件；
    # finally 清理残留临时文件，即使中途异常也不留垃圾。
    def _atomic_write(self, name: str, data: bytes) -> Path:
        path = self._path(name)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile('wb', dir=self.run_dir, delete=False, suffix='.tmp')
        temporary = Path(handle.name)
        try:
            with handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    def write_json(self, name: str, payload: object) -> Path:
        return self._atomic_write(name, json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8'))

    def write_bytes(self, name: str, data: bytes) -> Path:
        return self._atomic_write(name, data)

    # 写入但登记为私有（如 DSCARNet 的 AggMap/PCA joblib 拟合对象）：文件落盘
    # 供内部复用，finalize 时因 _private_names 被强制标记为不可下载。
    def write_private_bytes(self, name: str, data: bytes) -> Path:
        self._private_names.add(name)
        return self.write_bytes(name, data)

    # 训练结果字典的批量落盘入口：result['artifacts'] 中 bytes 值原样写、其余按
    # JSON 写；非 dict 或缺少 artifacts 键时静默跳过，兼容不同模型的返回结构。
    def write_run_result(self, result: object) -> None:
        if not isinstance(result, dict):
            return
        artifacts = result.get('artifacts')
        if isinstance(artifacts, dict):
            for name, payload in artifacts.items():
                if isinstance(payload, bytes):
                    self.write_bytes(str(name), payload)
                else:
                    self.write_json(str(name), payload)

    # finalize：训练结束后扫描 Run 目录，为每个文件生成 Manifest 条目并发布 manifest.json。
    # 关键行为：
    # - 跳过 manifest.json 自身与残留 .tmp 文件；按文件名排序保证输出确定性。
    # - 每个条目记录 sha256、size_bytes、downloadable、required、category、format、
    #   media_type；downloadable 是三重条件的合取：catalog 允许 + 未被
    #   write_private_bytes 标记私有 + （config.json 时）不含服务器路径字段。
    # - volatile 条目（status.json）额外打标，表示终态后仍会被刷新，校验时豁免。
    # - metadata（如 model_family/model_type）随 Manifest 保存，供 _is_applicable 使用。
    # manifest.json 本身也用 _atomic_write 发布——读者要么看到完整新版，要么看到旧版。
    def finalize(self, *, run_id: str, metadata: dict[str, object] | None = None, validate=None, active=lambda: None) -> dict[str, object]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        entries: dict[str, dict[str, object]] = {}
        for path in sorted(self.run_dir.iterdir(), key=lambda item: item.name):
            active()
            if not path.is_file() or path.name == 'manifest.json' or path.name.endswith('.tmp'):
                continue
            policy = _catalog_policy(path.name)
            volatile = bool(policy.get('volatile'))
            allowed_by_content = not (
                path.name == 'config.json' and _json_contains_path_fields(path)
            )
            entry: dict[str, object] = {
                'sha256': _sha256(path, active),
                'size_bytes': path.stat().st_size,
                'downloadable': (
                    bool(policy.get('downloadable'))
                    and path.name not in self._private_names
                    and allowed_by_content
                ),
                'required': bool(policy.get('required')),
                'category': str(policy.get('category') or 'internal'),
                'format': _format_for_name(path.name),
                'media_type': _media_type(path.name),
            }
            if volatile:
                entry['volatile'] = True
            if not allowed_by_content:
                entry['download_reason'] = '配置包含服务器路径，仅供内部使用'
            entries[path.name] = entry
        manifest: dict[str, object] = {
            'schema_version': MANIFEST_SCHEMA_VERSION,
            'run_id': run_id,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'artifacts': entries,
        }
        if metadata:
            manifest['metadata'] = metadata
        encoded = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
        if validate is not None:
            validate(encoded)
        self._atomic_write('manifest.json', encoded)
        return manifest

    # 读取并严格校验 manifest.json。校验项（任一失败抛 ManifestCorruptError）：
    # JSON 合法性、顶层必须是含 artifacts 对象的 dict、Manifest 内 run_id 与目录名一致、
    # 每个 artifact 名是合法扁平文件名、size_bytes 是非负 int（注意 bool 是 int 子类，
    # 要显式排除）、sha256 必须是 str。读侧从严，是为了尽早发现损坏/篡改，
    # 不让坏 Manifest 流向下游的结果页与下载逻辑。
    def load_manifest(self) -> dict[str, Any]:
        manifest_path = self.run_dir / 'manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ManifestCorruptError('manifest.json 不是有效 JSON') from exc
        if not isinstance(manifest, dict) or not isinstance(manifest.get('artifacts'), dict):
            raise ManifestCorruptError('manifest.json 缺少 artifacts 对象')
        manifest_run_id = manifest.get('run_id')
        if manifest_run_id is not None and str(manifest_run_id) != self.run_dir.name:
            raise ManifestCorruptError('manifest.json 的 run_id 与目录不一致')
        for name, entry in manifest['artifacts'].items():
            if not isinstance(name, str) or not name or Path(name).name != name or name in {'.', '..'}:
                raise ManifestCorruptError('manifest.json 含无效 artifact 文件名')
            if not isinstance(entry, dict):
                raise ManifestCorruptError(f'manifest.json 的 artifact 条目无效: {name}')
            size = entry.get('size_bytes')
            if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
                raise ManifestCorruptError(f'manifest.json 的 artifact 大小无效: {name}')
            sha256 = entry.get('sha256')
            if sha256 is not None and not isinstance(sha256, str):
                raise ManifestCorruptError(f'manifest.json 的 artifact 哈希无效: {name}')
        return manifest

    # 按名字在 Manifest 中定位条目并解析出安全的磁盘路径：
    # 条目必须存在且是 dict；路径 resolve 后必须仍在 run_dir 内（_is_within 防穿越）
    # 且真实存在，否则统一抛 FileNotFoundError。
    def _resolve_manifest_entry(self, manifest: dict[str, Any], name: str) -> tuple[dict[str, Any], Path]:
        entry = manifest['artifacts'].get(name)
        if not isinstance(entry, dict):
            raise FileNotFoundError(name)
        path = self._path(name).resolve()
        if not _is_within(path, self.run_dir) or not path.is_file():
            raise FileNotFoundError(name)
        return entry, path

    @staticmethod
    # 校验单个 artifact 与 Manifest 条目的一致性。
    # volatile 条目（status.json）直接豁免：它在终态提交后仍会被刷新，大小/哈希必然漂移。
    # verify_hash=False 时只做 O(1) 的大小检查（结果页列表用）；verify_hash=True 才
    # 重算 sha256（实际下载前用），避免列表页为每个文件全量读盘。
    def _verify_entry(path: Path, entry: dict[str, Any], *, verify_hash: bool = True) -> None:
        if entry.get('volatile'):
            return
        expected_size = entry.get('size_bytes')
        if expected_size is not None and int(expected_size) != path.stat().st_size:
            raise ArtifactIntegrityError(f'artifact size mismatch: {path.name}')
        expected_sha = entry.get('sha256')
        if verify_hash and expected_sha and str(expected_sha) != _sha256(path):
            raise ArtifactIntegrityError(f'artifact sha256 mismatch: {path.name}')

    # 下载前的最终闸门，所有检查在调用当下重新执行（不信任任何先前状态）：
    # 1) Manifest 必须可读且结构合法；2) 条目存在、路径未逃逸、文件在盘上；
    # 3) Manifest 条目自身标记 downloadable；4) 若是 v2 Manifest，文件名还必须在
    #    当前 ARTIFACT_CATALOG 白名单中——除非属于 LEGACY_DIRECT_DOWNLOAD_NAMES
    #    的窄范围历史兼容；5) 最后重算 sha256 做完整性校验。
    # 任一失败分别抛 PermissionError / FileNotFoundError / ArtifactIntegrityError。
    def resolve_download(self, name: str) -> Path:
        manifest = self.load_manifest()
        entry, path = self._resolve_manifest_entry(manifest, name)
        is_v2 = manifest.get('schema_version') == MANIFEST_SCHEMA_VERSION
        catalog_downloadable = bool(_catalog_policy(name).get('downloadable'))
        legacy_direct_download = name in LEGACY_DIRECT_DOWNLOAD_NAMES
        # v1 继续尊重历史 downloadable；v2 通常还必须通过当前显式 catalog。
        # 已发布过的全局重要性仅保留已知文件名的窄范围直接下载，不重新进入结果页。
        if not entry.get('downloadable') or (
            is_v2 and not catalog_downloadable and not legacy_direct_download
        ):
            raise PermissionError(f'artifact is not downloadable: {name}')
        self._verify_entry(path, entry, verify_hash=True)
        return path

    # 为结果页生成每个 artifact 的安全描述符（绝不暴露本机绝对路径）。
    # 输出是“catalog 全部已知名字 ∪ Manifest 实际登记名字”的并集，逐项给出：
    # - required/applicable：v2 才采纳 required 语义；applicable 由 _is_applicable 按
    #   模型家族/类型判定（但已登记在 Manifest 的常规文件一律视为适用）。
    # - integrity：ok / corrupt / missing / not_generated / volatile。结果页只做
    #   大小级校验（verify_hash=False），sha256 留到真正下载时再算。
    # - downloadable 是 exists + integrity==ok + applicable + catalog 允许 +
    #   Manifest 允许 的合取；不允许时附带人类可读的 reason 供前端灰显解释。
    # - LEGACY_DIRECT_DOWNLOAD_NAMES 被显式跳过，不进入结果页列表。
    def descriptors(self, *, run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """返回结果页所需的安全 artifact 描述，不暴露本机路径。"""
        manifest = self.load_manifest()
        manifest_entries: dict[str, Any] = manifest['artifacts']
        names = sorted(set(ARTIFACT_CATALOG) | set(manifest_entries))
        is_v2 = manifest.get('schema_version') == MANIFEST_SCHEMA_VERSION
        descriptors: list[dict[str, Any]] = []
        for name in names:
            if name in LEGACY_DIRECT_DOWNLOAD_NAMES:
                continue
            policy = _catalog_policy(name)
            entry = manifest_entries.get(name)
            exists = isinstance(entry, dict) and self._path(name).is_file()
            required = (
                bool(entry.get('required')) if isinstance(entry, dict) and entry.get('required') is not None
                else bool(policy.get('required'))
            ) if is_v2 else False
            applicable = _is_applicable(
                name,
                path=self._path(name),
                manifest=manifest,
                exists=exists,
                required=required,
            )
            if isinstance(entry, dict) and name not in {
                'history.csv',
                'hyperparameter_search.csv',
                'dscarnet_mapping.json',
                'status.json',
            }:
                applicable = True
            integrity = 'not_generated'
            reason: str | None = None
            if isinstance(entry, dict) and exists:
                if entry.get('volatile') or policy.get('volatile'):
                    # status.json 是终态提交后仍会刷新的兼容投影。旧 Manifest 没有
                    # volatile 标记时也按当前 catalog 解释，不能因此把 Run 判为 partial。
                    integrity = 'volatile'
                else:
                    try:
                        # 结果页只做常数时间的大小检查；实际下载前重新计算 sha256。
                        self._verify_entry(self._path(name), entry, verify_hash=False)
                        integrity = 'ok'
                    except ArtifactIntegrityError:
                        integrity = 'corrupt'
                        reason = '文件大小或内容哈希与 Manifest 不一致'
            elif isinstance(entry, dict):
                integrity = 'missing'
                reason = 'Manifest 已登记，但文件不存在'
            elif required:
                integrity = 'missing'
                reason = '必需结果文件未生成'

            catalog_downloadable = bool(policy.get('downloadable'))
            manifest_downloadable = bool(entry.get('downloadable')) if isinstance(entry, dict) else False
            downloadable = bool(
                exists
                and integrity == 'ok'
                and applicable
                and catalog_downloadable
                and manifest_downloadable
            )
            if not reason and exists and not catalog_downloadable:
                reason = '该文件属于内部产物，不在结果页下载白名单中'
            elif not reason and exists and catalog_downloadable and not manifest_downloadable:
                reason = str(entry.get('download_reason') or 'Manifest 策略不允许下载该文件')
            elif not reason and not applicable:
                if exists and name in {'history.csv', 'hyperparameter_search.csv'}:
                    reason = '该文件对当前模型不适用或没有可用数据行'
                else:
                    reason = '本次训练未生成该产物'

            descriptors.append(
                {
                    'name': name,
                    'label': str(policy.get('label') or name),
                    'category': str((entry or {}).get('category') or policy.get('category') or 'internal'),
                    'format': str((entry or {}).get('format') or _format_for_name(name)),
                    'media_type': str((entry or {}).get('media_type') or _media_type(name)),
                    'size_bytes': int(entry['size_bytes']) if isinstance(entry, dict) and entry.get('size_bytes') is not None else None,
                    'sha256': str(entry['sha256']) if isinstance(entry, dict) and entry.get('sha256') else None,
                    'required': required,
                    'applicable': applicable,
                    'exists': exists,
                    'integrity': integrity,
                    'downloadable': downloadable,
                    'reason': reason,
                    'download_url': (
                        f'/api/training/runs/{quote(run_id, safe="")}/artifact/{quote(name, safe="")}'
                        if downloadable
                        else None
                    ),
                    'suggested_filename': f'run_{run_id[:12]}__{name}',
                }
            )
        return manifest, descriptors

    # Manifest 缺失/损坏时的降级输出：仍按固定 ARTIFACT_CATALOG 逐项给出描述符，
    # 但 downloadable 一律 False、integrity 标 unverified/missing，并统一附上 reason。
    # 这样前端在“Run 产物不可用”时也能渲染完整列表并解释原因，而不是空白报错。
    def unavailable_descriptors(self, *, run_id: str, reason: str) -> list[dict[str, Any]]:
        """Manifest 不可用时仍返回固定 catalog，供前端逐项解释禁用原因。"""
        descriptors = []
        for name in sorted(ARTIFACT_CATALOG):
            policy = _catalog_policy(name)
            exists = self._path(name).is_file()
            descriptors.append(
                {
                    'name': name,
                    'label': str(policy.get('label') or name),
                    'category': str(policy.get('category') or 'internal'),
                    'format': _format_for_name(name),
                    'media_type': _media_type(name),
                    'size_bytes': self._path(name).stat().st_size if exists else None,
                    'sha256': None,
                    'required': bool(policy.get('required')),
                    'applicable': exists or bool(policy.get('required')),
                    'exists': exists,
                    'integrity': 'unverified' if exists else 'missing',
                    'downloadable': False,
                    'reason': reason,
                    'download_url': None,
                    'suggested_filename': f'run_{run_id[:12]}__{name}',
                }
            )
        return descriptors

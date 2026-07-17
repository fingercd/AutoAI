"""Run artifact 的原子写入、Manifest 发布和安全下载解析。

所有文件先写同目录临时文件、fsync 后再 os.replace。Run 只有在 Manifest 完成后
才能成功；下载时重新检查 Manifest、downloadable 标记、路径边界和文件存在性。
joblib 映射对象默认私有，避免把内部拟合对象当成公共 API。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..version import ARTIFACT_MANIFEST_CONTRACT_VERSION


MANIFEST_SCHEMA_VERSION = ARTIFACT_MANIFEST_CONTRACT_VERSION
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
    'cv_predictions.csv': {'label': 'OOF / 兼容预测明细', 'category': 'predictions', 'required': False, 'downloadable': True},
    'history.csv': {'label': '训练过程', 'category': 'training', 'required': False, 'downloadable': True},
    'hyperparameter_search.csv': {'label': '参数搜索记录', 'category': 'training', 'required': False, 'downloadable': True},
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


class ManifestCorruptError(ValueError):
    """Manifest 不是可解释的对象或 artifact 索引损坏。"""


class ArtifactIntegrityError(OSError):
    """artifact 的实际大小或内容哈希与 Manifest 不一致。"""


def _is_within(target: Path, root: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _format_for_name(name: str) -> str:
    suffix = Path(name).suffix.lower().lstrip('.')
    return suffix or 'binary'


def _media_type(name: str) -> str:
    return {
        'json': 'application/json',
        'csv': 'text/csv',
        'txt': 'text/plain',
    }.get(_format_for_name(name), 'application/octet-stream')


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


class RunArtifactWriter:
    """在单个 Run 目录内安全写入并最终发布一组产物。"""
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir).resolve()
        self._private_names: set[str] = set()

    def _path(self, name: str) -> Path:
        if not name or Path(name).name != name or name in {'.', '..'}:
            raise ValueError(f'invalid artifact name: {name!r}')
        return self.run_dir / name

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

    def write_private_bytes(self, name: str, data: bytes) -> Path:
        self._private_names.add(name)
        return self.write_bytes(name, data)

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

    def finalize(self, *, run_id: str, metadata: dict[str, object] | None = None) -> dict[str, object]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        entries: dict[str, dict[str, object]] = {}
        for path in sorted(self.run_dir.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.name == 'manifest.json' or path.name.endswith('.tmp'):
                continue
            policy = _catalog_policy(path.name)
            volatile = bool(policy.get('volatile'))
            allowed_by_content = not (
                path.name == 'config.json' and _json_contains_path_fields(path)
            )
            entry: dict[str, object] = {
                'sha256': _sha256(path),
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
        self._atomic_write('manifest.json', encoded)
        return manifest

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

    def _resolve_manifest_entry(self, manifest: dict[str, Any], name: str) -> tuple[dict[str, Any], Path]:
        entry = manifest['artifacts'].get(name)
        if not isinstance(entry, dict):
            raise FileNotFoundError(name)
        path = self._path(name).resolve()
        if not _is_within(path, self.run_dir) or not path.is_file():
            raise FileNotFoundError(name)
        return entry, path

    @staticmethod
    def _verify_entry(path: Path, entry: dict[str, Any], *, verify_hash: bool = True) -> None:
        if entry.get('volatile'):
            return
        expected_size = entry.get('size_bytes')
        if expected_size is not None and int(expected_size) != path.stat().st_size:
            raise ArtifactIntegrityError(f'artifact size mismatch: {path.name}')
        expected_sha = entry.get('sha256')
        if verify_hash and expected_sha and str(expected_sha) != _sha256(path):
            raise ArtifactIntegrityError(f'artifact sha256 mismatch: {path.name}')

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

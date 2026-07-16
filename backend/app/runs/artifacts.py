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
            entries[path.name] = {
                'sha256': _sha256(path),
                'size_bytes': path.stat().st_size,
                'downloadable': path.name not in self._private_names and not path.name.endswith('.joblib'),
            }
        manifest: dict[str, object] = {
            'run_id': run_id,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'artifacts': entries,
        }
        if metadata:
            manifest['metadata'] = metadata
        encoded = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
        self._atomic_write('manifest.json', encoded)
        return manifest

    def resolve_download(self, name: str) -> Path:
        manifest_path = self.run_dir / 'manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        entry = (manifest.get('artifacts') or {}).get(name)
        if entry is None:
            raise FileNotFoundError(name)
        if not isinstance(entry, dict) or not entry.get('downloadable'):
            raise PermissionError(f'artifact is not downloadable: {name}')
        path = self._path(name).resolve()
        if not _is_within(path, self.run_dir) or not path.is_file():
            raise FileNotFoundError(name)
        return path

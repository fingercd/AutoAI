"""Durable, scope-checked comparison archives, independent of Run success state."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile

from .batch_projection import project_model_comparison
from .comparison_figures import DRAWING_VERSION, METRICS, figure_data, figure_specs, render_figure
from .contracts import Principal
from .repository import RunNotFound

ARCHIVE_VERSION = 'batch-comparison-archive-v1'
_SAFE_ID = re.compile(r'^[a-zA-Z0-9_-]{1,128}$')
_SAFE_FILE = re.compile(r'^(comparison\.json|[a-zA-Z0-9_-]+\.(csv|svg|png))$')


class ArchiveBusy(RuntimeError):
    pass


class ArchiveUnavailable(ValueError):
    pass


def archive_root(repository):
    # Repository-relative storage also isolates tests and alternate deployments.
    return repository.database_path.parent / 'batches'


def archive_dir(repository, batch_id):
    if not _SAFE_ID.fullmatch(batch_id):
        raise ArchiveUnavailable('无效批次标识')
    root = archive_root(repository).resolve()
    path = root / batch_id
    if path.resolve().parent != root or path.is_symlink():
        raise ArchiveUnavailable('无效归档路径')
    return path


def discard_batch_archive(repository, batch_id):
    path = archive_dir(repository, batch_id)
    if path.exists():
        shutil.rmtree(path)


@contextmanager
def _lock(repository, batch_id):
    lock_root = archive_root(repository) / '.locks'
    lock_root.mkdir(parents=True, exist_ok=True)
    with (lock_root / f'{batch_id}.lock').open('a+b') as stream:
        stream.seek(0); stream.write(b'0'); stream.flush(); stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ArchiveBusy('正在保存对比结果') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _json(data):
    return json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _public(value):
    if isinstance(value, dict):
        return {key: _public(item) for key, item in value.items() if not any(word in key.lower() for word in ('path', 'owner', 'tenant', 'token', 'traceback'))}
    if isinstance(value, list):
        return [_public(item) for item in value]
    if isinstance(value, str) and (value.startswith(('/', '\\')) or re.match(r'^[A-Za-z]:[\\/]', value)):
        return None
    return value


def _manifest(directory, batch_id):
    try:
        manifest_path = directory / 'manifest.json'
        if manifest_path.is_symlink():
            raise ValueError('symlink manifest')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['schema_version'] != ARCHIVE_VERSION or manifest['batch_id'] != batch_id:
            raise ValueError('wrong manifest')
        if not isinstance(manifest['files'], dict):
            raise ValueError('invalid file catalog')
        for name, item in manifest['files'].items():
            if not _SAFE_FILE.fullmatch(name) or not isinstance(item, dict) or not re.fullmatch('[a-f0-9]{64}', str(item.get('sha256', ''))) or not isinstance(item.get('size_bytes'), int) or item['size_bytes'] < 0:
                raise ValueError('invalid filename')
        return manifest
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ArchiveUnavailable('归档缺失或损坏，请重新生成') from exc


def _read_file(directory, manifest, name):
    if not _SAFE_FILE.fullmatch(name) or name not in manifest['files']:
        raise ArchiveUnavailable('该文件不在归档下载清单中')
    path = directory / name
    if path.is_symlink() or path.resolve().parent != directory.resolve():
        raise ArchiveUnavailable('无效文件路径')
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ArchiveUnavailable('归档文件缺失') from exc
    item = manifest['files'][name]
    if len(content) != item['size_bytes'] or _hash(content) != item['sha256']:
        raise ArchiveUnavailable('归档文件校验失败，请重新生成')
    return content


def scoped_records(repository, batch_id, principal):
    batch = repository.get_batch_scoped(batch_id, principal=principal)
    records = repository.list_batch_runs_scoped(batch_id, principal=principal)
    return batch, records


def archive_status(repository, batch_id, principal):
    _batch, records = scoped_records(repository, batch_id, principal)
    if any(record.state == 'cancelled' for record in records):
        return {'state': 'discarded', 'message': '已停止，不保存对比结果'}
    if any(record.state in {'queued', 'running'} for record in records):
        return {'state': 'pending', 'message': '训练结束后自动保存'}
    directory = archive_dir(repository, batch_id)
    try:
        manifest = _manifest(directory, batch_id)
        # The comparison document is required; downloadable images verify on access.
        _read_file(directory, manifest, 'comparison.json')
        return {'state': 'ready', 'created_at': manifest['created_at'], 'drawing_version': manifest['drawing_version'], 'files': list(manifest['files'])}
    except ArchiveUnavailable:
        state = 'failed' if (directory / 'error.json').exists() or (directory / 'manifest.json').exists() else 'missing'
        return {'state': state, 'message': '对比归档尚未完成，可重试保存'}


def load_archive(repository, batch_id, principal):
    status = archive_status(repository, batch_id, principal)
    if status['state'] != 'ready':
        raise ArchiveUnavailable(status['message'])
    directory = archive_dir(repository, batch_id)
    manifest = _manifest(directory, batch_id)
    return json.loads(_read_file(directory, manifest, 'comparison.json'))


def _csv(spec):
    stream = StringIO(newline='')
    writer = csv.writer(stream)
    # Quote formula-like labels so opening an exported CSV cannot execute a formula.
    safe = lambda value: "'" + value if isinstance(value, str) and value.startswith(('=', '+', '-', '@')) else value
    writer.writerow(['Model / Class / Feature', *map(safe, spec['columns'])])
    for name, row in zip(spec['rows'], spec['values']):
        writer.writerow([safe(name), *row])
    return ('\ufeff' + stream.getvalue()).encode('utf-8')


def ensure_archive(repository, batch_id, principal, run_dir_for, *, force=False):
    directory = archive_dir(repository, batch_id)
    batch, records = scoped_records(repository, batch_id, principal)
    status = archive_status(repository, batch_id, principal)
    if status['state'] in {'pending', 'discarded'}:
        raise ArchiveUnavailable(status['message'])
    if status['state'] == 'ready' and not force:
        return status
    with _lock(repository, batch_id):
        if not force and archive_status(repository, batch_id, principal)['state'] == 'ready':
            return archive_status(repository, batch_id, principal)
        staging = Path(tempfile.mkdtemp(prefix=f'.{batch_id}-', dir=archive_root(repository)))
        try:
            comparison = _public(project_model_comparison(batch_id=batch_id, records=records, run_dir_for=run_dir_for))
            # Internal legacy stability/history payloads do not belong to the new UI/archive.
            comparison.pop('repeat_stability', None)
            comparison.pop('run_comparisons', None)
            comparison['archive'] = {'schema_version': ARCHIVE_VERSION, 'drawing_version': DRAWING_VERSION,
                                    'dataset_name': batch.dataset_snapshot.get('name') or batch.config.get('dataset_name') or batch.dataset_id,
                                    'dataset_id': batch.dataset_id, 'created_at': batch.created_at,
                                    'runs': [{'run_id': r.run_id, 'model_type': r.config.get('model_type'), 'state': r.state,
                                              'message': '训练失败，详见该模型记录' if r.state == 'failed' else None} for r in records]}
            entries = {}
            def write(name, content):
                if not _SAFE_FILE.fullmatch(name):
                    raise ArchiveUnavailable('无效归档文件名')
                (staging / name).write_bytes(content)
                entries[name] = {'size_bytes': len(content), 'sha256': _hash(content)}
            write('comparison.json', _json(comparison))
            for spec in figure_specs(comparison):
                _, latest = scoped_records(repository, batch_id, principal)
                if any(record.state not in {'succeeded', 'failed'} for record in latest):
                    raise ArchiveUnavailable('批次已停止，不再生成图像')
                options = {key: value for key, value in spec.items() if key != 'id'}
                width = 900 if spec['kind'] in {'recall', 'features', 'samples'} else 360
                write(spec['id'] + '.csv', _csv(figure_data(comparison, **options)))
                for fmt in ('svg', 'png'):
                    content, _ = render_figure(comparison, format=fmt, width=width, **options)
                    write(spec['id'] + '.' + fmt, content)
            manifest = {'schema_version': ARCHIVE_VERSION, 'drawing_version': DRAWING_VERSION, 'batch_id': batch_id,
                        'created_at': datetime.now(timezone.utc).isoformat(), 'files': entries}
            (staging / 'manifest.json').write_bytes(_json(manifest))
            # Publish under the same SQLite write lock as STOP/DELETE. If cancellation
            # wins, this build is discarded. If publishing wins, STOP deletes it.
            with repository._connection() as connection:
                connection.execute('BEGIN IMMEDIATE')
                current = connection.execute('SELECT state FROM runs WHERE batch_id=?', (batch_id,)).fetchall()
                if not current or any(row['state'] not in {'succeeded', 'failed'} for row in current):
                    raise ArchiveUnavailable('批次已停止、删除或尚未结束')
                if directory.exists():
                    shutil.rmtree(directory)
                os.replace(staging, directory)
                connection.commit()
            return archive_status(repository, batch_id, principal)
        except Exception:
            # This error marker is separate from Run state and is never downloadable.
            with repository._connection() as connection:
                connection.execute('BEGIN IMMEDIATE')
                current = connection.execute('SELECT state FROM runs WHERE batch_id=?', (batch_id,)).fetchall()
                if current and all(row['state'] in {'succeeded', 'failed'} for row in current):
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / 'error.json').write_bytes(_json({'message': '对比归档失败，请重试'}))
                connection.commit()
            raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def archive_download(repository, batch_id, principal, name):
    if archive_status(repository, batch_id, principal)['state'] != 'ready':
        raise ArchiveUnavailable('归档尚未就绪')
    directory = archive_dir(repository, batch_id)
    manifest = _manifest(directory, batch_id)
    if name != 'all.zip':
        return _read_file(directory, manifest, name)
    output = BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for filename in manifest['files']:
            bundle.writestr(filename, _read_file(directory, manifest, filename))
        bundle.writestr('manifest.json', _json(manifest))
    return output.getvalue()


def finalize_worker_batch(repository, run):
    """Called after completion, outside the Run success/failure exception handler."""
    if not run.batch_id:
        return
    from ..paths import RUNS_DIR
    principal = Principal(owner_id=run.owner_id, tenant_id=run.tenant_id)
    status = archive_status(repository, run.batch_id, principal)
    if status['state'] in {'missing', 'failed'}:
        from threading import Event, Thread
        stop = Event()
        def heartbeat():
            while not stop.wait(5):
                repository.record_worker_heartbeat(worker_id=run.worker_id or 'archive-worker', now=datetime.now(timezone.utc))
        thread = Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            ensure_archive(repository, run.batch_id, principal, lambda rid: RUNS_DIR / rid)
        finally:
            stop.set(); thread.join(timeout=1)

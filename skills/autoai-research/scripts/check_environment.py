"""Check a selected Python without requiring third-party packages in this launcher.

Only inspect packages and import libraries. Do not install packages, start services,
read credentials, upload data, or train models.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import redirect_stdout
import hashlib
from importlib import import_module, metadata
import io
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import warnings


SKILL_ROOT = Path(__file__).resolve().parents[1]
IMPORT_NAMES = {
    'httpx': 'httpx', 'openpyxl': 'openpyxl', 'fastapi': 'fastapi',
    'uvicorn': 'uvicorn', 'python-multipart': 'python_multipart',
    'pydantic': 'pydantic', 'numpy': 'numpy', 'pandas': 'pandas',
    'scipy': 'scipy', 'joblib': 'joblib', 'matplotlib': 'matplotlib',
    'scikit-learn': 'sklearn', 'xgboost': 'xgboost', 'torch': 'torch',
    'rampy': 'rampy',
}


def requirement_lines(path):
    lines = []
    for raw in path.read_text(encoding='utf-8-sig').splitlines():
        line = raw.partition('#')[0].strip()
        if line.startswith('-r '):
            lines.extend(requirement_lines(path.parent / line[3:].strip()))
        elif line:
            lines.append(line)
    return lines


def dependency_issues(requirements):
    """Inspect the active dependency graph rooted in this task's requirements."""
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name

    pending = deque(Requirement(text) for text in requirements)
    inspected, seen_issues, issues, versions = set(), set(), [], {}
    while pending:
        requirement = pending.popleft()
        name = canonicalize_name(requirement.name)
        try:
            distribution = metadata.distribution(name)
            version = distribution.version
        except metadata.PackageNotFoundError:
            distribution, version = None, None
        versions[name] = version
        if version is None or not requirement.specifier.contains(version, prereleases=True):
            issue_key = (str(requirement), version)
            if issue_key not in seen_issues:
                issues.append({'package': name, 'required': str(requirement),
                               'installed': version,
                               'status': 'missing' if version is None else 'version_conflict'})
                seen_issues.add(issue_key)
        scope = (name, tuple(sorted(requirement.extras)))
        if distribution is None or scope in inspected:
            continue
        inspected.add(scope)
        for text in distribution.requires or []:
            child = Requirement(text)
            contexts = {'', *requirement.extras}
            if child.marker is None or any(child.marker.evaluate({'extra': extra}) for extra in contexts):
                pending.append(child)
    return issues, versions


def probe(payload):
    result = {'python': sys.executable, 'python_version': platform.python_version(),
              'platform': platform.platform(), 'bits': struct.calcsize('P') * 8,
              'blocking_issues': [], 'warnings': [], 'imports': [], 'versions': {}}
    if sys.version_info < (3, 10):
        result['blocking_issues'].append({'code': 'python_too_old', 'recommended': 'Python 3.12'})
    elif sys.version_info[:2] != (3, 12):
        result['warnings'].append({'code': 'python_not_verified', 'verified_major_minor': '3.12'})
    if payload['mode'] == 'local' and result['bits'] != 64:
        result['blocking_issues'].append({'code': 'requires_64_bit_python'})
    try:
        import_module('pip')
    except ImportError:
        result['blocking_issues'].append({'code': 'pip_missing'})
        return result
    issues, versions = dependency_issues(payload['requirements'])
    result['blocking_issues'].extend(issues)
    result['versions'] = versions
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name
    modules = {IMPORT_NAMES[canonicalize_name(Requirement(text).name)]
               for text in payload['requirements']
               if canonicalize_name(Requirement(text).name) in IMPORT_NAMES}
    for name in sorted(modules):
        with warnings.catch_warnings(record=True) as caught, redirect_stdout(io.StringIO()):
            warnings.simplefilter('always')
            try:
                module = import_module(name)
                result['imports'].append({'module': name, 'status': 'ok'})
                if name == 'torch':
                    result['torch'] = {'version': module.__version__,
                                       'cuda_build': module.version.cuda,
                                       'cuda_available': module.cuda.is_available()}
            except Exception as exc:
                result['imports'].append({'module': name, 'status': 'failed', 'error_type': type(exc).__name__})
                result['blocking_issues'].append({'code': 'import_failed', 'module': name,
                                                 'error_type': type(exc).__name__})
        result['warnings'].extend({'code': 'import_warning', 'module': name, 'message': str(item.message)}
                                  for item in caught)
    return result


def check_environment(*, python, mode, project_dir=None, timeout=90):
    files = [SKILL_ROOT / 'requirements.txt']
    if mode == 'local':
        if project_dir is None:
            return {'ready': False, 'mode': mode, 'blocking_issues': [{'code': 'project_missing'}]}
        project = Path(project_dir).resolve()
        expected = ['backend/requirements.txt', 'backend/app/main.py',
                    'backend/app/routers/preflight.py', 'backend/app/runs/worker.py', 'run_classic.py']
        missing = [name for name in expected if not (project / name).is_file()]
        if missing:
            return {'ready': False, 'mode': mode, 'project_dir': str(project),
                    'blocking_issues': [{'code': 'project_incomplete', 'missing_files': missing}]}
        files.append(project / 'backend/requirements.txt')
    payload = {'mode': mode, 'requirements': [text for path in files for text in requirement_lines(path)]}
    try:
        completed = subprocess.run([str(python), str(Path(__file__).resolve()), '_probe'],
                                   input=json.dumps(payload), capture_output=True, encoding='utf-8',
                                   env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'ready': False, 'mode': mode,
                'blocking_issues': [{'code': 'python_probe_failed', 'error_type': type(exc).__name__}]}
    if completed.returncode != 0:
        return {'ready': False, 'mode': mode,
                'blocking_issues': [{'code': 'python_probe_failed', 'exit_code': completed.returncode}]}
    report = json.loads(completed.stdout)
    report.update(mode=mode, ready=not report['blocking_issues'],
                  checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  requirements=[{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                                for path in files])
    if mode == 'local':
        report['project_dir'] = str(Path(project_dir).resolve())
    return report


def main():
    if sys.argv[1:] == ['_probe']:
        print(json.dumps(probe(json.load(sys.stdin)), ensure_ascii=True))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', help='Exact interpreter to check; defaults to runtime.json or this interpreter')
    parser.add_argument('--mode', choices=['local', 'client'], default='local')
    parser.add_argument('--project-dir', type=Path)
    parser.add_argument('--output', type=Path, help='Optional UTF-8 local diagnostic report')
    args = parser.parse_args()
    settings_path = SKILL_ROOT / 'runtime.json'
    settings = json.loads(settings_path.read_text(encoding='utf-8-sig')) if settings_path.exists() else {}
    report = check_environment(python=args.python or settings.get('python') or sys.executable,
                               mode=args.mode,
                               project_dir=args.project_dir or os.environ.get('AUTOAI_PROJECT_DIR') or settings.get('project_dir'))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True))
    raise SystemExit(0 if report['ready'] else 1)


if __name__ == '__main__':
    main()

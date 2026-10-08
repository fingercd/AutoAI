"""Environment bootstrap checks use local fixtures and never install packages."""
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / 'skills' / 'autoai-research' / 'scripts'
spec = importlib.util.spec_from_file_location('skill_environment', SCRIPTS / 'check_environment.py')
environment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(environment)


def test_dependency_check_finds_related_transitive_conflict_only(monkeypatch):
    installed = {
        'httpx': SimpleNamespace(version='0.27.0', requires=['httpcore==1.*']),
        'httpcore': SimpleNamespace(version='1.0.2', requires=['h11<0.15,>=0.13']),
        'h11': SimpleNamespace(version='0.16.0', requires=[]),
    }
    monkeypatch.setattr(environment.metadata, 'distribution', installed.__getitem__)
    issues, versions = environment.dependency_issues(['httpx>=0.27,<1'])
    assert issues == [{'package': 'h11', 'required': 'h11<0.15,>=0.13',
                       'installed': '0.16.0', 'status': 'version_conflict'}]
    assert set(versions) == set(installed)


def test_installed_prerelease_is_not_rejected_by_an_unbounded_requirement(monkeypatch):
    monkeypatch.setattr(environment.metadata, 'distribution',
                        lambda name: SimpleNamespace(version='6.0.0rc0', requires=[]))
    assert environment.dependency_issues(['plotly'])[0] == []


def test_incomplete_project_is_reported_before_spawning_python(tmp_path, monkeypatch):
    monkeypatch.setattr(environment.subprocess, 'run', lambda *args, **kwargs: pytest.fail('no subprocess needed'))
    report = environment.check_environment(python=sys.executable, mode='local', project_dir=tmp_path)
    assert not report['ready']
    assert report['blocking_issues'][0]['code'] == 'project_incomplete'
    assert 'backend/requirements.txt' in report['blocking_issues'][0]['missing_files']


def test_checker_runs_with_a_target_environment_without_pip(tmp_path):
    target = tmp_path / 'empty-environment'
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(target)], check=True)
    python = target / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    report = environment.check_environment(python=python, mode='client')
    assert not report['ready']
    assert report['blocking_issues'] == [{'code': 'pip_missing'}]
    assert Path(report['python']) == python


def test_unavailable_interpreter_produces_an_actionable_report(tmp_path):
    report = environment.check_environment(python=tmp_path / 'missing-python', mode='client')
    assert not report['ready']
    assert report['blocking_issues'][0]['code'] == 'python_probe_failed'

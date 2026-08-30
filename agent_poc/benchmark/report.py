"""Safe benchmark trial reports (JSON and Markdown)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_GIT_REVISION = re.compile(r'^[0-9a-f]{40}$')
_SAFE_CODE = re.compile(r'^[a-z0-9][a-z0-9_.-]{0,127}$')
_ABSOLUTE_PATH = re.compile(
    r'(?:[A-Za-z]:[\\/]|/(?:users|home|var|tmp|opt|srv|etc|root|mnt|data)/)'
    r'[^\s,;]*',
    flags=re.IGNORECASE,
)
_FORBIDDEN_KEY_PARTS = (
    'total_score',
    'token',
    'prompt',
    'raw_error',
    'prediction',
    'confusion',
    'artifact',
    'path',
)


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{field} must be a non-negative integer')
    return value


def _finite_nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{field} must be a finite non-negative number')
    selected = float(value)
    if not math.isfinite(selected) or selected < 0.0:
        raise ValueError(f'{field} must be a finite non-negative number')
    return selected


def _optional_metric(value: object, field: str) -> float | None:
    if value is None:
        return None
    selected = _finite_nonnegative(value, field)
    if selected > 1.0:
        raise ValueError(f'{field} must be within [0, 1]')
    return selected


def _safe_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or _ABSOLUTE_PATH.search(value):
        raise ValueError(f'{field} must be a non-empty path-free string')
    return value


def assert_report_safe(value: object, *, location: str = 'report') -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in _FORBIDDEN_KEY_PARTS):
                raise ValueError(f'forbidden report key at {location}: {key}')
            assert_report_safe(item, location=f'{location}.{key}')
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_report_safe(item, location=f'{location}[{index}]')
    elif isinstance(value, str) and _ABSOLUTE_PATH.search(value):
        raise ValueError(f'absolute path at {location}')


def build_trial_report(
    *,
    dataset_name: str,
    dataset_role: str,
    dataset_sha256: str,
    code_revision: str,
    model_key: str,
    model_revision: str,
    seed: int,
    status: str,
    failure_code: str | None,
    macro_f1: float | None,
    balanced_accuracy: float | None,
    wall_clock_seconds: float,
    llm_call_count: int,
    upload_api_attempts: int,
    agent_api_attempts: int,
    evaluator_api_attempts: int,
    retry_attempt_count: int,
    model_fit_count: int | None,
) -> dict[str, Any]:
    """Build the only persisted per-trial shape.

    The smoke dataset remains auditable but is explicitly excluded from the
    formal metric aggregate.
    """
    dataset_name = _safe_text(dataset_name, 'dataset_name')
    dataset_role = _safe_text(dataset_role, 'dataset_role')
    if not isinstance(dataset_sha256, str) or not _SHA256.fullmatch(dataset_sha256):
        raise ValueError('dataset_sha256 must be lowercase SHA-256')
    code_revision = _safe_text(code_revision, 'code_revision')
    if not _GIT_REVISION.fullmatch(code_revision):
        raise ValueError('code_revision must be a lowercase 40-character Git SHA')
    model_key = _safe_text(model_key, 'model_key')
    model_revision = _safe_text(model_revision, 'model_revision')
    status = _safe_text(status, 'status')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    if failure_code is not None and (
        not isinstance(failure_code, str) or not _SAFE_CODE.fullmatch(failure_code)
    ):
        raise ValueError('failure_code must be a safe public code')
    if status == 'completed' and failure_code is not None:
        raise ValueError('completed trial cannot contain failure_code')
    if status == 'completed' and (
        macro_f1 is None or balanced_accuracy is None
    ):
        raise ValueError('completed trial requires both evaluation metrics')
    if status != 'completed' and (macro_f1 is not None or balanced_accuracy is not None):
        raise ValueError('non-completed trial cannot publish evaluation metrics')

    model_fits = (
        None
        if model_fit_count is None
        else _nonnegative_int(model_fit_count, 'model_fit_count')
    )
    report = {
        'schema_version': 'benchmark-trial-v1',
        'dataset': {
            'name': dataset_name,
            'role': dataset_role,
            'sha256': dataset_sha256,
            'included_in_formal_summary': dataset_role != 'pipeline_smoke_only',
        },
        'revision': {
            'code': code_revision,
            'model_key': model_key,
            'model': model_revision,
        },
        'protocol': {
            'seed': seed,
            'split': 'stratified_holdout',
            'ratio': '8:1:1',
        },
        'outcome': {
            'status': status,
            'failure_code': failure_code,
            'metrics': {
                'macro_f1': _optional_metric(macro_f1, 'macro_f1'),
                'balanced_accuracy': _optional_metric(
                    balanced_accuracy, 'balanced_accuracy'
                ),
            },
        },
        'cost': {
            'wall_clock_seconds': round(
                _finite_nonnegative(wall_clock_seconds, 'wall_clock_seconds'), 6
            ),
            'llm_call_count': _nonnegative_int(llm_call_count, 'llm_call_count'),
            'api_attempts': {
                'upload': _nonnegative_int(
                    upload_api_attempts, 'upload_api_attempts'
                ),
                'agent': _nonnegative_int(agent_api_attempts, 'agent_api_attempts'),
                'evaluator': _nonnegative_int(
                    evaluator_api_attempts, 'evaluator_api_attempts'
                ),
            },
            'retry_attempt_count': _nonnegative_int(
                retry_attempt_count, 'retry_attempt_count'
            ),
            'model_fit_count': model_fits,
        },
    }
    assert_report_safe(report)
    return report


def render_trial_markdown(report: dict[str, Any]) -> str:
    assert_report_safe(report)
    if report.get('schema_version') != 'benchmark-trial-v1':
        raise ValueError('unsupported trial report')
    dataset = report['dataset']
    outcome = report['outcome']
    metrics = outcome['metrics']
    cost = report['cost']
    heading = (
        '流水线 Smoke（不进入正式汇总）'
        if not dataset['included_in_formal_summary']
        else '正式评测 Trial'
    )
    metric_text = (
        f"macro-F1={metrics['macro_f1']:.6f}，"
        f"balanced accuracy={metrics['balanced_accuracy']:.6f}"
        if metrics['macro_f1'] is not None
        and metrics['balanced_accuracy'] is not None
        else '未产生独立评测指标'
    )
    attempts = cost['api_attempts']
    lines = [
        f'# {heading}: {dataset["name"]}',
        '',
        f'- 数据角色：`{dataset["role"]}`',
        f'- 数据 SHA-256：`{dataset["sha256"]}`',
        f'- 协议：seed={report["protocol"]["seed"]}，8:1:1 stratified holdout',
        f'- 状态：`{outcome["status"]}`',
        f'- 失败码：`{outcome["failure_code"] or "none"}`',
        f'- 指标：{metric_text}',
        f'- 耗时：{cost["wall_clock_seconds"]:.6f}s',
        f'- 调用：LLM={cost["llm_call_count"]}；上传 API={attempts["upload"]}；Agent API={attempts["agent"]}；Evaluator API={attempts["evaluator"]}；重试={cost["retry_attempt_count"]}；model fits={cost["model_fit_count"]}',
        '',
    ]
    return '\n'.join(lines)


def write_trial_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    assert_report_safe(report)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', report['dataset']['name'])
    json_path = output_dir / f'{safe_name}.trial-v1.json'
    markdown_path = output_dir / f'{safe_name}.trial-v1.md'
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    markdown_path.write_text(render_trial_markdown(report), encoding='utf-8')
    return json_path, markdown_path


def summarize_formal_trials(trials: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Keep per-dataset outcomes and add transparent, metric-specific totals."""
    trials_list = list(trials)
    formal: list[dict[str, Any]] = []
    for trial in trials_list:
        assert_report_safe(trial)
        if trial.get('schema_version') != 'benchmark-trial-v1':
            raise ValueError('unsupported trial report')
        if trial['dataset']['included_in_formal_summary']:
            formal.append(trial)
    completed = [item for item in formal if item['outcome']['status'] == 'completed']
    macro = [item['outcome']['metrics']['macro_f1'] for item in completed]
    balanced = [
        item['outcome']['metrics']['balanced_accuracy'] for item in completed
    ]
    def outcome_row(item: dict[str, Any]) -> dict[str, Any]:
        return {
            'dataset': item['dataset']['name'],
            'role': item['dataset']['role'],
            'status': item['outcome']['status'],
            'failure_code': item['outcome']['failure_code'],
            'metrics': dict(item['outcome']['metrics']),
        }

    fit_counts = [
        item['cost']['model_fit_count']
        for item in formal
        if item['cost']['model_fit_count'] is not None
    ]
    summary = {
        'formal_trial_count': len(formal),
        'completed_formal_trial_count': len(completed),
        'failure_rate': (
            (len(formal) - len(completed)) / len(formal) if formal else None
        ),
        'mean_macro_f1': sum(macro) / len(macro) if macro else None,
        'mean_balanced_accuracy': (
            sum(balanced) / len(balanced) if balanced else None
        ),
        'formal_trials': [outcome_row(item) for item in formal],
        'cost_totals': {
            'trial_count': len(formal),
            'wall_clock_seconds': round(
                sum(item['cost']['wall_clock_seconds'] for item in formal), 6
            ),
            'llm_call_count': sum(
                item['cost']['llm_call_count'] for item in formal
            ),
            'api_attempts': {
                kind: sum(
                    item['cost']['api_attempts'][kind] for item in formal
                )
                for kind in ('upload', 'agent', 'evaluator')
            },
            'retry_attempt_count': sum(
                item['cost']['retry_attempt_count'] for item in formal
            ),
            'model_fit_count': {
                'known_total': sum(fit_counts),
                'known_trial_count': len(fit_counts),
                'formal_trial_count': len(formal),
                'unknown_trial_count': len(formal) - len(fit_counts),
            },
        },
        'smoke_trials': [
            outcome_row(item)
            for item in trials_list
            if not item['dataset']['included_in_formal_summary']
        ],
    }
    assert_report_safe(summary)
    return summary

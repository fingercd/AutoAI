"""agent-observation-v1 的安全投影与 Manifest 校验。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..runs.artifacts import ArtifactIntegrityError, ManifestCorruptError, RunArtifactWriter
from ..runs.contracts import RunRecord, public_error_message
from .contracts import AGENT_API_CONTRACT_VERSION, AGENT_OBSERVATION_VERSION


VALIDATION_METRICS = (
    'accuracy', 'balanced_accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1',
)
SAFE_PROGRESS_KEYS = ('stage', 'phase', 'epoch', 'epochs', 'percent', 'message')


def safe_progress(record: RunRecord) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in SAFE_PROGRESS_KEYS:
        value = record.progress.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            if isinstance(value, float) and not math.isfinite(value):
                continue
            result[key] = value
    return result


def validation_from_manifest(run_dir: Path, *, run_id: str) -> tuple[str, dict[str, float], str | None]:
    """只读取通过 v2 Manifest 大小和 SHA-256 校验的 metrics.valid。"""
    writer = RunArtifactWriter(run_dir)
    try:
        _manifest, descriptors = writer.descriptors(run_id=run_id)
        damaged = any(
            item.get('integrity') in {'missing', 'corrupt'}
            and (item.get('required') or item.get('applicable'))
            for item in descriptors
        )
        if damaged:
            return 'unavailable', {}, 'agent_manifest_incomplete'
        metrics_path = writer.resolve_download('metrics.json')
        payload = json.loads(metrics_path.read_text(encoding='utf-8-sig'))
    except FileNotFoundError:
        return 'unavailable', {}, 'agent_manifest_missing'
    except (ManifestCorruptError, ArtifactIntegrityError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return 'unavailable', {}, 'agent_manifest_invalid'
    if not isinstance(payload, dict) or not isinstance(payload.get('valid'), dict):
        return 'unavailable', {}, 'agent_validation_unavailable'
    metrics: dict[str, float] = {}
    for key in VALIDATION_METRICS:
        value = payload['valid'].get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                metrics[key] = number
    if not metrics:
        return 'unavailable', {}, 'agent_validation_unavailable'
    return 'ready', metrics, None


def allowed_actions(*, session_state: str, run_state: str, validation_status: str,
                    remaining_runs: int) -> list[str]:
    if session_state == 'finalized':
        return ['inspect_ml_session']
    if run_state in {'queued', 'running'}:
        return ['observe_ml_experiment', 'inspect_ml_session']
    actions = ['inspect_ml_session']
    if remaining_runs > 0:
        actions.append('submit_ml_experiment')
    if run_state == 'succeeded' and validation_status == 'ready':
        actions.append('finalize_ml_session')
    return actions


def build_observation(*, session_id: str, session_state: str, selection_metric: str,
                      run_dir: Path, record: RunRecord, attempt: int,
                      effective_action: dict[str, Any], remaining_runs: int) -> dict[str, Any]:
    validation_status, metrics, validation_error = ('pending', {}, None)
    public_error: dict[str, Any] | None = None
    if record.state == 'succeeded':
        validation_status, metrics, validation_error = validation_from_manifest(
            run_dir, run_id=record.run_id
        )
        if validation_error:
            public_error = {
                'code': validation_error,
                'message': '验证结果完整性检查未通过',
                'retryable': False,
            }
    elif record.state in {'failed', 'cancelled'}:
        validation_status = 'failed' if record.state == 'failed' else 'unavailable'
        code = (
            str(record.error_details.get('code'))
            if isinstance(record.error_details, dict) and record.error_details.get('code')
            else f'agent_run_{record.state}'
        )
        public_error = {
            'code': code,
            'message': public_error_message(record.error or ('训练失败' if record.state == 'failed' else '训练已取消')),
            'retryable': False,
        }
    actions = allowed_actions(
        session_state=session_state, run_state=record.state,
        validation_status=validation_status, remaining_runs=remaining_runs,
    )
    response: dict[str, Any] = {
        'contract_version': AGENT_API_CONTRACT_VERSION,
        'observation_version': AGENT_OBSERVATION_VERSION,
        'session_id': session_id,
        'run_id': record.run_id,
        'attempt': attempt,
        'state': record.state,
        'effective_action': effective_action,
        'selection_metric': selection_metric,
        'progress': safe_progress(record),
        'validation': {'status': validation_status, 'metrics': metrics},
        'remaining_runs': remaining_runs,
        'allowed_actions': actions,
        'error': public_error,
        'extensions': {},
    }
    response['validation_score'] = metrics.get(selection_metric) if validation_status == 'ready' else None
    if record.state in {'queued', 'running'}:
        response['retry_after_seconds'] = 2
    return response

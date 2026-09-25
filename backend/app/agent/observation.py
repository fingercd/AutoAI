"""agent-observation-v1 的安全投影与 Manifest 校验。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..runs.artifacts import ArtifactIntegrityError, ManifestCorruptError, RunArtifactWriter
from ..runs.contracts import RunRecord, public_error_message
from .contracts import AGENT_API_CONTRACT_VERSION, AGENT_OBSERVATION_VERSION
from .metadata import run_metadata


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
    from ..runs.guard import inspect_artifacts, validation_metrics
    from ..runs.artifacts import ARTIFACT_CATALOG
    try:
        snapshot = inspect_artifacts(run_dir, run_id,
            required={k for k,v in ARTIFACT_CATALOG.items() if v.get('required')})
        return 'ready', validation_metrics(snapshot.documents.get('metrics.json')), None
    except Exception:
        return 'unavailable', {}, 'agent_manifest_invalid'


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
                      effective_action: dict[str, Any], remaining_runs: int, contract_version: str = AGENT_API_CONTRACT_VERSION, capability_snapshot: dict | None = None, assessment=None, guard_projection=False) -> dict[str, Any]:
    validation_status, metrics, validation_error = ('pending', {}, None)
    public_error: dict[str, Any] | None = None
    if record.state == 'succeeded':
        if assessment is not None:
            validation_status, metrics, validation_error = assessment.status, assessment.metrics, assessment.reason
        else:
            validation_status, metrics, validation_error = validation_from_manifest(run_dir, run_id=record.run_id)
        if validation_status == 'ready' and selection_metric not in metrics:
            validation_status, metrics, validation_error = 'unavailable', {}, 'agent_validation_unavailable'
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
        'contract_version': contract_version,
        'observation_version': 'agent-observation-v2' if contract_version == 'agent-session-v2' else AGENT_OBSERVATION_VERSION,
        'session_id': session_id,
        'run_id': record.run_id,
        'attempt': attempt,
        'state': record.state,
        'effective_action': effective_action,
        **run_metadata(record, version=contract_version, snapshot=capability_snapshot),
        'selection_metric': selection_metric,
        'progress': safe_progress(record),
        'validation': {'status': validation_status, 'metrics': metrics},
        'remaining_runs': remaining_runs,
        'allowed_actions': actions,
        'error': public_error,
        'extensions': {},
    }
    if guard_projection and assessment is not None and assessment.report is not None:
        from ..runs.guard import project_guard
        response['extensions']['guard'] = project_guard(assessment.report)
    if record.state == 'succeeded' and validation_status == 'ready':
        try:
            if assessment is not None:
                summary = assessment.documents.get('search_summary.json', {})
            else:
                summary_file = RunArtifactWriter(run_dir).resolve_download('search_summary.json')
                summary = json.loads(summary_file.read_text(encoding='utf-8'))
            if summary.get('schema_version') == 'search-summary-v1' and summary.get('integrity') == 'complete':
                response['extensions']['finite_search'] = {
                    key: summary[key] for key in ('mode','planned_trials','completed_trials',
                        'effective_search','selection_metric','selected','candidate_fit_count',
                        'final_refit_count','actual_epochs')
                }
        except (FileNotFoundError, ManifestCorruptError, ArtifactIntegrityError,
                OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError):
            pass
    if contract_version == 'agent-session-v2':
        from .metadata import resolved_execution
        response['resolved_execution'] = resolved_execution(run_dir, record, assessment=assessment)
        # Free progress/error prose is untrusted and never crosses the v2 boundary.
        response['progress'] = {'stage': record.state}
        for key in ('epoch', 'epochs'):
            value = record.progress.get(key)
            if type(value) is int and value >= 0:
                response['progress'][key] = value
        if response['error'] is not None:
            response['error'] = {'code': 'agent_run_' + record.state, 'message': 'Run result unavailable', 'retryable': False}
    response['validation_score'] = metrics.get(selection_metric) if validation_status == 'ready' else None
    if record.state in {'queued', 'running'}:
        response['retry_after_seconds'] = 2
    return response

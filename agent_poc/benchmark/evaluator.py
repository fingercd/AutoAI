"""Independent, one-shot reader for the finalized Run result contract.

Nothing returned by this module is sent back to the Agent.  In particular,
predictions, confusion matrices, analysis payloads, and artifact descriptors
are intentionally discarded after the two whitelisted scalar metrics have
been validated.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable


_RESULT_ENDPOINT = '/api/training/runs/{run_id}/result'
_RUN_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')


class BenchmarkEvaluationError(ValueError):
    """A public-code-only failure at the hidden evaluation boundary."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _finalized(agent_result: object) -> tuple[dict[str, Any], str]:
    if not isinstance(agent_result, dict):
        raise BenchmarkEvaluationError('agent_result_invalid')
    state = agent_result.get('status', agent_result.get('state'))
    if state != 'finalized':
        raise BenchmarkEvaluationError('agent_not_finalized')
    run_id = agent_result.get('selected_run_id')
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise BenchmarkEvaluationError('selected_run_id_invalid')
    return agent_result, run_id


def _response_object(response: object) -> dict[str, Any]:
    if isinstance(response, dict):
        payload = response
    else:
        raise_for_status = getattr(response, 'raise_for_status', None)
        if callable(raise_for_status):
            raise_for_status()
        json_method = getattr(response, 'json', None)
        if not callable(json_method):
            raise BenchmarkEvaluationError('result_response_invalid')
        payload = json_method()
    if not isinstance(payload, dict):
        raise BenchmarkEvaluationError('result_payload_invalid')
    return payload


def _unit_metric(value: object, *, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkEvaluationError(code)
    selected = float(value)
    if not math.isfinite(selected) or selected < 0.0 or selected > 1.0:
        raise BenchmarkEvaluationError(code)
    return selected


class ResultEvaluator:
    """Fetch exactly one finalized result and return only two safe scalars."""

    def __init__(
        self,
        http_client: object | None = None,
        *,
        request_fn: Callable[[str], object] | None = None,
    ) -> None:
        if (http_client is None) == (request_fn is None):
            raise ValueError('provide exactly one of http_client or request_fn')
        if request_fn is None:
            get = getattr(http_client, 'get', None)
            if not callable(get):
                raise ValueError('http_client must provide get(path)')
            request_fn = get
        self._request_fn = request_fn
        self.attempt_count = 0

    def evaluate(self, agent_result: object) -> dict[str, Any]:
        """Evaluate a finalized selection without retaining the full payload."""
        _, selected_run_id = _finalized(agent_result)
        endpoint = _RESULT_ENDPOINT.format(run_id=selected_run_id)
        self.attempt_count += 1
        payload = _response_object(self._request_fn(endpoint))

        if payload.get('schema_version') != 'run-result-v1':
            raise BenchmarkEvaluationError('result_schema_mismatch')
        run = payload.get('run')
        if not isinstance(run, dict) or run.get('run_id') != selected_run_id:
            raise BenchmarkEvaluationError('result_run_id_mismatch')
        if run.get('state') != 'succeeded' or run.get('result_state') != 'ready':
            raise BenchmarkEvaluationError('result_not_ready')

        evaluation = payload.get('evaluation')
        if not isinstance(evaluation, dict):
            raise BenchmarkEvaluationError('evaluation_contract_invalid')
        if (
            evaluation.get('strategy') != 'stratified_holdout'
            or evaluation.get('primary_split') != 'test'
            or evaluation.get('primary_aggregation') != 'direct'
        ):
            raise BenchmarkEvaluationError('evaluation_contract_invalid')

        metrics = payload.get('metrics')
        primary = metrics.get('primary') if isinstance(metrics, dict) else None
        if not isinstance(primary, dict):
            raise BenchmarkEvaluationError('primary_metrics_missing')
        macro_f1 = _unit_metric(
            primary.get('macro_f1'), code='macro_f1_invalid'
        )
        balanced_accuracy = _unit_metric(
            primary.get('balanced_accuracy'),
            code='balanced_accuracy_invalid',
        )
        return {
            'schema_version': 'benchmark-evaluation-v1',
            'run_id': selected_run_id,
            'macro_f1': macro_f1,
            'balanced_accuracy': balanced_accuracy,
        }

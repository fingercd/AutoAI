"""End-to-end benchmark runner with a one-way hidden evaluation boundary."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from ..orchestrator import run_agent
from ..trace import assert_trace_safe
from .adapter import adapt_pmlb_dataset
from .evaluator import BenchmarkEvaluationError, ResultEvaluator
from .manifest import verify_benchmark
from .report import build_trial_report


class BenchmarkRunnerError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _counter(value: object, name: str) -> int:
    selected = getattr(value, name, 0)
    if isinstance(selected, bool) or not isinstance(selected, int) or selected < 0:
        return 0
    return selected


def _delta(value: object, name: str, before: int) -> int:
    return max(0, _counter(value, name) - before)


def _safe_status(agent_result: object) -> str:
    if not isinstance(agent_result, dict):
        return 'invalid_agent_result'
    status = agent_result.get('status', agent_result.get('state'))
    if isinstance(status, str) and status in {
        'finalized', 'needs_human', 'budget_exhausted'
    }:
        return status
    return 'invalid_agent_result'


def _safe_metric_count(agent_result: object, field: str) -> int | None:
    metrics = agent_result.get('agent_metrics') if isinstance(agent_result, dict) else None
    value = metrics.get(field) if isinstance(metrics, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


class DatasetUploadClient:
    """Single-attempt benchmark CSV uploader with its own counter."""

    def __init__(self, http_client: object) -> None:
        post = getattr(http_client, 'post', None)
        if not callable(post):
            raise ValueError('http_client must provide post(path, files=...)')
        self._post = post
        self.attempt_count = 0

    def upload(self, csv_path: Path) -> str:
        csv_path = Path(csv_path)
        self.attempt_count += 1
        with csv_path.open('rb') as handle:
            response = self._post(
                '/api/datasets/upload',
                files={'file': (csv_path.name, handle, 'text/csv')},
            )
        if isinstance(response, dict):
            payload = response
        else:
            raise_for_status = getattr(response, 'raise_for_status', None)
            if callable(raise_for_status):
                raise_for_status()
            json_method = getattr(response, 'json', None)
            if not callable(json_method):
                raise BenchmarkRunnerError('upload_response_invalid')
            payload = json_method()
        dataset_id = payload.get('dataset_id') if isinstance(payload, dict) else None
        if not isinstance(dataset_id, str) or not dataset_id:
            raise BenchmarkRunnerError('upload_dataset_id_missing')
        return dataset_id


class BenchmarkRunner:
    def __init__(
        self,
        *,
        verify_fn: Callable[..., object] = verify_benchmark,
        adapt_fn: Callable[..., object] = adapt_pmlb_dataset,
        run_agent_fn: Callable[..., dict[str, Any]] = run_agent,
        trace_audit_fn: Callable[[Path], None] = assert_trace_safe,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._verify = verify_fn
        self._adapt = adapt_fn
        self._run_agent = run_agent_fn
        self._trace_audit = trace_audit_fn
        self._clock = clock

    def run_trial(
        self,
        *,
        benchmark_root: Path,
        policy_manifest_path: Path,
        dataset_name: str,
        adapter_output_dir: Path,
        cfg: object,
        llm: object,
        autoai: object,
        trace: object,
        uploader: DatasetUploadClient,
        evaluator: ResultEvaluator,
        model_key: str,
        model_revision: str,
        code_revision: str,
        seed: int,
        sleep_fn: Callable[[float], None] = time.sleep,
        budget_controller: object | None = None,
    ) -> dict[str, Any]:
        """Run one trial; evaluation is impossible before safe finalization."""
        started = self._clock()
        upload_before = uploader.attempt_count
        agent_before = _counter(autoai, 'attempt_count')
        retry_before = _counter(autoai, 'retry_attempt_count')
        evaluator_before = evaluator.attempt_count
        llm_before = _counter(llm, 'call_count')

        dataset = None
        status = 'failed'
        failure_code: str | None = 'runner_failed'
        evaluation: dict[str, Any] | None = None
        agent_result: dict[str, Any] | None = None
        try:
            verified = self._verify(Path(benchmark_root), Path(policy_manifest_path))
            dataset_method = getattr(verified, 'dataset', None)
            if not callable(dataset_method):
                raise BenchmarkRunnerError('verified_manifest_invalid')
            dataset = dataset_method(dataset_name)
            if getattr(dataset, 'allowed_split', None) != 'stratified_holdout':
                raise BenchmarkRunnerError('dataset_split_not_allowed')
            adapted = self._adapt(dataset, Path(adapter_output_dir))
            if getattr(adapted, 'split_strategy', None) != 'stratified_holdout':
                raise BenchmarkRunnerError('adapter_split_invalid')
            dataset_id = uploader.upload(Path(getattr(adapted, 'output_path')))

            trace_path = Path(getattr(trace, 'path'))
            if (
                getattr(cfg, 'code_revision', None) != code_revision
                or getattr(trace, 'code_revision', None) != code_revision
            ):
                raise BenchmarkRunnerError('code_revision_mismatch')
            locked_cfg = replace(
                cfg,
                dataset_id=dataset_id,
                source_role='benchmark',
                case_write=False,
                split_mode='stratified_holdout',
                split_train=8,
                split_valid=1,
                split_test=1,
                seed=seed,
                trace_path=trace_path,
            )
            run_kwargs: dict[str, Any] = {
                'trace': trace,
                'sleep_fn': sleep_fn,
            }
            if budget_controller is not None:
                run_kwargs['budget_controller'] = budget_controller
            agent_result = self._run_agent(
                locked_cfg, llm, autoai, **run_kwargs
            )

            # The JSONL audit is the last gate before any hidden evaluation
            # payload can be requested.  It also runs for non-final outcomes.
            self._trace_audit(trace_path)
            agent_status = _safe_status(agent_result)
            if agent_status != 'finalized':
                status = agent_status
                failure_code = f'agent_{agent_status}'
            else:
                evaluation = evaluator.evaluate(agent_result)
                status = 'completed'
                failure_code = None
        except BenchmarkEvaluationError as exc:
            status = 'failed'
            failure_code = f'evaluator_{exc.code}'
        except BenchmarkRunnerError as exc:
            status = 'failed'
            failure_code = exc.code
        except Exception:
            # The trial artifact intentionally records only a stable public
            # code; exception text may contain paths, credentials, or payloads.
            status = 'failed'
            failure_code = 'runner_exception'

        if dataset is None:
            dataset_role = 'unknown'
            dataset_sha256 = '0' * 64
        else:
            dataset_role = str(getattr(dataset, 'role'))
            dataset_sha256 = str(getattr(dataset, 'data_sha256'))
        reported_agent_api_attempts = _safe_metric_count(
            agent_result, 'api_call_count'
        )
        agent_api_attempts = (
            reported_agent_api_attempts
            if reported_agent_api_attempts is not None
            else _delta(autoai, 'attempt_count', agent_before)
        )
        reported_retry_attempts = _safe_metric_count(
            agent_result, 'retry_attempt_count'
        )
        retry_attempts = (
            reported_retry_attempts
            if reported_retry_attempts is not None
            else _delta(autoai, 'retry_attempt_count', retry_before)
        )
        reported_llm_calls = _safe_metric_count(agent_result, 'llm_call_count')
        llm_calls = (
            reported_llm_calls
            if reported_llm_calls is not None
            else _delta(llm, 'call_count', llm_before)
        )
        model_fit_count = _safe_metric_count(agent_result, 'model_fit_count')
        return build_trial_report(
            dataset_name=dataset_name,
            dataset_role=dataset_role,
            dataset_sha256=dataset_sha256,
            code_revision=code_revision,
            model_key=model_key,
            model_revision=model_revision,
            seed=seed,
            status=status,
            failure_code=failure_code,
            macro_f1=(evaluation or {}).get('macro_f1'),
            balanced_accuracy=(evaluation or {}).get('balanced_accuracy'),
            wall_clock_seconds=max(0.0, self._clock() - started),
            llm_call_count=llm_calls,
            upload_api_attempts=max(0, uploader.attempt_count - upload_before),
            agent_api_attempts=agent_api_attempts,
            evaluator_api_attempts=max(
                0, evaluator.attempt_count - evaluator_before
            ),
            retry_attempt_count=retry_attempts,
            model_fit_count=model_fit_count,
        )

"""Controlled Session → Experiment → Validation Feedback → Finalize loop."""

from __future__ import annotations

import time
from time import monotonic
from typing import Any, Callable

import httpx

from .clients.autoai_client import AutoAIClient
from .clients.llm_client import LLMClient
from .prompts import build_messages
from .schemas import (
    FinalizeDecision,
    PolicyViolation,
    RequestHumanDecision,
    RunExperimentDecision,
    decision_json_schema,
)
from .state import AgentConfig, LoopState, action_config_hash
from .trace import TraceRecorder


class AgentTimeout(TimeoutError):
    pass


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _needs_human(session_id: str, reason: str) -> dict[str, Any]:
    return {'status': 'needs_human', 'session_id': session_id, 'reason': reason}


def _seed_state_from_observation(state: LoopState, observation: dict[str, Any]) -> None:
    experiments = observation.get('experiments')
    if not isinstance(experiments, list):
        return
    for item in experiments:
        if not isinstance(item, dict):
            continue
        run_id = item.get('run_id')
        if not isinstance(run_id, str) or not run_id:
            continue
        if run_id not in state.run_ids:
            state.run_ids.append(run_id)
        if item.get('state') == 'succeeded' and item.get('validation_score') is not None:
            state.successful_run_ids.add(run_id)


def poll_until_terminal(
    autoai: AutoAIClient,
    *,
    session_id: str,
    run_id: str,
    interval_seconds: float,
    timeout_seconds: float,
    trace: TraceRecorder | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Poll feedback only; this function never calls the LLM."""
    deadline = monotonic() + timeout_seconds
    while True:
        started = _now_ms()
        feedback = autoai.feedback(session_id, run_id)
        latency_ms = _now_ms() - started
        if trace:
            trace.record_feedback(feedback, latency_ms=latency_ms)
        state = feedback.get('state')
        if state in {'succeeded', 'failed', 'cancelled'}:
            return feedback
        if state not in {'queued', 'running'}:
            raise PolicyViolation(f'unknown AutoAI feedback state: {state!r}')
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise AgentTimeout(f'run {run_id} exceeded {timeout_seconds} seconds')
        sleep_fn(min(max(0.0, interval_seconds), remaining))


def _reconcile_after_unknown_post(
    autoai: AutoAIClient,
    *,
    session_id: str,
    action: dict[str, Any],
) -> dict[str, Any] | None:
    """After a transport failure, inspect the Session before considering retry."""
    observation = autoai.get_session(session_id)
    for experiment in observation.get('experiments', []):
        if not isinstance(experiment, dict):
            continue
        effective = experiment.get('effective_action')
        if isinstance(effective, dict) and all(effective.get(k) == action.get(k) for k in (
            'model_type', 'normalization', 'class_balance', 'parent_run_id'
        )):
            return experiment
    return None


def run_agent(
    cfg: AgentConfig,
    llm: LLMClient,
    autoai: AutoAIClient,
    *,
    trace: TraceRecorder | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run one complete controlled Agent session."""
    health = autoai.health()
    worker = health.get('worker')
    if isinstance(worker, dict) and (
        worker.get('available') is False or worker.get('compatible') is False
    ):
        raise PolicyViolation('AutoAI worker is unavailable or contract-incompatible')

    session = autoai.create_session(cfg.session_payload())
    session_id = session.get('session_id')
    if not isinstance(session_id, str) or not session_id:
        raise PolicyViolation('AutoAI did not return a session_id')
    state = LoopState(max_runs=cfg.max_runs)

    while True:
        observation = autoai.get_session(session_id)
        _seed_state_from_observation(state, observation)
        started = _now_ms()
        if state.remaining_runs > 0:
            allowed_decisions = ('RUN_EXPERIMENT',)
            selected_run_ids: tuple[str, ...] = ()
        elif state.successful_run_ids:
            allowed_decisions = ('FINALIZE',)
            selected_run_ids = tuple(sorted(state.successful_run_ids))
        else:
            allowed_decisions = ('REQUEST_HUMAN',)
            selected_run_ids = ()
        messages = build_messages(
            cfg,
            observation,
            allowed_decisions=allowed_decisions,
            selected_run_ids=selected_run_ids,
        )
        decision_schema = decision_json_schema(
            allowed_decisions=allowed_decisions,
            allowed_models=cfg.allowed_models,
            selected_run_ids=selected_run_ids,
        )
        try:
            decision = llm.decide(messages, decision_schema=decision_schema)
        except TypeError as exc:
            # Keep small injected test doubles/source-compatible adapters
            # usable while the production LLMClient enforces the wire schema.
            if 'decision_schema' not in str(exc):
                raise
            decision = llm.decide(messages)
        latency_ms = _now_ms() - started
        if trace:
            trace.record_decision(decision, latency_ms=latency_ms)

        if isinstance(decision, RequestHumanDecision):
            if trace:
                trace.record('request_human', reason=decision.reason)
            return _needs_human(session_id, decision.reason)

        if isinstance(decision, FinalizeDecision):
            selected_run_id = decision.selected_run_id
            if selected_run_id not in state.successful_run_ids:
                # A model may copy a long opaque id imperfectly even though the
                # decision schema is valid. Only reconcile to the server's
                # already computed best_run_id, and only when Python has also
                # observed that run as successfully validated in this Session.
                authoritative_run_id = observation.get('best_run_id')
                if (
                    isinstance(authoritative_run_id, str)
                    and authoritative_run_id in state.successful_run_ids
                ):
                    if trace:
                        trace.record(
                            'finalize_reconciled',
                            requested_run_id=selected_run_id,
                            selected_run_id=authoritative_run_id,
                            reason='server_best_run_id_reconciled_opaque_id_copy',
                        )
                    selected_run_id = authoritative_run_id
                else:
                    reason = 'LLM selected a run without successful validation in this session'
                    if trace:
                        trace.record_error(reason)
                    return _needs_human(session_id, reason)
            started = _now_ms()
            result = autoai.finalize(session_id, selected_run_id)
            if trace:
                trace.record_finalize(result, latency_ms=_now_ms() - started)
            return result

        assert isinstance(decision, RunExperimentDecision)
        if state.remaining_runs <= 0:
            reason = 'Python budget guard rejected RUN_EXPERIMENT after max_runs'
            if trace:
                trace.record_error(reason)
            return _needs_human(session_id, reason)
        server_remaining = observation.get('remaining_runs')
        if isinstance(server_remaining, int) and server_remaining <= 0:
            reason = 'AutoAI session reports no remaining runs'
            if trace:
                trace.record_error(reason)
            return _needs_human(session_id, reason)
        if decision.model_type not in cfg.allowed_models:
            reason = f'model_type {decision.model_type!r} is outside locked allowed_models'
            if trace:
                trace.record_error(reason)
            return _needs_human(session_id, reason)

        action = {
            'model_type': decision.model_type,
            'normalization': decision.normalization,
            'class_balance': decision.class_balance,
            'parent_run_id': decision.parent_run_id,
            'rationale': decision.rationale,
        }
        config_hash = action_config_hash(action)
        if config_hash in state.submitted_config_hashes:
            reason = 'Python duplicate-config guard rejected RUN_EXPERIMENT'
            if trace:
                trace.record_error(reason)
            return _needs_human(session_id, reason)

        started = _now_ms()
        try:
            experiment = autoai.create_experiment(session_id, action)
        except httpx.TransportError:
            experiment = _reconcile_after_unknown_post(
                autoai, session_id=session_id, action=action
            )
            if experiment is None:
                raise
        state.submitted_config_hashes.add(config_hash)
        run_id = experiment.get('run_id') if isinstance(experiment, dict) else None
        if not isinstance(run_id, str) or not run_id:
            raise PolicyViolation('AutoAI experiment response did not contain run_id')
        state.run_ids.append(run_id)
        if trace:
            trace.record_experiment(experiment, latency_ms=_now_ms() - started)

        feedback = poll_until_terminal(
            autoai,
            session_id=session_id,
            run_id=run_id,
            interval_seconds=cfg.poll_interval_seconds,
            timeout_seconds=cfg.run_timeout_seconds,
            trace=trace,
            sleep_fn=sleep_fn,
        )
        validation = feedback.get('validation')
        if (
            feedback.get('state') == 'succeeded'
            and isinstance(validation, dict)
            and validation.get('status') == 'ready'
        ):
            state.successful_run_ids.add(run_id)
        # Failed/cancelled feedback is the only observation carried to the next
        # LLM call; polling itself never increases llm.call_count.

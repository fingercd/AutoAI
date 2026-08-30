"""Controlled Session → Experiment → Validation Feedback → Finalize loop."""

from __future__ import annotations

import time
from time import monotonic
from typing import Any, Callable

import httpx

from .budget import BudgetController, BudgetExceeded
from .clients.autoai_client import AutoAIClient
from .clients.llm_client import LLMClient
from .priors import load_prior_catalog, project_priors
from .prompts import build_messages
from .schemas import (
    FinalizeDecision,
    PolicyViolation,
    ReplanDecision,
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


def _proposal_recipes(observation: dict[str, Any]) -> tuple[dict[str, str], ...]:
    context = observation.get('context')
    catalog = context.get('proposal_catalog') if isinstance(context, dict) else None
    if not isinstance(catalog, dict) or catalog.get('version') != 'restricted-policy-v1':
        return ()
    proposals = catalog.get('proposals') if isinstance(catalog, dict) else None
    if not isinstance(proposals, list):
        return ()
    if catalog.get('proposal_count') != len(proposals):
        return ()
    result = []
    for item in proposals:
        if not isinstance(item, dict):
            continue
        keys = ('proposal_id', 'model_type', 'normalization', 'class_balance')
        if all(isinstance(item.get(key), str) for key in keys):
            result.append({key: item[key] for key in keys})
    proposal_ids = [item['proposal_id'] for item in result]
    if len(result) != len(proposals) or len(proposal_ids) != len(set(proposal_ids)):
        return ()
    return tuple(result)


def _training_key(value: dict[str, Any]) -> tuple[object, object, object]:
    return (
        value.get('model_type'),
        value.get('normalization', 'zscore'),
        value.get('class_balance', 'none'),
    )


def _unused_proposal_recipes(
    observation: dict[str, Any],
    proposals: tuple[dict[str, str], ...],
) -> tuple[dict[str, str], ...]:
    used_ids: set[str] = set()
    used_keys: set[tuple[object, object, object]] = set()
    experiments = observation.get('experiments')
    if isinstance(experiments, list):
        for experiment in experiments:
            if not isinstance(experiment, dict):
                continue
            action = experiment.get('effective_action')
            if not isinstance(action, dict):
                continue
            proposal_id = action.get('proposal_id')
            if isinstance(proposal_id, str):
                used_ids.add(proposal_id)
            used_keys.add(_training_key(action))
    return tuple(
        proposal
        for proposal in proposals
        if proposal['proposal_id'] not in used_ids
        and _training_key(proposal) not in used_keys
    )


def _failed_replan_context(
    observation: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    experiments = observation.get('experiments')
    if not isinstance(experiments, list):
        return (), ()
    replanned_parents = set()
    for item in experiments:
        if not isinstance(item, dict):
            continue
        effective_action = item.get('effective_action')
        parent_run_id = item.get('parent_run_id')
        if (
            isinstance(parent_run_id, str)
            and isinstance(effective_action, dict)
            and effective_action.get('action_id') == 'choose_unused_proposal'
        ):
            replanned_parents.add(parent_run_id)
    for experiment in reversed(experiments):
        if not isinstance(experiment, dict) or experiment.get('state') != 'failed':
            continue
        run_id = experiment.get('run_id')
        if run_id in replanned_parents:
            continue
        diagnosis = experiment.get('diagnosis')
        actions = (
            diagnosis.get('allowed_action_ids')
            if isinstance(diagnosis, dict)
            else None
        )
        if isinstance(run_id, str):
            if not isinstance(actions, list):
                actions = []
            if isinstance(experiment.get('parent_run_id'), str):
                actions = [
                    item for item in actions
                    if item != 'choose_unused_proposal'
                ]
            return (
                (run_id,),
                tuple(item for item in actions if isinstance(item, str)),
            )
    return (), ()


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
    budget_controller: BudgetController | None = None,
) -> dict[str, Any]:
    """Poll feedback only; this function never calls the LLM."""
    deadline = monotonic() + timeout_seconds
    while True:
        if budget_controller is not None:
            budget_controller.check_wall_clock()
        started = _now_ms()
        feedback = autoai.feedback(session_id, run_id)
        if budget_controller is not None:
            budget_controller.check_wall_clock()
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
        sleep_seconds = min(max(0.0, interval_seconds), remaining)
        if budget_controller is not None:
            sleep_seconds = budget_controller.bounded_sleep_seconds(sleep_seconds)
        sleep_fn(sleep_seconds)
        if budget_controller is not None:
            budget_controller.check_wall_clock()


def _reconcile_after_unknown_post(
    autoai: AutoAIClient,
    *,
    session_id: str,
    action: dict[str, Any],
    budget_controller: BudgetController | None = None,
) -> dict[str, Any] | None:
    """After a transport failure, inspect the Session before considering retry."""
    if budget_controller is not None:
        budget_controller.check_wall_clock()
    observation = autoai.get_session(session_id)
    if budget_controller is not None:
        budget_controller.observe_session(observation)
    for experiment in observation.get('experiments', []):
        if not isinstance(experiment, dict):
            continue
        effective = experiment.get('effective_action')
        if isinstance(effective, dict) and all(effective.get(k) == action.get(k) for k in (
            'model_type', 'normalization', 'class_balance', 'parent_run_id'
        )):
            return experiment
    return None


def _is_model_fit_budget_error(exc: httpx.HTTPStatusError) -> bool:
    """Recognize only the backend's explicit model-fit budget rejection."""
    if exc.response.status_code != 409:
        return False
    try:
        payload = exc.response.json()
    except ValueError:
        return False
    if not isinstance(payload, dict):
        return False
    if (
        payload.get('code') == 'agent_budget_exhausted'
        and payload.get('dimension') == 'max_model_fits'
    ):
        return True
    detail = payload.get('detail')
    if isinstance(detail, dict):
        return (
            detail.get('code') == 'agent_budget_exhausted'
            and detail.get('dimension') == 'max_model_fits'
        )
    return isinstance(detail, str) and detail.startswith(
        'model-fit budget exceeded:'
    )


def _run_agent_impl(
    cfg: AgentConfig,
    llm: LLMClient,
    autoai: AutoAIClient,
    *,
    trace: TraceRecorder | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    budget_controller: BudgetController,
) -> dict[str, Any]:
    """Run one complete controlled Agent session."""
    # The fixed catalog is loaded exactly once per controller invocation.  It
    # is immutable thereafter; only its safe projection is recomputed as the
    # server-owned proposal catalog is consumed.
    prior_catalog = load_prior_catalog() if cfg.modules.case_memory else None
    if trace and prior_catalog is not None:
        trace.record_prior_metadata(
            schema_version=prior_catalog.schema_version,
            digest=prior_catalog.digest,
        )
    budget_controller.check_wall_clock()
    health = autoai.health()
    worker = health.get('worker')
    if isinstance(worker, dict) and (
        worker.get('available') is False or worker.get('compatible') is False
    ):
        raise PolicyViolation('AutoAI worker is unavailable or contract-incompatible')

    budget_controller.check_wall_clock()
    session = autoai.create_session(cfg.session_payload())
    session_id = session.get('session_id')
    if not isinstance(session_id, str) or not session_id:
        raise PolicyViolation('AutoAI did not return a session_id')
    budget_controller.bind_session(session_id)
    state = LoopState(max_runs=cfg.max_runs)

    while True:
        budget_controller.check_wall_clock()
        observation = autoai.get_session(session_id)
        budget_controller.observe_session(observation)
        _seed_state_from_observation(state, observation)
        locked_proposal_recipes = _proposal_recipes(observation)
        locked = observation.get('locked_config')
        modules = locked.get('modules') if isinstance(locked, dict) else None
        restricted = (
            isinstance(modules, dict)
            and modules.get('restricted_strategy_pool') is True
        )
        limited_replanning = (
            isinstance(modules, dict)
            and modules.get('limited_replanning') is True
        )
        if restricted and not locked_proposal_recipes:
            return _needs_human(session_id, 'locked proposal catalog is unavailable')
        proposal_recipes = _unused_proposal_recipes(
            observation,
            locked_proposal_recipes,
        ) if restricted else ()
        prior_guidance = (
            project_priors(
                prior_catalog,
                observation=observation,
                allowed_models=cfg.allowed_models,
                proposal_recipes=proposal_recipes,
            )
            if prior_catalog is not None
            else None
        )
        failed_run_ids, failure_allowed_actions = _failed_replan_context(
            observation
        )
        allowed_replan_actions = (
            failure_allowed_actions if limited_replanning else ()
        )
        decision_support = observation.get('decision_support')
        stop_recommendation = (
            decision_support.get('stop_recommendation')
            if isinstance(decision_support, dict)
            else None
        )
        recommended_run_id = (
            decision_support.get('recommended_run_id')
            if isinstance(decision_support, dict)
            else None
        )
        force_finalize = (
            isinstance(stop_recommendation, dict)
            and stop_recommendation.get('action') == 'finalize'
            and isinstance(recommended_run_id, str)
            and recommended_run_id in state.successful_run_ids
            and (
                not failed_run_ids
                or 'stop' in failure_allowed_actions
            )
        )
        started = _now_ms()
        if force_finalize:
            allowed_decisions = ('FINALIZE',)
            selected_run_ids = (recommended_run_id,)
        elif failed_run_ids:
            can_replan = (
                state.remaining_runs > 0
                and proposal_recipes
                and 'choose_unused_proposal' in allowed_replan_actions
            )
            can_finalize = (
                bool(state.successful_run_ids)
                and 'stop' in failure_allowed_actions
            )
            if can_replan and can_finalize:
                allowed_decisions = ('REPLAN', 'FINALIZE')
                selected_run_ids = tuple(sorted(state.successful_run_ids))
            elif can_replan:
                allowed_decisions = ('REPLAN',)
                selected_run_ids = ()
            elif can_finalize:
                allowed_decisions = ('FINALIZE',)
                selected_run_ids = tuple(sorted(state.successful_run_ids))
            else:
                allowed_decisions = ('REQUEST_HUMAN',)
                selected_run_ids = ()
        elif state.remaining_runs > 0 and (not restricted or proposal_recipes):
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
            proposal_recipes=proposal_recipes,
            failed_run_ids=failed_run_ids,
            allowed_replan_actions=allowed_replan_actions,
            prior_guidance=prior_guidance,
        )
        decision_schema = decision_json_schema(
            allowed_decisions=allowed_decisions,
            allowed_models=cfg.allowed_models,
            selected_run_ids=selected_run_ids,
            proposal_recipes=proposal_recipes,
            failed_run_ids=failed_run_ids,
            allowed_replan_actions=allowed_replan_actions,
        )
        try:
            budget_controller.check_wall_clock()
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
            budget_controller.check_wall_clock()
            result = autoai.finalize(session_id, selected_run_id)
            if trace:
                trace.record_finalize(result, latency_ms=_now_ms() - started)
            return result

        assert isinstance(decision, (RunExperimentDecision, ReplanDecision))
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

        if isinstance(decision, ReplanDecision):
            if (
                decision.parent_run_id not in failed_run_ids
                or decision.action_id not in allowed_replan_actions
            ):
                reason = 'Python replanning guard rejected parent/action'
                if trace:
                    trace.record_error(reason)
                return _needs_human(session_id, reason)

        if not restricted and decision.proposal_id is not None:
            reason = 'Python proposal guard rejected proposal_id in unrestricted mode'
            if trace:
                trace.record_error(reason)
            return _needs_human(session_id, reason)

        if restricted:
            recipes_by_id = {
                item['proposal_id']: item for item in proposal_recipes
            }
            recipe = recipes_by_id.get(decision.proposal_id or '')
            if recipe is None or any(
                getattr(decision, field) != recipe[field]
                for field in ('model_type', 'normalization', 'class_balance')
            ):
                reason = 'Python proposal guard rejected non-canonical recipe'
                if trace:
                    trace.record_error(reason)
                return _needs_human(session_id, reason)

        action = {
            'model_type': decision.model_type,
            'normalization': decision.normalization,
            'class_balance': decision.class_balance,
            'parent_run_id': decision.parent_run_id,
            'proposal_id': decision.proposal_id,
            'rationale': decision.rationale,
        }
        if isinstance(decision, ReplanDecision):
            action['action_id'] = decision.action_id
        if decision.proposal_id is None:
            action.pop('proposal_id')
        config_hash = action_config_hash(action)
        if config_hash in state.submitted_config_hashes:
            reason = 'Python duplicate-config guard rejected RUN_EXPERIMENT'
            if trace:
                trace.record_error(reason)
            return _needs_human(session_id, reason)

        budget_controller.require_model_fit_capacity(observation)
        budget_controller.check_wall_clock()
        started = _now_ms()
        try:
            experiment = autoai.create_experiment(session_id, action)
        except httpx.HTTPStatusError as exc:
            if _is_model_fit_budget_error(exc) and budget_controller.limits is not None:
                usage = budget_controller.safe_usage()['model_fit_count']
                raise BudgetExceeded(
                    'max_model_fits',
                    limit=budget_controller.limits.max_model_fits,
                    used=usage if isinstance(usage, int) else 0,
                ) from None
            raise
        except httpx.TransportError:
            experiment = _reconcile_after_unknown_post(
                autoai,
                session_id=session_id,
                action=action,
                budget_controller=budget_controller,
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
            budget_controller=budget_controller,
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


def _agent_metrics(
    controller: BudgetController,
    *,
    llm: LLMClient,
    autoai: AutoAIClient,
) -> dict[str, int | float | None]:
    metrics = controller.safe_usage()
    llm_calls = getattr(llm, 'call_count', None)
    if isinstance(llm_calls, int) and not isinstance(llm_calls, bool):
        metrics['llm_call_count'] = max(metrics['llm_call_count'], llm_calls)
    api_calls = getattr(autoai, 'attempt_count', None)
    if isinstance(api_calls, int) and not isinstance(api_calls, bool):
        metrics['api_call_count'] = max(metrics['api_call_count'], api_calls)
    api_retries = getattr(autoai, 'retry_attempt_count', None)
    if isinstance(api_retries, int) and not isinstance(api_retries, bool):
        metrics['retry_attempt_count'] = max(
            metrics['retry_attempt_count'], api_retries
        )
    return metrics


def run_agent(
    cfg: AgentConfig,
    llm: LLMClient,
    autoai: AutoAIClient,
    *,
    trace: TraceRecorder | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    budget_controller: BudgetController | None = None,
) -> dict[str, Any]:
    """Run one Session and always return safe process-side usage metrics."""
    limits = cfg.budget if cfg.modules.budget_control else None
    if cfg.modules.budget_control and limits is None:
        raise ValueError('budget_control requires an explicit budget')
    if not cfg.modules.budget_control and cfg.budget is not None:
        raise ValueError('budget requires budget_control=true')
    controller = budget_controller or BudgetController(limits)
    if cfg.modules.budget_control and controller.limits != limits:
        raise ValueError('budget controller limits do not match locked config')
    if not cfg.modules.budget_control and controller.limits is not None:
        raise ValueError('budget controller limits require budget_control=true')

    set_autoai_controller = getattr(autoai, 'set_budget_controller', None)
    if callable(set_autoai_controller):
        set_autoai_controller(controller)
    set_llm_controller = getattr(llm, 'set_budget_controller', None)
    if callable(set_llm_controller):
        set_llm_controller(controller)

    try:
        result = _run_agent_impl(
            cfg,
            llm,
            autoai,
            trace=trace,
            sleep_fn=sleep_fn,
            budget_controller=controller,
        )
        controller.check_wall_clock()
    except BudgetExceeded as exc:
        result = {
            'status': 'budget_exhausted',
            'dimension': exc.dimension,
        }
        if controller.session_id is not None:
            result['session_id'] = controller.session_id

    metrics = _agent_metrics(controller, llm=llm, autoai=autoai)
    result = {**result, 'agent_metrics': metrics}
    if trace is not None:
        trace.record_budget_usage(
            status=str(result.get('status', 'unknown')),
            dimension=(
                str(result['dimension'])
                if isinstance(result.get('dimension'), str)
                else None
            ),
            usage=metrics,
        )
    return result

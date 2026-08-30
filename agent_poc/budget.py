"""Process-local hard budget accounting for the Agent POC.

The backend owns model-fit reservations.  This controller owns the four
dimensions that can only be measured at the POC process boundary: real HTTP
attempts, real LLM completions, retries (including structured-output repair),
and monotonic wall-clock time.
"""

from __future__ import annotations

import math
from threading import RLock
from time import monotonic
from typing import Any, Callable

from .state import AgentBudgetConfig


_COUNT_DIMENSIONS = {
    'api_calls': 'max_api_calls',
    'llm_calls': 'max_llm_calls',
    'retry_attempts': 'max_retry_attempts',
}


class BudgetExceeded(RuntimeError):
    """Raised before an operation that would exceed a locked budget."""

    def __init__(self, dimension: str, *, limit: int | float, used: int | float) -> None:
        self.dimension = dimension
        self.limit = limit
        self.used = used
        super().__init__(f'agent budget exhausted: {dimension}')


class BudgetController:
    """Atomic, monotonic accounting shared by HTTP, LLM, and the loop.

    ``limits=None`` is a compatibility/tracking mode: counters are still
    collected for ``agent_metrics`` but no new limit is enforced.
    """

    def __init__(
        self,
        limits: AgentBudgetConfig | None = None,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.limits = limits
        self._clock = clock
        self._started_at = float(clock())
        self._lock = RLock()
        self._counts = {
            'api_calls': 0,
            'llm_calls': 0,
            'retry_attempts': 0,
        }
        self._model_fit_count: int | None = None
        self.session_id: str | None = None

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, float(self._clock()) - self._started_at)

    def bind_session(self, session_id: str) -> None:
        if isinstance(session_id, str) and session_id:
            with self._lock:
                self.session_id = session_id

    def check_wall_clock(self) -> None:
        with self._lock:
            if self.limits is None:
                return
            elapsed = self.elapsed_seconds
            if elapsed >= self.limits.max_wall_clock_seconds:
                raise BudgetExceeded(
                    'max_wall_clock_seconds',
                    limit=self.limits.max_wall_clock_seconds,
                    used=elapsed,
                )

    def reserve(
        self,
        *,
        api_calls: int = 0,
        llm_calls: int = 0,
        retry_attempts: int = 0,
    ) -> None:
        """Check all requested counters, then debit them as one operation."""
        requested = {
            'api_calls': api_calls,
            'llm_calls': llm_calls,
            'retry_attempts': retry_attempts,
        }
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in requested.values()):
            raise ValueError('budget reservation counts must be non-negative integers')
        with self._lock:
            self.check_wall_clock()
            if self.limits is not None:
                for usage_key, amount in requested.items():
                    limit_key = _COUNT_DIMENSIONS[usage_key]
                    limit = getattr(self.limits, limit_key)
                    projected = self._counts[usage_key] + amount
                    if projected > limit:
                        raise BudgetExceeded(
                            limit_key,
                            limit=limit,
                            used=self._counts[usage_key],
                        )
            for usage_key, amount in requested.items():
                self._counts[usage_key] += amount

    def before_api_attempt(self, *, retry: bool = False) -> None:
        self.reserve(api_calls=1, retry_attempts=1 if retry else 0)

    def before_llm_completion(self, *, retry: bool = False) -> None:
        self.reserve(llm_calls=1, retry_attempts=1 if retry else 0)

    def bounded_sleep_seconds(self, requested_seconds: float) -> float:
        requested = float(requested_seconds)
        if not math.isfinite(requested) or requested < 0:
            raise ValueError('sleep duration must be a finite non-negative number')
        with self._lock:
            self.check_wall_clock()
            if self.limits is None:
                return requested
            remaining = self.limits.max_wall_clock_seconds - self.elapsed_seconds
            if remaining <= 0:
                self.check_wall_clock()
            return min(requested, max(0.0, remaining))

    def observe_session(self, payload: dict[str, Any]) -> None:
        """Project only the backend's safe aggregate model-fit counter."""
        with self._lock:
            if self.limits is None:
                # Disabled mode preserves the legacy metric projection without
                # turning the new budget schema into a compatibility requirement.
                experiments = payload.get('experiments')
                if not isinstance(experiments, list) or not experiments:
                    return
                counts: list[int] = []
                for item in experiments:
                    count = (
                        item.get('model_fit_count')
                        if isinstance(item, dict)
                        else None
                    )
                    if (
                        isinstance(count, bool)
                        or not isinstance(count, int)
                        or count < 0
                    ):
                        return
                    counts.append(count)
                self._model_fit_count = max(
                    self._model_fit_count or 0,
                    sum(counts),
                )
                return
            self._validate_and_store_model_fit_usage_locked(payload)

    def _model_fit_contract_failure_locked(self) -> None:
        assert self.limits is not None
        raise BudgetExceeded(
            'max_model_fits',
            limit=self.limits.max_model_fits,
            used=self.limits.max_model_fits,
        )

    def _validate_and_store_model_fit_usage_locked(
        self,
        payload: dict[str, Any],
    ) -> dict[str, int]:
        """Validate one complete backend snapshot and store monotonic actual."""
        assert self.limits is not None
        budget_usage = payload.get('budget_usage')
        if (
            not isinstance(budget_usage, dict)
            or budget_usage.get('schema_version') != 'agent-budget-usage-v1'
        ):
            self._model_fit_contract_failure_locked()
        model_fits = budget_usage.get('model_fits')
        expected_fields = {
            'limit', 'actual', 'reserved', 'charged', 'remaining',
        }
        if not isinstance(model_fits, dict) or set(model_fits) != expected_fields:
            self._model_fit_contract_failure_locked()
        values: dict[str, int] = {}
        for field in expected_fields:
            value = model_fits.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                self._model_fit_contract_failure_locked()
            values[field] = value
        if values['limit'] != self.limits.max_model_fits:
            self._model_fit_contract_failure_locked()
        if values['charged'] != values['actual'] + values['reserved']:
            self._model_fit_contract_failure_locked()
        if values['remaining'] != values['limit'] - values['charged']:
            self._model_fit_contract_failure_locked()
        if (
            self._model_fit_count is not None
            and values['actual'] < self._model_fit_count
        ):
            self._model_fit_contract_failure_locked()
        self._model_fit_count = values['actual']
        return values

    def require_model_fit_capacity(self, payload: dict[str, Any]) -> None:
        """Fail locally when the backend reports no uncharged fit capacity."""
        with self._lock:
            if self.limits is None:
                return
            values = self._validate_and_store_model_fit_usage_locked(payload)
            if values['remaining'] == 0:
                raise BudgetExceeded(
                    'max_model_fits',
                    limit=self.limits.max_model_fits,
                    used=values['charged'],
                )

    def safe_usage(self) -> dict[str, int | float | None]:
        with self._lock:
            return {
                'api_call_count': self._counts['api_calls'],
                'llm_call_count': self._counts['llm_calls'],
                'retry_attempt_count': self._counts['retry_attempts'],
                'wall_clock_seconds': round(self.elapsed_seconds, 6),
                'model_fit_count': self._model_fit_count,
            }

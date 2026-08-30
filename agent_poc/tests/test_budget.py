from __future__ import annotations

import pytest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from agent_poc.budget import BudgetController, BudgetExceeded
from agent_poc.state import AgentBudgetConfig


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _limits(**overrides) -> AgentBudgetConfig:
    values = {
        'max_model_fits': 20,
        'max_llm_calls': 3,
        'max_api_calls': 10,
        'max_wall_clock_seconds': 30,
        'max_retry_attempts': 2,
    }
    values.update(overrides)
    return AgentBudgetConfig(**values)


def _usage(
    *,
    limit: int = 7,
    actual: int = 4,
    reserved: int = 0,
) -> dict:
    charged = actual + reserved
    return {
        'budget_usage': {
            'schema_version': 'agent-budget-usage-v1',
            'model_fits': {
                'limit': limit,
                'actual': actual,
                'reserved': reserved,
                'charged': charged,
                'remaining': limit - charged,
            },
        }
    }


def test_reservation_is_atomic_and_happens_before_attempt():
    controller = BudgetController(_limits(max_retry_attempts=0))
    controller.before_api_attempt()
    with pytest.raises(BudgetExceeded) as result:
        controller.before_api_attempt(retry=True)
    assert result.value.dimension == 'max_retry_attempts'
    assert controller.safe_usage()['api_call_count'] == 1
    assert controller.safe_usage()['retry_attempt_count'] == 0


def test_wall_clock_uses_monotonic_clock_and_bounds_sleep():
    clock = FakeClock()
    controller = BudgetController(
        _limits(max_wall_clock_seconds=5),
        clock=clock,
    )
    clock.advance(3)
    assert controller.bounded_sleep_seconds(10) == 2
    clock.advance(2)
    with pytest.raises(BudgetExceeded) as result:
        controller.check_wall_clock()
    assert result.value.dimension == 'max_wall_clock_seconds'


def test_model_fit_usage_reads_backend_safe_aggregate_only():
    controller = BudgetController(_limits(max_model_fits=7))
    session = _usage()
    controller.observe_session(session)
    assert controller.safe_usage()['model_fit_count'] == 4
    rollback = _usage(actual=0)
    with pytest.raises(BudgetExceeded):
        controller.observe_session(rollback)
    assert controller.safe_usage()['model_fit_count'] == 4
    controller.require_model_fit_capacity(session)

    exhausted = _usage(actual=7)
    with pytest.raises(BudgetExceeded) as result:
        controller.require_model_fit_capacity(exhausted)
    assert result.value.dimension == 'max_model_fits'


@pytest.mark.parametrize(
    'session',
    [
        {},
        {'budget_usage': {}},
        {'budget_usage': {'model_fits': {}}},
        {'budget_usage': {'model_fits': {'remaining': True}}},
        {'budget_usage': {'model_fits': {'remaining': -1}}},
        {'budget_usage': {'model_fits': {'remaining': 8}}},
    ],
)
def test_model_fit_capacity_fails_closed_on_contract_drift(session):
    controller = BudgetController(_limits(max_model_fits=7))
    with pytest.raises(BudgetExceeded) as result:
        controller.require_model_fit_capacity(session)
    assert result.value.dimension == 'max_model_fits'


@pytest.mark.parametrize('method_name', ['observe_session', 'require_model_fit_capacity'])
def test_model_fit_contract_rejects_only_remaining_missing(method_name):
    payload = _usage()
    payload['budget_usage']['model_fits'].pop('remaining')
    controller = BudgetController(_limits(max_model_fits=7))
    with pytest.raises(BudgetExceeded):
        getattr(controller, method_name)(payload)


@pytest.mark.parametrize('method_name', ['observe_session', 'require_model_fit_capacity'])
@pytest.mark.parametrize('drift', ['schema', 'limit', 'charged', 'remaining'])
def test_model_fit_contract_rejects_schema_limit_and_identity_drift(
    method_name,
    drift,
):
    payload = _usage()
    if drift == 'schema':
        payload['budget_usage']['schema_version'] = 'unknown'
    elif drift == 'limit':
        payload = _usage(limit=8)
    elif drift == 'charged':
        payload['budget_usage']['model_fits']['charged'] = 5
        payload['budget_usage']['model_fits']['remaining'] = 2
    else:
        payload['budget_usage']['model_fits']['remaining'] = 2
    controller = BudgetController(_limits(max_model_fits=7))
    with pytest.raises(BudgetExceeded):
        getattr(controller, method_name)(payload)


@pytest.mark.parametrize('method_name', ['observe_session', 'require_model_fit_capacity'])
def test_model_fit_contract_rejects_actual_rollback(method_name):
    controller = BudgetController(_limits(max_model_fits=7))
    controller.observe_session(_usage(actual=4))
    with pytest.raises(BudgetExceeded):
        getattr(controller, method_name)(_usage(actual=3))
    assert controller.safe_usage()['model_fit_count'] == 4


@pytest.mark.parametrize(
    'field',
    ['limit', 'actual', 'reserved', 'charged', 'remaining'],
)
def test_model_fit_contract_rejects_boolean_counter(field):
    payload = _usage()
    payload['budget_usage']['model_fits'][field] = True
    controller = BudgetController(_limits(max_model_fits=7))
    with pytest.raises(BudgetExceeded):
        controller.observe_session(payload)


def test_concurrent_reservations_allow_only_one_at_the_boundary():
    controller = BudgetController(_limits(max_api_calls=1))
    barrier = Barrier(2)

    def attempt() -> str:
        barrier.wait()
        try:
            controller.before_api_attempt()
            return 'accepted'
        except BudgetExceeded:
            return 'rejected'

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _index: attempt(), range(2)))
    assert sorted(outcomes) == ['accepted', 'rejected']
    assert controller.safe_usage()['api_call_count'] == 1

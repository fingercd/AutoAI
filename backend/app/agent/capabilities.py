"""服务端权威的 Agent 能力目录。"""

from __future__ import annotations

from typing import Any

from .contracts import AGENT_ALLOWED_MODELS, AGENT_API_CONTRACT_VERSION


_PLANNED_MODULES = (
    'evidence_card', 'dynamic_preprocessing', 'restricted_strategy_pool',
    'bounded_hpo', 'fail_fast_guard', 'feedback_diagnosis', 'limited_replanning',
    'uncertainty_selection', 'case_memory', 'budget_control',
    'constrained_code_evolution',
)


def module_catalog() -> dict[str, dict[str, Any]]:
    return {
        name: {
            'available': False,
            'status': 'unavailable',
            'schema_version': None,
            'reason': '基础接口适配层尚未实现该研究模块',
        }
        for name in _PLANNED_MODULES
    }


def health_payload(*, worker_available: bool, worker_compatible: bool) -> dict[str, Any]:
    status = 'ready'
    if worker_available and not worker_compatible:
        status = 'unavailable'
    elif not worker_available:
        status = 'degraded'
    return {
        'contract_version': AGENT_API_CONTRACT_VERSION,
        'status': status,
        'capabilities': {
            'create_session': True,
            'create_experiment': worker_compatible or not worker_available,
            'read_session': True,
            'read_feedback': True,
            'finalize_session': True,
        },
        'models': list(AGENT_ALLOWED_MODELS),
        'modules': module_catalog(),
    }

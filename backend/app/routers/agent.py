"""Agent Session HTTP 路由（POST /api/agent/sessions、/experiments、/finalize 等）。

约束：
- 不复用 ``/api/training/runs`` 路由，但**复用其底下的 RunRepository** 创建
  Training Run——这样 Session 内的每个实验都对应一个真实 Training Run，状态
  走的是现有的 queued → running → succeeded 状态机。
- 不暴露任何 Test、artifact、explainability 字段。``service._scrub`` 是最后一
  道防线；路由层也绝不引用 ``/api/training/runs/{id}/result`` 的契约对象。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from fastapi import APIRouter, Body, Depends, HTTPException, Path

from ..contracts import TrainingConfigValidationError
from ..datasets.repository import DatasetRepository
from ..http.principal import Principal, get_principal
from ..paths import AGENT_DATABASE, DATASETS_DATABASE, RUNS_DATABASE, RUNS_DIR, STORAGE_DIR
from ..runs.repository import RunRepository
from ..runs.status_projection import project_status
from ..runs.submission import RunSubmissionService
from ..version import WORKER_CONTRACT_VERSION
from ..agent.capabilities import health_payload
from ..agent.contracts import (
    AgentDomainError,
    CreateAgentExperimentRequest,
    CreateAgentSessionRequest,
    FinalizeAgentSessionRequest,
    ReconcileAgentSessionRequest,
)
from ..agent.repository import (
    AgentConfigCollision,
    AgentExperimentNotFound,
    AgentSessionClosed,
    AgentSessionNotFound,
    AgentSessionRepository,
)
from ..agent.service import AgentService
from ..agent.reconciliation import AgentReconciliationService


router = APIRouter()


def _agent_service(contract_version: str = 'agent-session-v1') -> AgentService:
    """Legacy dependency seam, delegated to the shared factory."""
    import sys
    from .deps import get_agent_service
    return get_agent_service(contract_version, storage=sys.modules[__name__])


def _reconciliation_service() -> AgentReconciliationService:
    try:
        sessions = AgentSessionRepository(AGENT_DATABASE)
        sessions.initialize()
        runs = RunRepository(RUNS_DATABASE)
        runs.initialize()
    except (sqlite3.Error, OSError) as exc:
        raise AgentDomainError(
            'agent_reconciliation_unavailable', '对账存储暂时不可用',
            status_code=503, retryable=True,
            allowed_actions=('inspect_ml_session',),
        ) from exc
    return AgentReconciliationService(
        session_repository=sessions,
        run_repository=runs,
    )


def _to_http_exception(exc: Exception) -> HTTPException:
    """把领域异常映射到 HTTP 状态码，避免在每个路由重复 try/except。"""
    if isinstance(exc, AgentDomainError):
        return HTTPException(status_code=exc.status_code, detail=exc.detail())
    if isinstance(exc, AgentSessionNotFound):
        return HTTPException(status_code=404, detail='agent session 不存在')
    if isinstance(exc, AgentExperimentNotFound):
        return HTTPException(status_code=404, detail='experiment 不存在')
    if isinstance(exc, AgentSessionClosed):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, AgentConfigCollision):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, TrainingConfigValidationError):
        return HTTPException(status_code=422, detail=str(exc))
    # ``payload.validate()`` 抛出 ``ValueError``（Pydantic 风格的业务校验）：
    # 全部映射成 422 让客户端可以与 schema 错误区分对待。
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=500, detail='agent session 内部错误')


@router.get('/api/agent/health')
def agent_health() -> dict[str, object]:
    runs = RunRepository(RUNS_DATABASE)
    runs.initialize()
    health = runs.worker_health(
        now=datetime.now(timezone.utc),
        expected_contract_version=WORKER_CONTRACT_VERSION,
    )
    return health_payload(
        worker_available=bool(health.get('available')),
        worker_compatible=bool(health.get('compatible')),
    )


@router.post('/api/agent/sessions', status_code=201)
def create_session(
    payload: CreateAgentSessionRequest,
    principal: Principal = Depends(get_principal),
) -> dict[str, object]:
    """创建 Agent Session。dataset_id 必须真实存在；locked_config 是 Session
    创建后冻结的策略集合，单次实验不能覆盖其中任何键。"""
    service = _agent_service()
    try:
        return service.create_session(payload, principal=principal)
    except (AgentDomainError, TrainingConfigValidationError, ValueError) as exc:
        # Pydantic 已经在更外层捕获了 extra='forbid'，这里只覆盖业务校验。
        raise _to_http_exception(exc) from exc


@router.post('/api/agent/sessions/{session_id}/experiments', status_code=202)
def create_experiment(
    payload: CreateAgentExperimentRequest,
    session_id: str = Path(..., min_length=1),
    principal: Principal = Depends(get_principal),
) -> dict[str, object]:
    """在 Session 内提交一次实验。

    通过 ``RunRepository.create_queued`` 入队一个真实 Training Run，让现有
    worker 在自己的状态机里执行。Agent 自身不复制任何训练逻辑。
    """
    service = _agent_service()
    try:
        return service.create_experiment(
            session_id=session_id, payload=payload, principal=principal
        )
    except (AgentDomainError, TrainingConfigValidationError) as exc:
        raise _to_http_exception(exc) from exc


@router.get('/api/agent/sessions/{session_id}/experiments/{run_id}/feedback')
def get_feedback(
    session_id: str = Path(..., min_length=1),
    run_id: str = Path(..., min_length=1),
    principal: Principal = Depends(get_principal),
) -> dict[str, object]:
    """返回安全版训练反馈：queued/running 只给 state+progress；succeeded 只
    给 validation 标量；failed 给出脱敏 error；其余字段（test、artifact、
    explainability）一律被 ``_scrub`` 过滤掉。"""
    service = _agent_service()
    try:
        return service.get_feedback(
            session_id=session_id, run_id=run_id, principal=principal
        )
    except (AgentDomainError, TrainingConfigValidationError) as exc:
        raise _to_http_exception(exc) from exc


@router.get('/api/agent/sessions/{session_id}')
def get_session(
    session_id: str = Path(..., min_length=1),
    principal: Principal = Depends(get_principal),
) -> dict[str, object]:
    """Session 详情 + 所有实验的安全摘要。"""
    service = _agent_service()
    try:
        return service.get_session(session_id=session_id, principal=principal)
    except (AgentDomainError, TrainingConfigValidationError) as exc:
        raise _to_http_exception(exc) from exc


@router.post('/api/agent/sessions/{session_id}/reconcile')
def reconcile_session(
    session_id: str = Path(..., min_length=1),
    payload: ReconcileAgentSessionRequest = Body(
        default_factory=ReconcileAgentSessionRequest
    ),
    principal: Principal = Depends(get_principal),
) -> dict[str, object]:
    """显式执行 Principal-scoped 对账；HTTP 不接受阈值、ID 或目标状态。"""
    del payload
    try:
        _agent_service()._session(session_id, principal)
        service = _reconciliation_service()
        return service.reconcile_session(session_id=session_id, principal=principal)
    except AgentDomainError as exc:
        raise _to_http_exception(exc) from exc


@router.post('/api/agent/sessions/{session_id}/finalize')
def finalize_session(
    payload: FinalizeAgentSessionRequest,
    session_id: str = Path(..., min_length=1),
    principal: Principal = Depends(get_principal),
) -> dict[str, object]:
    """Finalize Session。

    - 选中的 run_id 必须属于当前 Session
    - run 必须训练成功
    - finalize 后拒绝继续提交实验（service 层校验 session.state）
    - 重复 finalize 幂等：相同 selected_run_id 返回旧记录，不同 selected_run_id 报错
    """
    service = _agent_service()
    try:
        return service.finalize_session(
            session_id=session_id,
            selected_run_id=payload.selected_run_id,
            principal=principal,
        )
    except (AgentDomainError, TrainingConfigValidationError) as exc:
        raise _to_http_exception(exc) from exc


# Negotiated current URLs share the service and legacy HTTP error boundary.
from fastapi import Header
from .deps import get_agent_service
from ..agent.contracts import V2, CreateAgentSessionRequestV2, CreateAgentExperimentRequestV2, CreateRecipeExperimentRequest, CreateKnowledgeExperimentRequest, CreateStructuredExperimentRequest
from ..model_config import model_capability_snapshot

current_router = APIRouter(prefix='/api/agent/v2')

def revision_header(x_autoai_agent_revision: str | None = Header(None)):
    return x_autoai_agent_revision


def require_negotiation(service, principal, revision, session_id=None, payload=None):
    profile = (service._session(session_id, principal).frozen_preparation if session_id else
               getattr(payload, 'execution_profile', None))
    expected = (profile['protocol_revision'] if session_id and profile else getattr(payload, 'protocol_revision', None))
    if profile and revision != expected:
        raise AgentDomainError('agent_version_incompatible', 'Recipe protocol negotiation required', status_code=409)


def call(method, principal, revision=None, **kwargs):
    try:
        service = get_agent_service(V2)
        require_negotiation(service, principal, revision, kwargs.get('session_id'), kwargs.get('payload'))
        return getattr(service, method)(principal=principal, **kwargs)
    except (AgentDomainError, TrainingConfigValidationError, ValueError) as exc:
        raise _to_http_exception(exc) from exc


@current_router.get('/health')
def current_health(revision: str | None = Depends(revision_header)):
    result = agent_health()
    snapshot = model_capability_snapshot()
    result.update(contract_version=V2, **snapshot)
    result['capabilities']['create_experiment'] &= any(m['available'] for m in snapshot['models'])
    if not any(m['available'] for m in snapshot['models']):
        result['status'] = 'unavailable'
    if revision in ('agent-recipes-revision-v1','agent-recipes-revision-v2'):
        result.update(protocol_revision=revision,execution_profiles=['train-evidence-recipes-v1'])
    if revision == 'agent-recipes-revision-v2':
        from ..agent.capabilities import module_catalog
        result['modules'] = module_catalog(revision)
    return result


@current_router.post('/sessions', status_code=201)
def current_create_session(payload: CreateAgentSessionRequestV2, principal: Principal = Depends(get_principal), revision: str | None = Depends(revision_header)):
    return call('create_session', principal, revision=revision, payload=payload)


@current_router.get('/sessions/{session_id}')
def current_get_session(session_id: str, principal: Principal = Depends(get_principal), revision: str | None = Depends(revision_header)):
    return call('get_session', principal, revision=revision, session_id=session_id)


@current_router.post('/sessions/{session_id}/experiments', status_code=202)
def current_create_experiment(session_id: str, payload: CreateStructuredExperimentRequest | CreateKnowledgeExperimentRequest | CreateAgentExperimentRequestV2 | CreateRecipeExperimentRequest, principal: Principal = Depends(get_principal), revision: str | None = Depends(revision_header)):
    return call('create_experiment', principal, revision=revision, session_id=session_id, payload=payload)


@current_router.get('/sessions/{session_id}/experiments/{run_id}/feedback')
def current_feedback(session_id: str, run_id: str, principal: Principal = Depends(get_principal), revision: str | None = Depends(revision_header)):
    return call('get_feedback', principal, revision=revision, session_id=session_id, run_id=run_id)


@current_router.post('/sessions/{session_id}/finalize')
def current_finalize(session_id: str, payload: FinalizeAgentSessionRequest, principal: Principal = Depends(get_principal), revision: str | None = Depends(revision_header)):
    return call('finalize_session', principal, revision=revision, session_id=session_id, selected_run_id=payload.selected_run_id)


@current_router.post('/sessions/{session_id}/reconcile')
def current_reconcile(session_id: str, payload: ReconcileAgentSessionRequest, principal: Principal = Depends(get_principal), revision: str | None = Depends(revision_header)):
    try:
        require_negotiation(get_agent_service(V2), principal, revision, session_id)
        result = _reconciliation_service().reconcile_session(session_id=session_id, principal=principal)
        result['contract_version'] = V2
        return result
    except AgentDomainError as exc:
        raise _to_http_exception(exc) from exc

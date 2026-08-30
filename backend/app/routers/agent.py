"""Agent Session HTTP 路由（POST /api/agent/sessions、/experiments、/finalize 等）。

约束：
- 不复用 ``/api/training/runs`` 路由，但**复用其底下的 RunRepository** 创建
  Training Run——这样 Session 内的每个实验都对应一个真实 Training Run，状态
  走的是现有的 queued → running → succeeded 状态机。
- 不暴露任何 Test、artifact、explainability 字段。``service._scrub`` 是最后一
  道防线；路由层也绝不引用 ``/api/training/runs/{id}/result`` 的契约对象。
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from ..contracts import TrainingConfigValidationError
from ..datasets.repository import DatasetRepository
from ..http.principal import Principal, get_principal
from ..paths import AGENT_DATABASE, DATASETS_DATABASE, RUNS_DATABASE, STORAGE_DIR
from ..runs.repository import RunRepository
from ..agent.contracts import (
    AGENT_API_CAPABILITIES,
    AGENT_API_CONTRACT_VERSION,
    CreateAgentExperimentRequest,
    CreateAgentSessionRequest,
    FinalizeAgentSessionRequest,
)
from ..agent.repository import (
    AgentConfigCollision,
    AgentExperimentNotFound,
    AgentSessionClosed,
    AgentSessionNotFound,
    AgentSessionRepository,
)
from ..agent.service import AgentService
from ..agent.runtime_health import (
    AgentRegistryUnavailable,
    probe_runtime_health,
)


router = APIRouter()


def _agent_service() -> AgentService:
    """每次请求新建一个轻量服务实例，Repository 是 SQLite 句柄集合可以共享。

    与现有 ``get_run_repository`` 风格一致；Repository 本身不带跨请求状态。
    """
    sessions = AgentSessionRepository(AGENT_DATABASE)
    sessions.initialize()
    runs = RunRepository(RUNS_DATABASE)
    runs.initialize()
    datasets = DatasetRepository(DATASETS_DATABASE, storage_root=STORAGE_DIR)
    datasets.initialize()
    return AgentService(
        session_repository=sessions,
        run_repository=runs,
        dataset_repository=datasets,
    )


def _to_http_exception(exc: Exception) -> HTTPException:
    """把领域异常映射到 HTTP 状态码，避免在每个路由重复 try/except。"""
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
def get_agent_health(
    probe: Literal['models', 'inference'] = Query('models'),
    model_key: str | None = Query(None, min_length=1),
    _principal: Principal = Depends(get_principal),
) -> dict[str, object]:
    """Probe Agent persistence plus configured model runtimes without leaks."""
    try:
        _agent_service()
    except Exception:
        raise HTTPException(
            status_code=503,
            detail={'code': 'agent_backend_unavailable'},
        ) from None
    try:
        runtime = probe_runtime_health(probe=probe, model_key=model_key)
    except AgentRegistryUnavailable:
        raise HTTPException(
            status_code=503,
            detail={'code': 'agent_registry_unavailable'},
        ) from None
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={'code': 'unknown_model_key'},
        ) from None
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail={'code': 'model_key_required_for_inference'},
        ) from None
    runtime.update({
        'agent_contract': AGENT_API_CONTRACT_VERSION,
        'capabilities': list(AGENT_API_CAPABILITIES),
        'database_ready': True,
        'training_worker_required': True,
        'frontend_required': False,
    })
    return runtime


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
    except (TrainingConfigValidationError, ValueError) as exc:
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
    except (
        AgentSessionNotFound,
        AgentExperimentNotFound,
        AgentSessionClosed,
        AgentConfigCollision,
        TrainingConfigValidationError,
    ) as exc:
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
    except (
        AgentSessionNotFound,
        AgentExperimentNotFound,
        TrainingConfigValidationError,
    ) as exc:
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
    except (AgentSessionNotFound, TrainingConfigValidationError) as exc:
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
    except (
        AgentSessionNotFound,
        AgentExperimentNotFound,
        AgentSessionClosed,
        TrainingConfigValidationError,
    ) as exc:
        raise _to_http_exception(exc) from exc

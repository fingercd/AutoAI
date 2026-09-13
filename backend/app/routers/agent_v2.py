"""V2 routes use the same service, repository and Run submission boundary."""
from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from . import agent
from ..http.principal import Principal, get_principal
from ..agent.contracts import AgentDomainError, FinalizeAgentSessionRequest, ReconcileAgentSessionRequest
from ..agent.contracts_v2 import V2, CreateAgentSessionRequestV2, CreateAgentExperimentRequestV2
from ..agent.capabilities import health_payload
from ..model_config import model_capability_snapshot
from ..contracts import TrainingConfigValidationError

router = APIRouter(prefix='/api/agent/v2')


def call(method, principal, **kwargs):
    try:
        return getattr(agent._agent_service(V2), method)(principal=principal, **kwargs)
    except (AgentDomainError, TrainingConfigValidationError, ValueError) as exc:
        raise agent._to_http_exception(exc) from exc


@router.get('/health')
def health():
    result = agent.agent_health()
    snapshot = model_capability_snapshot()
    result.update(contract_version=V2, **snapshot)
    result['capabilities']['create_experiment'] &= any(m['available'] for m in snapshot['models'])
    if not any(m['available'] for m in snapshot['models']):
        result['status'] = 'unavailable'
    return result


@router.post('/sessions', status_code=201)
def create_session(payload: CreateAgentSessionRequestV2, principal: Principal = Depends(get_principal)):
    return call('create_session', principal, payload=payload)


@router.get('/sessions/{session_id}')
def get_session(session_id: str, principal: Principal = Depends(get_principal)):
    return call('get_session', principal, session_id=session_id)


@router.post('/sessions/{session_id}/experiments', status_code=202)
def create_experiment(session_id: str, payload: CreateAgentExperimentRequestV2, principal: Principal = Depends(get_principal)):
    return call('create_experiment', principal, session_id=session_id, payload=payload)


@router.get('/sessions/{session_id}/experiments/{run_id}/feedback')
def feedback(session_id: str, run_id: str, principal: Principal = Depends(get_principal)):
    return call('get_feedback', principal, session_id=session_id, run_id=run_id)


@router.post('/sessions/{session_id}/finalize')
def finalize(session_id: str, payload: FinalizeAgentSessionRequest, principal: Principal = Depends(get_principal)):
    return call('finalize_session', principal, session_id=session_id, selected_run_id=payload.selected_run_id)


@router.post('/sessions/{session_id}/reconcile')
def reconcile(session_id: str, payload: ReconcileAgentSessionRequest, principal: Principal = Depends(get_principal)):
    try:
        agent._agent_service(V2)._session(session_id, principal)
        result = agent._reconciliation_service().reconcile_session(session_id=session_id, principal=principal)
        result['contract_version'] = V2
        return result
    except AgentDomainError as exc:
        raise agent._to_http_exception(exc) from exc

"""V2 routes use the same service, repository and Run submission boundary."""
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Header
from . import agent
from .deps import get_agent_service
from ..http.principal import Principal, get_principal
from ..agent.contracts import AgentDomainError, FinalizeAgentSessionRequest, ReconcileAgentSessionRequest
from ..agent.contracts_v2 import V2, CreateAgentSessionRequestV2, CreateAgentExperimentRequestV2, CreateRecipeExperimentRequest, CreateKnowledgeExperimentRequest
from ..agent.capabilities import health_payload
from ..model_config import model_capability_snapshot
from ..contracts import TrainingConfigValidationError

router = APIRouter(prefix='/api/agent/v2')


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
        raise agent._to_http_exception(exc) from exc


@router.get('/health')
def health(revision: str | None = Depends(revision_header)):
    result = agent.agent_health()
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


@router.post('/sessions', status_code=201)
def create_session(payload: CreateAgentSessionRequestV2, principal: Principal = Depends(get_principal), revision: str | None = Depends(revision_header)):
    return call('create_session', principal, revision=revision, payload=payload)


@router.get('/sessions/{session_id}')
def get_session(session_id: str, principal: Principal = Depends(get_principal), revision: str | None = Depends(revision_header)):
    return call('get_session', principal, revision=revision, session_id=session_id)


@router.post('/sessions/{session_id}/experiments', status_code=202)
def create_experiment(session_id: str, payload: CreateKnowledgeExperimentRequest | CreateAgentExperimentRequestV2 | CreateRecipeExperimentRequest, principal: Principal = Depends(get_principal), revision: str | None = Depends(revision_header)):
    return call('create_experiment', principal, revision=revision, session_id=session_id, payload=payload)


@router.get('/sessions/{session_id}/experiments/{run_id}/feedback')
def feedback(session_id: str, run_id: str, principal: Principal = Depends(get_principal), revision: str | None = Depends(revision_header)):
    return call('get_feedback', principal, revision=revision, session_id=session_id, run_id=run_id)


@router.post('/sessions/{session_id}/finalize')
def finalize(session_id: str, payload: FinalizeAgentSessionRequest, principal: Principal = Depends(get_principal), revision: str | None = Depends(revision_header)):
    return call('finalize_session', principal, revision=revision, session_id=session_id, selected_run_id=payload.selected_run_id)


@router.post('/sessions/{session_id}/reconcile')
def reconcile(session_id: str, payload: ReconcileAgentSessionRequest, principal: Principal = Depends(get_principal), revision: str | None = Depends(revision_header)):
    try:
        require_negotiation(get_agent_service(V2), principal, revision, session_id)
        result = agent._reconciliation_service().reconcile_session(session_id=session_id, principal=principal)
        result['contract_version'] = V2
        return result
    except AgentDomainError as exc:
        raise agent._to_http_exception(exc) from exc

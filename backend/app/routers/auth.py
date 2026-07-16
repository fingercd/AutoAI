"""Authentication discovery and session verification endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ..http.principal import get_principal
from ..runs.contracts import Principal


router = APIRouter()


@router.get('/api/auth/config')
def auth_config(request: Request) -> dict[str, object]:
    """Expose non-secret browser bootstrap information."""

    settings = request.app.state.security_settings
    return {
        'mode': settings.mode,
        'auth_required': settings.auth_required,
        'token_storage': 'session',
    }


@router.get('/api/auth/session')
def auth_session(principal: Principal = Depends(get_principal)) -> dict[str, object]:
    """Validate the current Bearer token without exposing it."""

    return {
        'authenticated': True,
        'principal': {
            'owner_id': principal.owner_id,
            'tenant_id': principal.tenant_id,
        },
    }

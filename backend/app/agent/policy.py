"""Finite, versioned recipes for attributable Agent interventions."""

from __future__ import annotations

import hashlib
import json
from typing import Any


POLICY_CATALOG_VERSION = 'restricted-policy-v1'


def _proposal_id(recipe: dict[str, str]) -> str:
    encoded = json.dumps(
        recipe,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return 'p_' + hashlib.sha256(encoded).hexdigest()[:16]


def compile_proposal_catalog(
    *,
    allowed_models: list[str] | tuple[str, ...],
    evidence_card: dict[str, Any] | None,
    dynamic_preprocessing: bool,
) -> dict[str, Any]:
    """Compile deterministic recipes; no arbitrary parameter values enter."""
    risk_codes = set(
        (evidence_card or {}).get('statistics', {}).get('risk_codes', [])
    )
    normalizations = (
        ('zscore', 'minmax') if dynamic_preprocessing else ('zscore',)
    )
    class_balances = (
        ('class_weight', 'none')
        if dynamic_preprocessing and 'class_imbalance' in risk_codes
        else ('none',)
    )
    proposals: list[dict[str, str]] = []
    for model_type in sorted(set(allowed_models)):
        for normalization in normalizations:
            for class_balance in class_balances:
                recipe = {
                    'model_type': model_type,
                    'normalization': normalization,
                    'class_balance': class_balance,
                }
                proposals.append({
                    'proposal_id': _proposal_id(recipe),
                    **recipe,
                })
    if len(proposals) > 12:
        raise ValueError('proposal catalog exceeds bounded size')
    digest = hashlib.sha256(
        json.dumps(
            proposals,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
    ).hexdigest()
    return {
        'version': POLICY_CATALOG_VERSION,
        'dynamic_preprocessing': bool(dynamic_preprocessing),
        'proposal_count': len(proposals),
        'catalog_digest': digest,
        'proposals': proposals,
    }


def resolve_proposal(
    context: dict[str, Any],
    proposal_id: str | None,
) -> dict[str, str]:
    if not proposal_id:
        raise ValueError('proposal_id is required')
    catalog = context.get('proposal_catalog')
    if not isinstance(catalog, dict):
        raise ValueError('proposal catalog is unavailable')
    proposals = catalog.get('proposals')
    if not isinstance(proposals, list):
        raise ValueError('proposal catalog is unavailable')
    for item in proposals:
        if isinstance(item, dict) and item.get('proposal_id') == proposal_id:
            return {
                'proposal_id': str(item['proposal_id']),
                'model_type': str(item['model_type']),
                'normalization': str(item['normalization']),
                'class_balance': str(item['class_balance']),
            }
    raise ValueError('proposal_id is not in the locked catalog')

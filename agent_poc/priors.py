"""Strict, read-only small-sample priors for the Agent POC.

The bundled JSON is deliberately data-only.  A prior becomes visible to the
LLM only after it is matched to both train-only evidence and one canonical,
currently available proposal from the server-owned policy catalog.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PRIOR_SCHEMA_VERSION = 'small-sample-priors-v1'
PRIOR_PROJECTION_SCHEMA_VERSION = 'small-sample-prior-projection-v1'
DEFAULT_PRIOR_PATH = Path(__file__).resolve().parent / 'config' / 'priors-v1.json'
MAX_PRIOR_BYTES = 64 * 1024

_RISK_CODES = frozenset({
    'class_imbalance',
    'conflicting_duplicate_predictors',
    'duplicate_predictors',
    'high_dimension',
    'very_small_train_partition',
    'zero_variance_predictors',
})
_MODEL_TYPES = frozenset({'logistic_regression', 'svm', 'random_forest'})
_NORMALIZATIONS = frozenset({'zscore', 'minmax', 'area', 'none'})
_CLASS_BALANCES = frozenset({'none', 'class_weight'})
_RATIONALE_CODES = frozenset({
    'class_imbalance_compensation',
    'high_dimension_margin_check',
    'high_dimension_regularized_linear',
    'small_sample_regularized_linear',
    'small_sample_tree_comparison',
})
_RECIPE_KEYS = frozenset({
    'risk_codes',
    'model_type',
    'normalization',
    'class_balance',
    'rationale_code',
})
_PROPOSAL_ID = re.compile(r'^p_[0-9a-f]{16}$')


@dataclass(frozen=True)
class PriorRecipe:
    risk_codes: tuple[str, ...]
    model_type: str
    normalization: str
    class_balance: str
    rationale_code: str

    def as_dict(self) -> dict[str, Any]:
        return {
            'risk_codes': list(self.risk_codes),
            'model_type': self.model_type,
            'normalization': self.normalization,
            'class_balance': self.class_balance,
            'rationale_code': self.rationale_code,
        }


@dataclass(frozen=True)
class PriorCatalog:
    schema_version: str
    recipes: tuple[PriorRecipe, ...]
    digest: str


def _strict_string(value: Any, *, field: str, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f'prior field {field!r} is not an allowed enum value')
    return value


def _canonical_payload(
    schema_version: str,
    recipes: tuple[PriorRecipe, ...],
) -> dict[str, Any]:
    return {
        'schema_version': schema_version,
        'recipes': [recipe.as_dict() for recipe in recipes],
    }


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'prior catalog contains duplicate key {key!r}')
        result[key] = value
    return result


def load_prior_catalog(path: Path | None = None) -> PriorCatalog:
    """Load and fully validate the fixed prior catalog without mutating it."""
    selected = DEFAULT_PRIOR_PATH if path is None else Path(path)
    try:
        if selected.stat().st_size > MAX_PRIOR_BYTES:
            raise ValueError('prior catalog exceeds the fixed size limit')
        text = selected.read_text(encoding='utf-8')
    except (OSError, UnicodeError) as exc:
        raise ValueError('prior catalog cannot be read as UTF-8 JSON') from exc
    try:
        raw = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError('prior catalog cannot be read as UTF-8 JSON') from exc
    if not isinstance(raw, dict) or set(raw) != {'schema_version', 'recipes'}:
        raise ValueError('prior catalog must contain only schema_version and recipes')
    if raw.get('schema_version') != PRIOR_SCHEMA_VERSION:
        raise ValueError('prior catalog schema_version is invalid')
    raw_recipes = raw.get('recipes')
    if not isinstance(raw_recipes, list) or not raw_recipes:
        raise ValueError('prior catalog recipes must be a non-empty list')
    if len(raw_recipes) > 16:
        raise ValueError('prior catalog exceeds the fixed recipe limit')

    recipes: list[PriorRecipe] = []
    seen: set[tuple[object, ...]] = set()
    for raw_recipe in raw_recipes:
        if not isinstance(raw_recipe, dict) or set(raw_recipe) != _RECIPE_KEYS:
            raise ValueError('each prior recipe must use the exact fixed schema')
        risk_codes = raw_recipe.get('risk_codes')
        if (
            not isinstance(risk_codes, list)
            or not risk_codes
            or not all(isinstance(item, str) for item in risk_codes)
            or risk_codes != sorted(set(risk_codes))
            or not set(risk_codes).issubset(_RISK_CODES)
        ):
            raise ValueError('prior risk_codes must be sorted, unique, known enums')
        recipe = PriorRecipe(
            risk_codes=tuple(risk_codes),
            model_type=_strict_string(
                raw_recipe.get('model_type'),
                field='model_type',
                allowed=_MODEL_TYPES,
            ),
            normalization=_strict_string(
                raw_recipe.get('normalization'),
                field='normalization',
                allowed=_NORMALIZATIONS,
            ),
            class_balance=_strict_string(
                raw_recipe.get('class_balance'),
                field='class_balance',
                allowed=_CLASS_BALANCES,
            ),
            rationale_code=_strict_string(
                raw_recipe.get('rationale_code'),
                field='rationale_code',
                allowed=_RATIONALE_CODES,
            ),
        )
        identity = (
            recipe.risk_codes,
            recipe.model_type,
            recipe.normalization,
            recipe.class_balance,
        )
        if identity in seen:
            raise ValueError('prior catalog contains a duplicate recipe')
        seen.add(identity)
        recipes.append(recipe)

    frozen = tuple(recipes)
    canonical = _canonical_payload(PRIOR_SCHEMA_VERSION, frozen)
    return PriorCatalog(
        schema_version=PRIOR_SCHEMA_VERSION,
        recipes=frozen,
        digest=_digest(canonical),
    )


def _evidence_risks(observation: dict[str, Any]) -> frozenset[str]:
    context = observation.get('context')
    evidence = context.get('evidence_card') if isinstance(context, dict) else None
    if (
        not isinstance(evidence, dict)
        or evidence.get('schema_version') != 'small-sample-evidence-v1'
        or evidence.get('status') != 'ready'
        or evidence.get('scope') != 'train_only'
    ):
        return frozenset()
    statistics = evidence.get('statistics')
    raw_risks = statistics.get('risk_codes') if isinstance(statistics, dict) else None
    if (
        not isinstance(raw_risks, list)
        or not all(isinstance(item, str) for item in raw_risks)
        or raw_risks != sorted(set(raw_risks))
        or not set(raw_risks).issubset(_RISK_CODES)
    ):
        return frozenset()
    return frozenset(raw_risks)


def project_priors(
    catalog: PriorCatalog,
    *,
    observation: dict[str, Any],
    allowed_models: tuple[str, ...],
    proposal_recipes: tuple[dict[str, str], ...],
) -> dict[str, Any] | None:
    """Return prompt-safe advice mapped to available canonical proposals.

    The result contains no local path and no historical run identifier.  The
    only identifier is the server-issued proposal id needed to make the advice
    executable without copying arbitrary hyperparameters into an action.
    """
    if (
        catalog.schema_version != PRIOR_SCHEMA_VERSION
        or not catalog.recipes
        or any(
            not isinstance(recipe, PriorRecipe)
            or not recipe.risk_codes
            or recipe.risk_codes != tuple(sorted(set(recipe.risk_codes)))
            or not set(recipe.risk_codes).issubset(_RISK_CODES)
            or recipe.model_type not in _MODEL_TYPES
            or recipe.normalization not in _NORMALIZATIONS
            or recipe.class_balance not in _CLASS_BALANCES
            or recipe.rationale_code not in _RATIONALE_CODES
            for recipe in catalog.recipes
        )
        or catalog.digest != _digest(
            _canonical_payload(catalog.schema_version, catalog.recipes)
        )
    ):
        return None
    observed_risks = _evidence_risks(observation)
    if not observed_risks or not proposal_recipes:
        return None
    allowed = frozenset(allowed_models) & _MODEL_TYPES
    proposals: dict[tuple[str, str, str], dict[str, str]] = {}
    proposal_ids: set[str] = set()
    ambiguous_keys: set[tuple[str, str, str]] = set()
    for proposal in proposal_recipes:
        if not isinstance(proposal, dict):
            continue
        proposal_id = proposal.get('proposal_id')
        model_type = proposal.get('model_type')
        normalization = proposal.get('normalization')
        class_balance = proposal.get('class_balance')
        if (
            isinstance(proposal_id, str)
            and _PROPOSAL_ID.fullmatch(proposal_id)
            and isinstance(model_type, str)
            and model_type in allowed
            and isinstance(normalization, str)
            and normalization in _NORMALIZATIONS
            and isinstance(class_balance, str)
            and class_balance in _CLASS_BALANCES
        ):
            if proposal_id in proposal_ids:
                return None
            proposal_ids.add(proposal_id)
            key = (model_type, normalization, class_balance)
            if key in proposals:
                ambiguous_keys.add(key)
                continue
            proposals[key] = {
                'proposal_id': proposal_id,
                'model_type': model_type,
                'normalization': normalization,
                'class_balance': class_balance,
            }
    for key in ambiguous_keys:
        proposals.pop(key, None)

    recommendations: list[dict[str, Any]] = []
    seen_proposals: set[str] = set()
    for recipe in catalog.recipes:
        if recipe.model_type not in allowed:
            continue
        if not set(recipe.risk_codes).issubset(observed_risks):
            continue
        proposal = proposals.get((
            recipe.model_type,
            recipe.normalization,
            recipe.class_balance,
        ))
        if proposal is None or proposal['proposal_id'] in seen_proposals:
            continue
        seen_proposals.add(proposal['proposal_id'])
        recommendations.append({
            **proposal,
            'matched_risk_codes': list(recipe.risk_codes),
            'rationale_code': recipe.rationale_code,
        })
    if not recommendations:
        return None
    return {
        'schema_version': PRIOR_PROJECTION_SCHEMA_VERSION,
        'source_schema_version': catalog.schema_version,
        'source_digest': catalog.digest,
        'recommendations': recommendations,
    }

"""Server-authoritative budget estimation for Agent experiments.

The HTTP caller never supplies a model-fit estimate.  The service first builds and
validates the existing training configuration, then this module derives the maximum
number of ``fit`` calls from the locked HPO profile.  Agent Session V1 only permits a
single stratified holdout, so the bound is ``HPO candidates + one final fit``.
"""

from __future__ import annotations

from typing import Any, Mapping


class AgentBudgetEstimateError(ValueError):
    """A validated Agent training config cannot be safely budgeted."""


_STANDARD_MODEL_CANDIDATES = {
    'logistic_regression': 3,
    'svm': 5,
    'random_forest': 10,
}


def estimate_model_fit_upper_bound(config: Mapping[str, Any]) -> int:
    """Return the server-derived maximum number of model fits for one Run.

    Only fields already produced by the validated server-side training config are
    considered.  In particular, caller-provided counters such as
    ``model_fit_count`` are deliberately ignored.
    """
    model_type = config.get('model_type')
    if model_type not in _STANDARD_MODEL_CANDIDATES:
        raise AgentBudgetEstimateError('Agent budget only supports locked traditional models')

    split_mode = config.get('split_mode', 'stratified_holdout')
    if split_mode != 'stratified_holdout':
        raise AgentBudgetEstimateError('Agent budget requires stratified_holdout')

    profile = config.get('hpo_profile', 'standard')
    if not isinstance(profile, str) or profile not in {'off', 'tiny', 'standard'}:
        raise AgentBudgetEstimateError('unknown HPO profile')
    standard_candidates = _STANDARD_MODEL_CANDIDATES[model_type]
    candidate_limit = {
        'off': 1,
        'tiny': min(3, standard_candidates),
        'standard': standard_candidates,
    }[profile]

    envelope = config.get('agent_execution')
    if not isinstance(envelope, Mapping):
        raise AgentBudgetEstimateError('budgeted Agent Run requires a guard envelope')
    maximum = envelope.get('max_hpo_candidates')
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or maximum < candidate_limit
    ):
        raise AgentBudgetEstimateError('guard HPO envelope is below the locked profile')

    return candidate_limit + 1

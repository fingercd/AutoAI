"""LLM structured decision schema and validation helpers."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class RunExperimentDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')

    decision: Literal['RUN_EXPERIMENT']
    model_type: Literal['logistic_regression', 'svm', 'random_forest']
    normalization: Literal['zscore', 'minmax', 'area', 'none'] = 'zscore'
    class_balance: Literal['none', 'class_weight'] = 'none'
    parent_run_id: str | None = None
    proposal_id: str | None = Field(None, pattern=r'^p_[0-9a-f]{16}$')
    rationale: str = Field(min_length=1, max_length=2000)


class ReplanDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')

    decision: Literal['REPLAN']
    action_id: Literal['choose_unused_proposal']
    parent_run_id: str = Field(min_length=1)
    proposal_id: str = Field(pattern=r'^p_[0-9a-f]{16}$')
    model_type: Literal['logistic_regression', 'svm', 'random_forest']
    normalization: Literal['zscore', 'minmax', 'area', 'none']
    class_balance: Literal['none', 'class_weight']
    rationale: str = Field(min_length=1, max_length=2000)


class FinalizeDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')

    decision: Literal['FINALIZE']
    selected_run_id: str = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=2000)


class RequestHumanDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')

    decision: Literal['REQUEST_HUMAN']
    reason: str = Field(min_length=1, max_length=2000)


AgentDecision = Annotated[
    RunExperimentDecision | ReplanDecision | FinalizeDecision | RequestHumanDecision,
    Field(discriminator='decision'),
]
DECISION_ADAPTER = TypeAdapter(AgentDecision)

# Qwen3.5's constrained decoder can otherwise spend the full generation
# budget inside a free-form rationale string.  These short labels preserve
# the Pydantic decision contract while making the wire schema finite enough
# for all three local backbones to close a JSON object deterministically.
SHORT_DECISION_REASONS = (
    'baseline',
    'compare',
    'retry',
    'select_best',
    'finalize',
    'stop',
    'needs_human',
)


def decision_json_schema(
    *,
    allowed_decisions: tuple[str, ...] = (
        'RUN_EXPERIMENT',
        'FINALIZE',
        'REQUEST_HUMAN',
    ),
    allowed_models: tuple[str, ...] = (
        'logistic_regression',
        'svm',
        'random_forest',
    ),
    selected_run_ids: tuple[str, ...] = (),
    proposal_recipes: tuple[dict[str, str], ...] = (),
    failed_run_ids: tuple[str, ...] = (),
    allowed_replan_actions: tuple[str, ...] = (),
) -> dict[str, object]:
    """Return a finite wire schema while Pydantic remains the final validator.

    The wire schema is intentionally policy-scoped for weak/quantized local
    backbones.  A single allowed branch avoids unconstrained strings and lets
    the model spend its small output budget on the actual action fields.
    """
    reasons = {'enum': list(SHORT_DECISION_REASONS), 'type': 'string'}
    branches: list[dict[str, object]] = []
    if 'RUN_EXPERIMENT' in allowed_decisions:
        if proposal_recipes:
            recipe_branches: list[dict[str, object]] = []
            for recipe in proposal_recipes:
                if recipe.get('model_type') not in allowed_models:
                    continue
                recipe_branches.append({
                    'type': 'object',
                    'additionalProperties': False,
                    'properties': {
                        'decision': {'const': 'RUN_EXPERIMENT', 'type': 'string'},
                        'proposal_id': {'const': recipe['proposal_id'], 'type': 'string'},
                        'model_type': {'const': recipe['model_type'], 'type': 'string'},
                        'normalization': {'const': recipe['normalization'], 'type': 'string'},
                        'class_balance': {'const': recipe['class_balance'], 'type': 'string'},
                        'rationale': reasons,
                    },
                    'required': [
                        'decision', 'proposal_id', 'model_type',
                        'normalization', 'class_balance', 'rationale',
                    ],
                })
            if not recipe_branches:
                raise ValueError('proposal_recipes contain no allowed model')
            branches.append({'type': 'object', 'oneOf': recipe_branches})
        else:
            branches.append({
                'type': 'object',
                'additionalProperties': False,
                'properties': {
                    'decision': {'const': 'RUN_EXPERIMENT', 'type': 'string'},
                    'model_type': {'enum': list(allowed_models), 'type': 'string'},
                    'normalization': {
                        'enum': ['zscore', 'minmax', 'area', 'none'],
                        'type': 'string',
                    },
                    'class_balance': {
                        'enum': ['none', 'class_weight'],
                        'type': 'string',
                    },
                    'rationale': reasons,
                },
                'required': ['decision', 'model_type', 'rationale'],
            })
    if 'REPLAN' in allowed_decisions:
        if not proposal_recipes or not failed_run_ids:
            raise ValueError('REPLAN requires proposals and failed parent runs')
        if 'choose_unused_proposal' not in allowed_replan_actions:
            raise ValueError('REPLAN action is not allowed by diagnosis')
        replan_branches: list[dict[str, object]] = []
        for recipe in proposal_recipes:
            if recipe.get('model_type') not in allowed_models:
                continue
            replan_branches.append({
                'type': 'object',
                'additionalProperties': False,
                'properties': {
                    'decision': {'const': 'REPLAN', 'type': 'string'},
                    'action_id': {
                        'const': 'choose_unused_proposal', 'type': 'string',
                    },
                    'parent_run_id': {
                        'enum': list(failed_run_ids), 'type': 'string',
                    },
                    'proposal_id': {
                        'const': recipe['proposal_id'], 'type': 'string',
                    },
                    'model_type': {
                        'const': recipe['model_type'], 'type': 'string',
                    },
                    'normalization': {
                        'const': recipe['normalization'], 'type': 'string',
                    },
                    'class_balance': {
                        'const': recipe['class_balance'], 'type': 'string',
                    },
                    'rationale': reasons,
                },
                'required': [
                    'decision', 'action_id', 'parent_run_id', 'proposal_id',
                    'model_type', 'normalization', 'class_balance', 'rationale',
                ],
            })
        if not replan_branches:
            raise ValueError('REPLAN contains no allowed proposal')
        branches.append({'type': 'object', 'oneOf': replan_branches})
    if 'FINALIZE' in allowed_decisions:
        selected: dict[str, object]
        if selected_run_ids:
            selected = {'enum': list(selected_run_ids), 'type': 'string'}
        else:
            selected = {'minLength': 1, 'type': 'string'}
        branches.append({
            'type': 'object',
            'additionalProperties': False,
            'properties': {
                'decision': {'const': 'FINALIZE', 'type': 'string'},
                'selected_run_id': selected,
                'rationale': reasons,
            },
            'required': ['decision', 'selected_run_id', 'rationale'],
        })
    if 'REQUEST_HUMAN' in allowed_decisions:
        branches.append({
            'type': 'object',
            'additionalProperties': False,
            'properties': {
                'decision': {'const': 'REQUEST_HUMAN', 'type': 'string'},
                'reason': reasons,
            },
            'required': ['decision', 'reason'],
        })
    if not branches:
        raise ValueError('at least one decision branch is required')
    if len(branches) == 1:
        return branches[0]
    return {'type': 'object', 'oneOf': branches}


class DecisionParseError(ValueError):
    """模型输出为空、非法 JSON 或不符合判别联合 schema。"""


class PolicyViolation(RuntimeError):
    """Python 侧受控循环拒绝了不符合预算/状态的 LLM 决策。"""

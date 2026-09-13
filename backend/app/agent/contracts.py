"""Agent API v1 的稳定请求与领域错误契约。"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


AGENT_API_CONTRACT_VERSION = 'agent-session-v1'
AGENT_OBSERVATION_VERSION = 'agent-observation-v1'
AGENT_RESERVATION_PROTOCOL_VERSION = 'agent-reservation-reconciliation-v1'
AGENT_METADATA_VERSION = 'agent-metadata-v1'

from ..model_catalog import LEGACY_AGENT_MODELS as AGENT_ALLOWED_MODELS
AGENT_SELECTION_METRICS = ('macro_f1', 'balanced_accuracy')
AGENT_NORMALIZATIONS = ('zscore', 'minmax', 'area', 'none')
AGENT_CLASS_BALANCES = ('none', 'class_weight')

_REQUEST_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$')


class AgentDomainError(RuntimeError):
    """统一、可安全公开的 Agent 领域错误。"""

    def __init__(self, code: str, message: str, *, status_code: int,
                 retryable: bool = False, allowed_actions: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.allowed_actions = allowed_actions

    def detail(self) -> dict[str, object]:
        return {
            'code': self.code,
            'message': self.message,
            'retryable': self.retryable,
            'allowed_actions': list(self.allowed_actions),
        }


class AgentEvaluationBlock(BaseModel):
    model_config = ConfigDict(extra='forbid')

    split_mode: Literal['stratified_holdout'] = 'stratified_holdout'
    split_train: int = Field(8, strict=True, ge=1, le=9)
    split_valid: int = Field(1, strict=True, ge=1, le=9)
    split_test: int = Field(1, strict=True, ge=1, le=9)


class AgentContextPolicy(BaseModel):
    model_config = ConfigDict(extra='forbid')

    source_role: Literal['development', 'benchmark', 'domain'] = 'development'
    case_write: bool = False


class _IdempotentRequest(BaseModel):
    client_request_id: str | None = Field(None, min_length=1, max_length=128)

    @field_validator('client_request_id')
    @classmethod
    def valid_request_id(cls, value: str | None) -> str | None:
        if value is not None and not _REQUEST_ID.fullmatch(value):
            raise ValueError('client_request_id 只能包含字母、数字、点、下划线、冒号和连字符')
        return value


class CreateAgentSessionRequest(_IdempotentRequest):
    model_config = ConfigDict(extra='forbid')

    dataset_id: str = Field(..., min_length=1, max_length=128)
    selection_metric: str
    allowed_models: list[str] = Field(..., min_length=1, max_length=3)
    max_runs: int = Field(..., strict=True, ge=1, le=10)
    seed: int = Field(42, strict=True, ge=0)
    evaluation: AgentEvaluationBlock = Field(default_factory=AgentEvaluationBlock)
    modules: list[str] = Field(default_factory=list, max_length=16)
    context_policy: AgentContextPolicy = Field(default_factory=AgentContextPolicy)

    def validate_business(self) -> None:
        if self.selection_metric not in AGENT_SELECTION_METRICS:
            raise AgentDomainError('agent_invalid_action', f'selection_metric 必须来自 {list(AGENT_SELECTION_METRICS)}', status_code=422)
        unknown = [item for item in self.allowed_models if item not in AGENT_ALLOWED_MODELS]
        if unknown:
            raise AgentDomainError('agent_invalid_action', f'allowed_models 仅支持 {list(AGENT_ALLOWED_MODELS)}，当前收到 {unknown}', status_code=422)
        if len(set(self.allowed_models)) != len(self.allowed_models):
            raise AgentDomainError(
                'agent_invalid_action', 'allowed_models 不能包含重复项', status_code=422
            )
        ratios = (self.evaluation.split_train, self.evaluation.split_valid, self.evaluation.split_test)
        if sum(ratios) != 10:
            raise AgentDomainError(
                'agent_invalid_action', 'evaluation 比例三项必须相加等于 10', status_code=422
            )

    validate = validate_business


class CreateAgentExperimentRequest(_IdempotentRequest):
    model_config = ConfigDict(extra='forbid')

    model_type: str
    normalization: str = 'zscore'
    class_balance: str = 'none'
    parent_run_id: str | None = Field(None, min_length=1, max_length=128)
    rationale: str | None = Field(None, max_length=2000)

    def validate_business(self) -> None:
        if self.model_type not in AGENT_ALLOWED_MODELS:
            raise AgentDomainError('agent_invalid_action', f'model_type 仅支持 {list(AGENT_ALLOWED_MODELS)}', status_code=422)
        if self.normalization not in AGENT_NORMALIZATIONS:
            raise AgentDomainError('agent_invalid_action', f'normalization 仅支持 {list(AGENT_NORMALIZATIONS)}', status_code=422)
        if self.class_balance not in AGENT_CLASS_BALANCES:
            raise AgentDomainError('agent_invalid_action', f'class_balance 仅支持 {list(AGENT_CLASS_BALANCES)}', status_code=422)

    validate = validate_business


class FinalizeAgentSessionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    selected_run_id: str = Field(..., min_length=1, max_length=128)


class ReconcileAgentSessionRequest(BaseModel):
    """显式空请求体；任何客户端提供的恢复参数都被拒绝。"""

    model_config = ConfigDict(extra='forbid')

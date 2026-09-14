"""Strict v2 requests. The original v1 wire models and hashes stay unchanged."""
from typing import Any, ClassVar, Literal
from pydantic import ConfigDict, Field
from .contracts import (CreateAgentSessionRequest, CreateAgentExperimentRequest,
                        AGENT_SELECTION_METRICS, AgentDomainError)
from ..model_catalog import MODELS_BY_ID, RETIRED_MODEL_ALIASES

V2 = 'agent-session-v2'


def invalid(message):
    return AgentDomainError('agent_invalid_action', message, status_code=422)


class CreateAgentSessionRequestV2(CreateAgentSessionRequest):
    model_config = ConfigDict(extra='forbid', strict=True)
    contract_version: ClassVar[str] = V2
    allowed_models: list[str] = Field(min_length=1)
    max_runs: int = Field(1, strict=True, ge=1, le=1)
    model_configs: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def validate_business(self):
        if type(self.max_runs) is not int or self.max_runs != 1:
            raise invalid('Second step requires max_runs=1')
        if self.selection_metric not in AGENT_SELECTION_METRICS:
            raise invalid('Unknown selection metric')
        if len(set(self.allowed_models)) != len(self.allowed_models):
            raise invalid('Duplicate model ID')
        if set(self.allowed_models) - (MODELS_BY_ID.keys() | RETIRED_MODEL_ALIASES.keys()):
            raise invalid('Unknown canonical model ID')
        if set(self.model_configs) - set(self.allowed_models):
            raise invalid('Model configuration outside allowed set')
        if self.evaluation.model_dump() != dict(split_mode='stratified_holdout',split_train=8,split_valid=1,split_test=1):
            raise invalid('Second step requires frozen group holdout')


class CreateAgentExperimentRequestV2(CreateAgentExperimentRequest):
    model_config = ConfigDict(extra='forbid', strict=True)
    contract_version: ClassVar[str] = V2
    normalization: Literal['zscore'] = 'zscore'
    class_balance: Literal['none'] = 'none'
    model_params: dict[str, Any] = Field(default_factory=dict)

    def validate_business(self):
        if self.model_type not in MODELS_BY_ID and self.model_type not in RETIRED_MODEL_ALIASES:
            raise invalid('Unknown canonical model ID')
        if self.parent_run_id is not None:
            raise invalid('Second step has no parent experiment')

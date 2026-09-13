"""V2 extends existing State blocks; v1 types and serialized defaults stay frozen."""
from typing import Literal
from pydantic import Field, model_validator
from . import state as v1
from agent_poc.clients.contracts_v2 import CapabilitySnapshot, FrozenSnapshot, ResolvedExecution, Scalar


class TaskState(v1.TaskState):
    model_configs: dict[v1.Identifier,dict[v1.Identifier,Scalar]] = Field(default_factory=dict)


    @model_validator(mode='after')
    def configured_models_are_allowed(self):
        if set(self.model_configs) - set(self.allowed_models):
            raise ValueError('model_configs keys must belong to allowed_models')
        return self


class VersionsState(v1.VersionsState):
    state: Literal['agent-state-v2'] = 'agent-state-v2'
    api: Literal['agent-session-v2'] = 'agent-session-v2'
    observation: Literal['agent-observation-v2'] = 'agent-observation-v2'
    metadata: Literal['agent-metadata-v2'] = 'agent-metadata-v2'
    context_projection: Literal['agent-context-step2-v1'] = 'agent-context-step2-v1'
    prompt: Literal['agent-decision-step2-v1'] = 'agent-decision-step2-v1'
    capabilities: v1.VersionRef = Field(default_factory=lambda:v1.VersionRef(status='ready',version='agent-model-catalog-v1'))


class ContextPolicy(v1.ContextPolicy):
    projection: Literal['agent-context-step2-v1'] = 'agent-context-step2-v1'


class ModulePolicyState(v1.ModulePolicyState):
    context_policy: ContextPolicy = Field(default_factory=ContextPolicy)


class CapabilitiesState(v1.CapabilitiesState):
    wire_snapshot: CapabilitySnapshot | None = None
    frozen_snapshot: FrozenSnapshot | None = None
    excluded_models: dict[v1.Identifier,v1.Identifier] = Field(default_factory=dict)


class DirectAction(v1.DirectAction):
    model_params: dict[v1.Identifier,Scalar]


class DecisionState(v1.DecisionState):
    action: DirectAction | None = None


class EffectiveConfig(v1.EffectiveConfig):
    config_stage: Literal['submission']
    config_policy_version: Literal['agent-model-config-v1','agent-model-config-v2']
    config_policy_digest: v1.Digest
    model_params: dict[v1.Identifier,Scalar]


class SessionRequest(v1.SessionRequest):
    context_policy: ContextPolicy = Field(default_factory=ContextPolicy)
    model_configs: dict[v1.Identifier,dict[v1.Identifier,Scalar]] = Field(default_factory=dict)


class ExperimentRequest(v1.ExperimentRequest):
    model_params: dict[v1.Identifier,Scalar]


class ExecutionState(v1.ExecutionState):
    submission_content: ExperimentRequest | None = None
    effective_action: DirectAction | None = None
    effective_config: EffectiveConfig | None = None
    metadata_version: Literal['agent-metadata-v2'] = 'agent-metadata-v2'
    resolved_execution: ResolvedExecution = Field(default_factory=lambda:ResolvedExecution(status='pending',parameters=None))


class PendingOperation(v1.PendingOperation):
    content: SessionRequest | ExperimentRequest | v1.FinalizeRequest | None = None


class RecoveryState(v1.RecoveryState):
    pending_operation: PendingOperation | None = None


class FeedbackState(v1.FeedbackState):
    observation_version: Literal['agent-observation-v2'] = 'agent-observation-v2'


class StateModel(v1.StateModel):
    task: TaskState
    versions: VersionsState
    module_policy: ModulePolicyState = Field(default_factory=ModulePolicyState)
    capabilities: CapabilitiesState = Field(default_factory=CapabilitiesState)
    decision: DecisionState = Field(default_factory=DecisionState)
    execution: ExecutionState = Field(default_factory=ExecutionState)
    recovery: RecoveryState = Field(default_factory=RecoveryState)
    feedback: FeedbackState = Field(default_factory=FeedbackState)

    @model_validator(mode='after')
    def frozen_parameters(self):
        frozen=self.capabilities.frozen_snapshot
        for action in (self.decision.action,self.execution.effective_action,self.execution.submission_content,self.execution.effective_config):
            if action is None:
                continue
            if frozen is None or action.model_type not in frozen.model_configs or action.model_params != frozen.model_configs[action.model_type]:
                raise ValueError('configuration differs from frozen Session')
        return self


def new_state(**kwargs):
    configs=kwargs.pop('model_configs',None) or {}
    kwargs['prompt_version']='agent-decision-step2-v1'
    old=v1.new_state(**kwargs)
    old['task']['model_configs']=configs
    old['versions'].update(state='agent-state-v2',api='agent-session-v2',observation='agent-observation-v2',
        metadata='agent-metadata-v2',context_projection='agent-context-step2-v1',
        capabilities={'status':'ready','version':'agent-model-catalog-v1'})
    old['module_policy']['context_policy']['projection']='agent-context-step2-v1'
    old['execution']['metadata_version']='agent-metadata-v2'
    old['feedback']['observation_version']='agent-observation-v2'
    old['identity']['startup_config_fingerprint']=v1.fingerprint(v1._startup_payload(old['task'],old['versions'],
        old['module_policy'],old['budget'],old['identity']['backend_fingerprint'],old['identity']['principal_fingerprint'],
        old['identity']['runtime_config_fingerprint']))
    return StateModel.model_validate(old).model_dump(mode='json')

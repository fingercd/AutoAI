"""Strict legacy v2 codec exports; no initialization or patch implementation."""
from .state import (TaskStateV2 as TaskState, VersionsStateV2 as VersionsState, ContextPolicyV2 as ContextPolicy, ModulePolicyStateV2 as ModulePolicyState, CapabilitiesStateV2 as CapabilitiesState, DirectActionV2 as DirectAction, DecisionStateV2 as DecisionState, EffectiveConfigV2 as EffectiveConfig, SessionRequestV2 as SessionRequest, ExperimentRequestV2 as ExperimentRequest, ExecutionStateV2 as ExecutionState, PendingOperationV2 as PendingOperation, RecoveryStateV2 as RecoveryState, FeedbackStateV2 as FeedbackState, StateModelV2 as StateModel)
from .state import new_state as _new_state

def new_state(**kwargs):
    return _new_state(**kwargs, wire_version="agent-state-v2")

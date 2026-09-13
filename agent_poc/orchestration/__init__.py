"""Persistent orchestration for one backend-owned scientific experiment."""

import os

# This boundary must run before any LangGraph/LangChain module is imported.
# Full remote SDK traces are outside the Agent's controlled State projection.
for _tracing_flag in ('LANGSMITH_TRACING', 'LANGCHAIN_TRACING_V2', 'LANGCHAIN_TRACING'):
    os.environ[_tracing_flag] = 'false'
del _tracing_flag

from .state import GraphState, StateModel, apply_patch, new_state

__all__ = ['GraphState', 'StateModel', 'apply_patch', 'new_state']

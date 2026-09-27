"""Closed version capability ranges; historical serialization stays version-specific."""
REVISIONS = tuple(f'agent-recipes-revision-v{i}' for i in range(1, 8))
STATES = tuple(f'agent-state-v{i}' for i in range(1, 10))
REVISIONS_SINCE = {i: frozenset(REVISIONS[i-1:]) for i in range(1, 8)}
STATES_SINCE = {i: frozenset(STATES[i-1:]) for i in range(1, 10)}

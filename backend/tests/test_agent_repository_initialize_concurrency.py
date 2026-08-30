from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from backend.app.agent.repository import AgentSessionRepository


def test_initialize_is_safe_under_concurrent_first_use(tmp_path):
    database = tmp_path / 'agent.sqlite3'

    def initialize(_):
        AgentSessionRepository(database).initialize()

    with ThreadPoolExecutor(max_workers=20) as executor:
        list(executor.map(initialize, range(40)))

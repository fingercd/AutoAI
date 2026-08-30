"""HTTP-only client for the five AutoAI Agent V1 endpoints."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

import httpx

from ..budget import BudgetController


class AutoAIClient:
    """No backend imports, SQLite access, filesystem access, or POST retry."""

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        *,
        timeout: httpx.Timeout | None = None,
        http_client: httpx.Client | None = None,
        budget_controller: BudgetController | None = None,
    ) -> None:
        headers = {'Authorization': f'Bearer {token}'} if token else {}
        self.http = http_client or httpx.Client(
            base_url=base_url.rstrip('/'),
            headers=headers,
            timeout=timeout or httpx.Timeout(30.0, connect=10.0),
        )
        self.budget_controller = budget_controller
        self.attempt_count = 0
        self.retry_attempt_count = 0

    def set_budget_controller(self, controller: BudgetController) -> None:
        self.budget_controller = controller

    def close(self) -> None:
        self.http.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        retry_get: bool = False,
        retry_finalize: bool = False,
    ) -> dict[str, Any]:
        attempts = 3 if retry_get or retry_finalize else 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            is_retry = attempt > 0
            if self.budget_controller is not None:
                self.budget_controller.before_api_attempt(retry=is_retry)
            self.attempt_count += 1
            if is_retry:
                self.retry_attempt_count += 1
            try:
                response = self.http.request(method, path, json=json)
                if self.budget_controller is not None:
                    self.budget_controller.check_wall_clock()
                if (
                    attempt + 1 < attempts
                    and response.status_code in {408, 429, 500, 502, 503, 504}
                ):
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError(f'{method} {path} returned a non-object JSON payload')
                if self.budget_controller is not None:
                    self.budget_controller.check_wall_clock()
                return payload
            except httpx.TransportError as exc:
                last_error = exc
                if self.budget_controller is not None:
                    self.budget_controller.check_wall_clock()
                if attempt + 1 >= attempts:
                    raise
            except httpx.HTTPStatusError:
                raise
        assert last_error is not None
        raise last_error

    def health(self) -> dict[str, Any]:
        return self._request('GET', '/health', retry_get=True)

    def agent_health(
        self,
        *,
        probe: str = 'models',
        model_key: str | None = None,
    ) -> dict[str, Any]:
        query = {'probe': probe}
        if model_key:
            query['model_key'] = model_key
        return self._request(
            'GET',
            f"/api/agent/health?{urlencode(query)}",
            retry_get=True,
        )

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request('POST', '/api/agent/sessions', json=payload)

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self._request('GET', f'/api/agent/sessions/{session_id}', retry_get=True)

    def create_experiment(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        # POST experiment is deliberately never retried: an unknown response may
        # already have queued a real Run.
        return self._request(
            'POST', f'/api/agent/sessions/{session_id}/experiments', json=payload
        )

    def feedback(self, session_id: str, run_id: str) -> dict[str, Any]:
        return self._request(
            'GET',
            f'/api/agent/sessions/{session_id}/experiments/{run_id}/feedback',
            retry_get=True,
        )

    def finalize(self, session_id: str, selected_run_id: str) -> dict[str, Any]:
        # Server finalize is idempotent for the same selected_run_id, so a
        # transient transport failure may be retried safely.
        return self._request(
            'POST',
            f'/api/agent/sessions/{session_id}/finalize',
            json={'selected_run_id': selected_run_id},
            retry_finalize=True,
        )

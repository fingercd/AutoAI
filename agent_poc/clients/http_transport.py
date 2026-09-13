"""Synchronous facade over a cancellable HTTP request with one wall-clock limit."""
from __future__ import annotations

import asyncio
import time
from typing import Any


class ResponseTooLarge(RuntimeError):
    pass


def bounded_request(method: str, url: str, *, headers: dict[str, str],
                    json: dict[str, Any] | None, timeout: float, max_response_bytes: int):
    """Bound connection, headers, and all streamed reads by a single deadline.

    HTTPX's individual read timeout is insufficient for a peer which sends a
    tiny chunk before every read timeout. asyncio.timeout cancels the entire
    request, including connection acquisition and response-body consumption.
    This facade is called by synchronous graph nodes; it creates no daemon
    threads and closes its event loop/client when the request ends.
    """
    import httpx

    async def send():
        deadline = time.monotonic() + timeout
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
                async with client.stream(method, url, headers=headers, json=json) as response:
                    chunks, size = [], 0
                    async for chunk in response.aiter_bytes():
                        if time.monotonic() >= deadline:
                            raise TimeoutError('HTTP request deadline exceeded')
                        size += len(chunk)
                        if size > max_response_bytes:
                            raise ResponseTooLarge('HTTP response size limit exceeded')
                        chunks.append(chunk)
                    if time.monotonic() >= deadline:
                        raise TimeoutError('HTTP request deadline exceeded')
                    return httpx.Response(response.status_code, content=b''.join(chunks))

    return asyncio.run(send())

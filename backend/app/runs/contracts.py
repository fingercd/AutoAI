"""Run 状态机使用的不可变领域对象和兼容状态映射。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RunState = Literal['queued', 'running', 'succeeded', 'failed', 'cancelled']

LEGACY_STATUS: dict[RunState, str] = {
    'queued': 'pending',
    'running': 'running',
    'succeeded': 'success',
    'failed': 'failed',
    'cancelled': 'paused',
}


@dataclass(frozen=True)
class Principal:
    """由服务端注入的所有者/租户范围；本地模式两个字段均为空。"""
    owner_id: str | None = None
    tenant_id: str | None = None


@dataclass(frozen=True)
class RunRecord:
    """SQLite 中一个 Run 的完整规范快照。"""
    run_id: str
    state: RunState
    version: int
    dataset_id: str | None
    legacy_data_path: str | None
    config: dict[str, Any]
    progress: dict[str, Any] = field(default_factory=dict)
    claim_token: str | None = None
    worker_id: str | None = None
    lease_expires_at: str | None = None
    manifest_name: str | None = None
    error: str | None = None

    @property
    def legacy_status(self) -> str:
        return LEGACY_STATUS[self.state]

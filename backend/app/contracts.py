"""HTTP 训练请求与内部兼容配置的轻量契约。

身份字段有意不出现在请求模型中；本地模式的 Principal 只能由服务端依赖注入。
`extra='forbid'` 同时阻止旧客户端偷偷传入 owner_id/tenant_id。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TrainingRunRequest(BaseModel):
    """创建 queued Run 时允许的顶层字段。"""
    model_config = ConfigDict(extra='forbid')

    dataset_id: str | None = None
    data_path: str | None = None
    test_dataset_id: str | None = None
    test_data_path: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class TrainingSpec:
    """把旧字典配置包成稳定的内部访问接口。"""
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = dict(values)

    @classmethod
    def from_legacy(cls, values: dict[str, Any] | None) -> 'TrainingSpec':
        return cls(values or {})

    def to_legacy_dict(self) -> dict[str, Any]:
        return dict(self.values)

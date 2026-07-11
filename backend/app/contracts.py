from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TrainingRunRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    dataset_id: str | None = None
    data_path: str | None = None
    test_dataset_id: str | None = None
    test_data_path: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class TrainingSpec:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = dict(values)

    @classmethod
    def from_legacy(cls, values: dict[str, Any] | None) -> 'TrainingSpec':
        return cls(values or {})

    def to_legacy_dict(self) -> dict[str, Any]:
        return dict(self.values)

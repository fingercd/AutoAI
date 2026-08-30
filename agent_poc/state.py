"""Configuration and deterministic local policy state for the POC."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 only
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(frozen=True)
class ModelConfig:
    key: str
    base_url: str
    served_model_name: str
    model_path: str
    revision: str = 'master'


@dataclass(frozen=True)
class AgentConfig:
    autoai_base_url: str
    dataset_id: str
    allowed_models: tuple[str, ...] = (
        'logistic_regression',
        'svm',
        'random_forest',
    )
    selection_metric: str = 'macro_f1'
    max_runs: int = 1
    seed: int = 42
    split_mode: str = 'stratified_holdout'
    split_train: int = 8
    split_valid: int = 1
    split_test: int = 1
    poll_interval_seconds: float = 15.0
    run_timeout_seconds: float = 3600.0
    llm_timeout_seconds: float = 600.0
    temperature: float = 0.0
    trace_path: Path = Path('outputs/agent_poc_acceptance/trace.jsonl')
    code_revision: str = 'unknown'

    def session_payload(self) -> dict[str, Any]:
        return {
            'dataset_id': self.dataset_id,
            'selection_metric': self.selection_metric,
            'allowed_models': list(self.allowed_models),
            'max_runs': self.max_runs,
            'seed': self.seed,
            'evaluation': {
                'split_mode': self.split_mode,
                'split_train': self.split_train,
                'split_valid': self.split_valid,
                'split_test': self.split_test,
            },
        }


@dataclass
class LoopState:
    """Python-owned budget/config state; never trust LLM self-reported budget."""

    max_runs: int
    submitted_config_hashes: set[str] = field(default_factory=set)
    run_ids: list[str] = field(default_factory=list)
    successful_run_ids: set[str] = field(default_factory=set)

    @property
    def remaining_runs(self) -> int:
        return max(0, self.max_runs - len(self.run_ids))


def action_config_hash(payload: dict[str, Any]) -> str:
    """与 AutoAI Adapter 一致：rationale 不改变 effective config。"""
    canonical = {
        key: value for key, value in payload.items() if key not in {'rationale', 'decision'}
    }
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'config field {key!r} must be a non-empty string')
    return value.strip()


def load_model_registry(path: Path) -> dict[str, ModelConfig]:
    with path.open('rb') as handle:
        raw = tomllib.load(handle)
    models = raw.get('models')
    if not isinstance(models, dict):
        raise ValueError('models.toml must contain [models.*] entries')
    result: dict[str, ModelConfig] = {}
    for key, item in models.items():
        if not isinstance(item, dict):
            raise ValueError(f'model config {key!r} must be a table')
        result[str(key)] = ModelConfig(
            key=str(key),
            base_url=_require_string(item, 'base_url'),
            served_model_name=_require_string(item, 'served_model_name'),
            model_path=_require_string(item, 'model_path'),
            revision=str(item.get('revision', 'master')),
        )
    return result


def load_agent_config(path: Path, *, dataset_id: str, max_runs: int | None = None, trace_path: Path | None = None) -> AgentConfig:
    with path.open('rb') as handle:
        raw = tomllib.load(handle)
    values = raw.get('agent', raw)
    if not isinstance(values, dict):
        raise ValueError('agent.toml must contain an [agent] table')
    allowed = values.get('allowed_models', list(AgentConfig.allowed_models))
    if not isinstance(allowed, list) or not all(isinstance(item, str) for item in allowed):
        raise ValueError('allowed_models must be a list of strings')
    selected_max_runs = int(values.get('max_runs', 1) if max_runs is None else max_runs)
    if selected_max_runs < 1 or selected_max_runs > 10:
        raise ValueError('max_runs must be between 1 and 10')
    return AgentConfig(
        autoai_base_url=_require_string(values, 'autoai_base_url'),
        dataset_id=dataset_id,
        allowed_models=tuple(allowed),
        selection_metric=str(values.get('selection_metric', 'macro_f1')),
        max_runs=selected_max_runs,
        seed=int(values.get('seed', 42)),
        split_mode=str(values.get('split_mode', 'stratified_holdout')),
        split_train=int(values.get('split_train', 8)),
        split_valid=int(values.get('split_valid', 1)),
        split_test=int(values.get('split_test', 1)),
        poll_interval_seconds=float(values.get('poll_interval_seconds', 15.0)),
        run_timeout_seconds=float(values.get('run_timeout_seconds', 3600.0)),
        llm_timeout_seconds=float(values.get('llm_timeout_seconds', 600.0)),
        temperature=float(values.get('temperature', 0.0)),
        trace_path=trace_path or Path(str(values.get('trace_path', 'outputs/agent_poc_acceptance/trace.jsonl'))),
        code_revision=str(values.get('code_revision', 'unknown')),
    )

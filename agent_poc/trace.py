"""Safe JSONL trace writer for Agent POC evidence."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import AgentDecision
from .state import ModelConfig


_FORBIDDEN_KEY_PARTS = (
    'token',
    'secret',
    'password',
    'authorization',
    'credential',
    'test',
    'artifact',
    'path',
    'filesystem',
    'prediction',
    'confusion',
    'explainability',
)
_ABSOLUTE_PATH = re.compile(r'(?:(?:[A-Za-z]:[\\/])|(?:^|\s)/)[^\s,;]+')
_URL = re.compile(r'https?://[^\s,;]+', flags=re.IGNORECASE)
_SECRET_ASSIGNMENT = re.compile(
    r'\b(?:token|secret|password|credential|authorization)\s*[:=]\s*[^\s,;]+',
    flags=re.IGNORECASE,
)


def _safe_string(value: str) -> str:
    value = re.sub(r'Bearer\s+\S+', '[redacted-auth]', value, flags=re.IGNORECASE)
    value = _SECRET_ASSIGNMENT.sub('[redacted-secret]', value)
    value = _URL.sub('[redacted-url]', value)
    value = _ABSOLUTE_PATH.sub('[redacted-path]', value)
    # Keep trace evidence focused on validation rather than any accidental
    # textual mention of a hidden test set.
    return re.sub(r'\btest\b', '[redacted-set]', value, flags=re.IGNORECASE)


def _safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if any(part in key_text.lower() for part in _FORBIDDEN_KEY_PARTS):
                continue
            cleaned[key_text] = _safe_value(item)
        return cleaned
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_value(item) for item in value]
    if isinstance(value, str):
        return _safe_string(value)
    return value


class TraceRecorder:
    def __init__(self, path: Path, *, model_cfg: ModelConfig, code_revision: str) -> None:
        self.path = Path(path)
        self.model_cfg = model_cfg
        self.code_revision = code_revision
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, **fields: Any) -> None:
        envelope = {
            'trace_schema_version': 'agent-trace-v1',
            'agent_version': '1.0.0-poc',
            'code_revision': self.code_revision,
            'model_key': self.model_cfg.key,
            'model_revision': self.model_cfg.revision,
            'event': event,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        safe = _safe_value(envelope)
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(safe, ensure_ascii=False, sort_keys=True) + '\n')

    def record_decision(self, decision: AgentDecision, *, latency_ms: float) -> None:
        self.record(
            'decision',
            latency_ms=round(float(latency_ms), 3),
            decision=decision.model_dump(mode='json'),
        )

    def record_experiment(self, experiment: dict[str, Any], *, latency_ms: float) -> None:
        self.record(
            'experiment',
            latency_ms=round(float(latency_ms), 3),
            run=experiment,
        )

    def record_feedback(self, feedback: dict[str, Any], *, latency_ms: float) -> None:
        self.record(
            'validation_feedback',
            latency_ms=round(float(latency_ms), 3),
            validation=feedback,
        )

    def record_finalize(self, result: dict[str, Any], *, latency_ms: float) -> None:
        self.record(
            'finalize',
            latency_ms=round(float(latency_ms), 3),
            run=result,
        )

    def record_error(self, error: str) -> None:
        self.record('error', error=error)


def assert_trace_safe(path: Path) -> None:
    """递归审计已有 JSONL；供测试和最终证据门禁调用。"""
    for line_number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        payload = json.loads(line)
        _assert_safe_value(payload, path=f'line {line_number}')


def _assert_safe_value(value: Any, *, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            assert not any(part in key_text for part in _FORBIDDEN_KEY_PARTS), (
                f'forbidden trace key at {path}: {key}'
            )
            _assert_safe_value(item, path=f'{path}.{key}')
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_safe_value(item, path=f'{path}[{index}]')
    elif isinstance(value, str):
        assert 'bearer ' not in value.lower(), f'auth value at {path}'
        assert not _SECRET_ASSIGNMENT.search(value), f'secret assignment at {path}'
        assert not _URL.search(value), f'url value at {path}'
        assert not _ABSOLUTE_PATH.search(value), f'absolute path at {path}'
        assert not re.search(r'\btest\b', value, flags=re.IGNORECASE), f'hidden set at {path}'

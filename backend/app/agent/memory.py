"""Safe, Principal-scoped Case Bank and Failure Ledger payloads.

This module deliberately has no storage or training imports.  It is the single
allow-list boundary between rich Run/Session objects and long-lived Agent
memory: historical identifiers, raw errors, rationale, paths, and held-out
evaluation data are never accepted by the builders below.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Iterable


MEMORY_CONTEXT_SCHEMA_VERSION = 'agent-memory-context-v1'
TERMINAL_OUTCOME_SCHEMA_VERSION = 'agent-terminal-outcome-v1'
MEMORY_RECORD_SCHEMA_VERSION = 'agent-memory-record-v1'
MAX_MEMORY_CONTEXT_ITEMS = 3

_ACTION_KEYS = ('model_type', 'normalization', 'class_balance')
_ACTION_VALUE = re.compile(r'^[a-z][a-z0-9_]{0,63}$')
_CODE_VALUE = re.compile(r'^[a-z][a-z0-9_]{0,63}$')
_TERMINAL_STATES = frozenset({'succeeded', 'failed', 'cancelled'})
_RECORD_TYPES = frozenset({'case', 'failure'})


class AgentMemoryPayloadError(ValueError):
    """A memory payload failed the strict persistence allow-list."""


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _finite_number(
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    digits: int = 6,
) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if minimum is not None and number < minimum:
        return None
    if maximum is not None and number > maximum:
        return None
    return round(number, digits)


def _positive_int(value: object) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        return None
    return int(value)


def _safe_code(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if _CODE_VALUE.fullmatch(normalized) else None


def _safe_codes(value: object, *, limit: int = 8) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    codes = {
        code
        for item in value[: max(0, limit * 2)]
        if (code := _safe_code(item)) is not None
    }
    return sorted(codes)[:limit]


def _safe_action(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise AgentMemoryPayloadError('memory action must be an object')
    action: dict[str, str] = {}
    for key in _ACTION_KEYS:
        item = value.get(key)
        if item is None and key != 'model_type':
            continue
        if not isinstance(item, str):
            raise AgentMemoryPayloadError(f'memory action {key} is invalid')
        normalized = item.strip().lower()
        if not _ACTION_VALUE.fullmatch(normalized):
            raise AgentMemoryPayloadError(f'memory action {key} is invalid')
        action[key] = normalized
    if 'model_type' not in action:
        raise AgentMemoryPayloadError('memory action requires model_type')
    return action


def build_memory_signatures(
    *,
    selection_metric: str,
    allowed_models: Iterable[str],
    evaluation_config: dict[str, Any],
    evidence_card: object,
) -> tuple[str, str]:
    """Return opaque storage-only task/evidence similarity keys.

    The evidence key uses coarse, anonymous train-only buckets.  It does not
    incorporate dataset digests, split fingerprints, labels, Sample_ID values,
    or any server path.  Neither signature is returned in Agent context.
    """
    task = {
        'problem': 'classification',
        'selection_metric': str(selection_metric),
        'allowed_models': sorted({str(item) for item in allowed_models}),
        'evaluation': {
            key: evaluation_config.get(key)
            for key in ('split_mode', 'split_train', 'split_valid', 'split_test')
        },
    }
    statistics = (
        evidence_card.get('statistics')
        if isinstance(evidence_card, dict)
        and isinstance(evidence_card.get('statistics'), dict)
        else {}
    )

    def bucket(value: object, boundaries: tuple[int, ...]) -> str:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return 'unknown'
        for boundary in boundaries:
            if value < boundary:
                return f'lt_{boundary}'
        return f'ge_{boundaries[-1]}'

    class_distribution = statistics.get('class_distribution')
    class_count = len(class_distribution) if isinstance(class_distribution, dict) else None
    repeat = statistics.get('repeat_measurements')
    evidence = {
        'scope': 'train_only',
        'data_format': (
            evidence_card.get('data_format')
            if isinstance(evidence_card, dict)
            and isinstance(evidence_card.get('data_format'), str)
            else 'unknown'
        ),
        'observation_bucket': bucket(
            statistics.get('observation_count'), (50, 100, 250, 500, 1000)
        ),
        'predictor_bucket': bucket(
            statistics.get('predictor_count'), (20, 100, 500, 2000, 10000)
        ),
        'class_count': class_count if isinstance(class_count, int) else 'unknown',
        'repeat_uniform': (
            repeat.get('uniform')
            if isinstance(repeat, dict) and isinstance(repeat.get('uniform'), bool)
            else None
        ),
        'risk_codes': _safe_codes(statistics.get('risk_codes')),
    }
    return _digest(task), _digest(evidence)


def build_terminal_outcome(
    *,
    state: str,
    selection_metric: str,
    action: dict[str, Any],
    validation_score: object = None,
    validation_std: object = None,
    diagnosis: object = None,
    duration_seconds: object = None,
    model_fit_count: object = None,
) -> dict[str, Any]:
    """Build the only terminal payload permitted to reach persistence."""
    normalized_state = str(state).strip().lower()
    if normalized_state not in _TERMINAL_STATES:
        raise AgentMemoryPayloadError('outcome state is not terminal')
    metric = _safe_code(selection_metric)
    if metric is None:
        raise AgentMemoryPayloadError('selection metric is invalid')
    normalized_diagnosis = diagnosis if isinstance(diagnosis, dict) else {}
    result: dict[str, Any] = {
        'diagnosis_status': _safe_code(normalized_diagnosis.get('status')) or 'unavailable',
        'diagnosis_category': (
            _safe_code(normalized_diagnosis.get('primary_category')) or 'none'
        ),
        'symptom_codes': _safe_codes(normalized_diagnosis.get('symptom_codes')),
        'allowed_action_ids': _safe_codes(
            normalized_diagnosis.get('allowed_action_ids')
        ),
    }
    optional_values = {
        'validation_score': _finite_number(
            validation_score, minimum=0.0, maximum=1.0
        ),
        'validation_std': _finite_number(
            validation_std, minimum=0.0, maximum=1.0
        ),
        'generalization_gap': _finite_number(
            normalized_diagnosis.get('generalization_gap'),
            minimum=-1.0,
            maximum=1.0,
        ),
        'duration_seconds': _finite_number(
            duration_seconds, minimum=0.0, maximum=31_536_000.0, digits=3
        ),
        'model_fit_count': _positive_int(model_fit_count),
    }
    result.update({key: value for key, value in optional_values.items() if value is not None})
    return {
        'schema_version': TERMINAL_OUTCOME_SCHEMA_VERSION,
        'state': normalized_state,
        'selection_metric': metric,
        'effective_action': _safe_action(action),
        'result': result,
    }


def normalize_terminal_outcome(payload: object) -> dict[str, Any]:
    """Rebuild an outcome through the allow-list before every DB write."""
    if not isinstance(payload, dict):
        raise AgentMemoryPayloadError('terminal outcome must be an object')
    result = payload.get('result')
    result = result if isinstance(result, dict) else {}
    diagnosis = {
        'status': result.get('diagnosis_status'),
        'primary_category': result.get('diagnosis_category'),
        'symptom_codes': result.get('symptom_codes'),
        'allowed_action_ids': result.get('allowed_action_ids'),
        'generalization_gap': result.get('generalization_gap'),
    }
    return build_terminal_outcome(
        state=str(payload.get('state') or ''),
        selection_metric=str(payload.get('selection_metric') or ''),
        action=payload.get('effective_action') or {},
        validation_score=result.get('validation_score'),
        validation_std=result.get('validation_std'),
        diagnosis=diagnosis,
        duration_seconds=result.get('duration_seconds'),
        model_fit_count=result.get('model_fit_count'),
    )


def build_memory_record(
    record_type: str,
    terminal_outcome: object,
) -> dict[str, Any]:
    """Project a safe terminal outcome into a reusable case or failure lesson."""
    normalized_type = str(record_type).strip().lower()
    if normalized_type not in _RECORD_TYPES:
        raise AgentMemoryPayloadError('memory record type is invalid')
    outcome = normalize_terminal_outcome(terminal_outcome)
    required_state = 'succeeded' if normalized_type == 'case' else 'failed'
    if outcome['state'] != required_state:
        raise AgentMemoryPayloadError(
            f'{normalized_type} memory requires {required_state} outcome'
        )
    result = dict(outcome['result'])
    reusable_result = {
        key: result[key]
        for key in (
            'validation_score',
            'validation_std',
            'generalization_gap',
            'diagnosis_category',
            'symptom_codes',
            'allowed_action_ids',
            'duration_seconds',
            'model_fit_count',
        )
        if key in result
    }
    return {
        'schema_version': MEMORY_RECORD_SCHEMA_VERSION,
        'record_type': normalized_type,
        'selection_metric': outcome['selection_metric'],
        'effective_action': dict(outcome['effective_action']),
        'outcome': reusable_result,
    }


def normalize_memory_record(payload: object) -> dict[str, Any]:
    """Rebuild a persisted record, discarding all non-allow-listed keys."""
    if not isinstance(payload, dict):
        raise AgentMemoryPayloadError('memory record must be an object')
    outcome = payload.get('outcome')
    outcome = outcome if isinstance(outcome, dict) else {}
    record_type = str(payload.get('record_type') or '')
    state = 'succeeded' if record_type == 'case' else 'failed'
    terminal = build_terminal_outcome(
        state=state,
        selection_metric=str(payload.get('selection_metric') or ''),
        action=payload.get('effective_action') or {},
        validation_score=outcome.get('validation_score'),
        validation_std=outcome.get('validation_std'),
        diagnosis={
            'status': 'ready',
            'primary_category': outcome.get('diagnosis_category'),
            'symptom_codes': outcome.get('symptom_codes'),
            'allowed_action_ids': outcome.get('allowed_action_ids'),
            'generalization_gap': outcome.get('generalization_gap'),
        },
        duration_seconds=outcome.get('duration_seconds'),
        model_fit_count=outcome.get('model_fit_count'),
    )
    return build_memory_record(record_type, terminal)


def freeze_memory_context(
    records: Iterable[object],
    *,
    max_items: int = MAX_MEMORY_CONTEXT_ITEMS,
) -> dict[str, Any]:
    """Freeze at most three normalized records into Agent-visible context."""
    limit = max(0, min(int(max_items), MAX_MEMORY_CONTEXT_ITEMS))
    normalized: list[dict[str, Any]] = []
    for record in records:
        if len(normalized) >= limit:
            break
        try:
            normalized.append(normalize_memory_record(record))
        except AgentMemoryPayloadError:
            # A malformed legacy row is ignored rather than exposed or allowed
            # to make an otherwise valid Session unavailable.
            continue
    return {
        'schema_version': MEMORY_CONTEXT_SCHEMA_VERSION,
        'status': 'ready' if normalized else 'empty',
        'entries': normalized,
    }

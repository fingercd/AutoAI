"""Prompt construction and observation redaction for the LLM boundary."""

from __future__ import annotations

import json
import re
from typing import Any

from .schemas import decision_json_schema
from .state import AgentConfig


_FORBIDDEN_KEY_PARTS = (
    'test',
    'artifact',
    'path',
    'token',
    'secret',
    'password',
    'authorization',
    'credential',
    'filesystem',
)
_ABSOLUTE_PATH = re.compile(r'(?:(?:[A-Za-z]:[\\/])|(?:^|\s)/)[^\s,;]+')


def _redact_string(value: str) -> str:
    value = re.sub(r'Bearer\s+\S+', '[redacted-auth]', value, flags=re.IGNORECASE)
    return _ABSOLUTE_PATH.sub('[redacted-path]', value)


def safe_observation(value: Any) -> Any:
    """递归过滤 Agent 不应看到的字段；不信任服务端以外的原始反馈。"""
    if isinstance(value, dict):
        return {
            str(key): safe_observation(item)
            for key, item in value.items()
            if not any(part in str(key).lower() for part in _FORBIDDEN_KEY_PARTS)
        }
    if isinstance(value, list):
        return [safe_observation(item) for item in value]
    if isinstance(value, str):
        return _redact_string(value)
    return value


def build_messages(
    cfg: AgentConfig,
    observation: dict[str, Any],
    *,
    allowed_decisions: tuple[str, ...] = (
        'RUN_EXPERIMENT',
        'FINALIZE',
        'REQUEST_HUMAN',
    ),
    selected_run_ids: tuple[str, ...] = (),
    proposal_recipes: tuple[dict[str, str], ...] = (),
) -> list[dict[str, str]]:
    schema = json.dumps(
        decision_json_schema(
            allowed_decisions=allowed_decisions,
            allowed_models=cfg.allowed_models,
            selected_run_ids=selected_run_ids,
            proposal_recipes=proposal_recipes,
        ),
        ensure_ascii=False,
        sort_keys=True,
    )
    visible_config = {
        'selection_metric': cfg.selection_metric,
        'allowed_models': list(cfg.allowed_models),
        'max_runs': cfg.max_runs,
        'seed': cfg.seed,
        'evaluation': {
            'split_mode': cfg.split_mode,
            'split_train': cfg.split_train,
            'split_valid': cfg.split_valid,
            'split_test': cfg.split_test,
        },
    }
    return [
        {
            'role': 'system',
            'content': (
                'You are the AutoAI Agent V1 experiment controller. '
                'Only use the supplied validation feedback. Never request or infer '
                'test-set results, artifacts, server paths, credentials, or filesystem data. '
                'Python enforces budget and session state. Return exactly one JSON object '
                'matching this schema and no markdown:\n' + schema
            ),
        },
        {
            'role': 'user',
            'content': json.dumps(
                {
                    'locked_agent_config': visible_config,
                    'observation': safe_observation(observation),
                    'instruction': (
                        'An empty experiments list and a null best_run_id are normal at the '
                        'start of a session; when a remaining run is available, choose '
                        'RUN_EXPERIMENT with one allowed, new effective configuration. '
                        'Choose RUN_EXPERIMENT only when a remaining run is available and '
                        'the proposed effective configuration is new. Choose FINALIZE only '
                        'for a known succeeded experiment with usable validation. When '
                        'choosing FINALIZE, copy selected_run_id character-for-character '
                        'from one observed experiment or best_run_id; never shorten, merge, '
                        'or synthesize a run id. If experiments already exist, never repeat '
                        'an effective configuration. When proposal recipes are supplied, '
                        'copy proposal_id and all canonical fields exactly. Prefer svm or '
                        'random_forest only when that model is present in allowed_models. '
                        'When allowed_models contains only logistic_regression, choose '
                        'logistic_regression. Otherwise '
                        'Use one short rationale/reason label from the schema enum. '
                        'choose REQUEST_HUMAN; do not request human input solely because no '
                        'experiment has been run yet.'
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]


def build_repair_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    repaired = list(messages)
    repaired.append(
        {
            'role': 'user',
            'content': (
                'Your previous response was not valid for the required schema. '
                'Repair it once: output only a single strict JSON object with no code fence, '
                'no explanation, no extra keys, and one of RUN_EXPERIMENT, FINALIZE, '
                'REQUEST_HUMAN.'
            ),
        }
    )
    return repaired

"""OpenAI-compatible LLM client with one structured-output repair attempt."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from ..prompts import build_repair_messages
from ..schemas import (
    decision_json_schema,
    DECISION_ADAPTER,
    AgentDecision,
    DecisionParseError,
    RequestHumanDecision,
)
from ..state import ModelConfig


class LLMClient:
    """The business layer receives a registry entry, never a hard-coded URL."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        *,
        temperature: float = 0.0,
        timeout_seconds: float = 120.0,
        client: Any | None = None,
    ) -> None:
        self.model_cfg = model_cfg
        self.temperature = temperature
        self.client = client or OpenAI(
            api_key='EMPTY',
            base_url=model_cfg.base_url,
            timeout=timeout_seconds,
            max_retries=0,
        )
        self.call_count = 0
        self.repair_count = 0
        # Token accounting is kept outside the redacted JSONL trace.  The
        # acceptance matrix needs real server-side usage evidence, while the
        # trace contract deliberately forbids token fields.
        self.usage_records: list[dict[str, int]] = []

    @staticmethod
    def _content(response: Any) -> str:
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise DecisionParseError('LLM response did not contain a chat message') from exc
        if not isinstance(content, str) or not content.strip():
            raise DecisionParseError('LLM returned empty content')
        return content

    def _complete(
        self,
        messages: list[dict[str, str]],
        *,
        decision_schema: dict[str, object] | None = None,
    ) -> str:
        self.call_count += 1
        response = self.client.chat.completions.create(
            model=self.model_cfg.served_model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=256,
            extra_body={
                'structured_outputs': {
                    'json': decision_schema or decision_json_schema(),
                },
                'chat_template_kwargs': {
                    # The Qwen3.x chat template otherwise spends the whole
                    # budget in the hidden reasoning channel and returns an
                    # empty ``content`` field.  Decisions are already guarded
                    # by the strict schema and Python policy loop, so this
                    # channel must stay disabled for the POC.
                    'enable_thinking': False,
                },
            },
        )
        usage = getattr(response, 'usage', None)
        usage_record: dict[str, int] = {}
        for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            value = getattr(usage, field, None) if usage is not None else None
            if isinstance(value, int):
                usage_record[field] = value
        self.usage_records.append(usage_record)
        return self._content(response)

    @staticmethod
    def _parse(content: str) -> AgentDecision:
        try:
            return DECISION_ADAPTER.validate_json(content)
        except Exception as exc:
            raise DecisionParseError('LLM decision failed Pydantic validation') from exc

    def decide(
        self,
        messages: list[dict[str, str]],
        *,
        decision_schema: dict[str, object] | None = None,
    ) -> AgentDecision:
        """Validate the first answer, repair exactly once, then stop safely."""
        try:
            return self._parse(self._complete(messages, decision_schema=decision_schema))
        except DecisionParseError:
            self.repair_count += 1
            try:
                return self._parse(self._complete(
                    build_repair_messages(messages),
                    decision_schema=decision_schema,
                ))
            except DecisionParseError as exc:
                return RequestHumanDecision(
                    decision='REQUEST_HUMAN',
                    reason=f'LLM structured decision invalid after one repair: {exc}',
                )

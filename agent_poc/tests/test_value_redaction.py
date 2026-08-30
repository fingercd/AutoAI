from __future__ import annotations

import json

from agent_poc.prompts import safe_observation
from agent_poc.state import ModelConfig
from agent_poc.trace import TraceRecorder, assert_trace_safe


MALICIOUS = (
    'Bearer abc token=one password=two credential=three '
    'https://internal.example /users/private C:\\private\\secret.txt'
)


def test_prompt_and_trace_redact_sensitive_string_values(tmp_path):
    visible = safe_observation({'rationale': MALICIOUS})
    rendered = json.dumps(visible).lower()
    for forbidden in (
        'bearer abc', 'token=one', 'password=two', 'credential=three',
        'internal.example', '/users/private', 'c:\\private',
    ):
        assert forbidden not in rendered

    trace_path = tmp_path / 'trace.jsonl'
    recorder = TraceRecorder(
        trace_path,
        model_cfg=ModelConfig(
            key='model',
            base_url='http://127.0.0.1:1/v1',
            served_model_name='model',
            model_path='/private/model',
        ),
        code_revision='test',
    )
    recorder.record('error', error=MALICIOUS)
    assert_trace_safe(trace_path)
    trace_text = trace_path.read_text(encoding='utf-8').lower()
    for forbidden in ('token=one', 'password=two', 'internal.example', '/users/private'):
        assert forbidden not in trace_text

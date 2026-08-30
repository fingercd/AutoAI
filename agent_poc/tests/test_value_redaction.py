from __future__ import annotations

import json

from agent_poc.prompts import safe_observation
from agent_poc.state import ModelConfig
from agent_poc.trace import TraceRecorder, assert_trace_safe


MALICIOUS = (
    'Bearer abc token=one password=two credential=three '
    'https://internal.example /users/private C:\\private\\secret.txt '
    'path=/users/fotile/assigned-secret '
    'Sample_ID=patient-42，split rejected'
)


def test_prompt_and_trace_redact_sensitive_string_values(tmp_path):
    visible = safe_observation({'rationale': MALICIOUS})
    rendered = json.dumps(visible).lower()
    for forbidden in (
        'bearer abc', 'token=one', 'password=two', 'credential=three',
        'internal.example', '/users/private', 'c:\\private', 'patient-42',
        'assigned-secret',
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
    recorder.record(
        'error',
        error=MALICIOUS,
        current_fold_sample_id='held-out-secret',
    )
    assert_trace_safe(trace_path)
    trace_text = trace_path.read_text(encoding='utf-8').lower()
    for forbidden in (
        'token=one', 'password=two', 'internal.example', '/users/private',
        'patient-42', 'held-out-secret', 'sample_id', 'assigned-secret',
    ):
        assert forbidden not in trace_text


def test_prompt_boundary_removes_sample_identifiers_recursively():
    visible = safe_observation({
        'progress': {'current_fold_sample_id': 'held-out-secret'},
        'error': 'Sample_ID=patient-42 留作测试后训练集缺少类别',
        'experiments': [{
            'run_id': 'safe-run-id',
            'validation_group_count': 12,
        }],
    })
    rendered = json.dumps(visible).lower()
    assert 'sample_id' not in rendered
    assert 'held-out-secret' not in rendered
    assert 'patient-42' not in rendered
    assert visible['experiments'][0]['run_id'] == 'safe-run-id'

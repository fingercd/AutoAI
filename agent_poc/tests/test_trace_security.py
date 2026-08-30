from __future__ import annotations

from agent_poc.state import ModelConfig
from agent_poc.trace import TraceRecorder, assert_trace_safe


def test_trace_drops_forbidden_keys_tokens_and_absolute_paths(tmp_path):
    model = ModelConfig(
        key='qwen35_9b',
        base_url='http://127.0.0.1:8101/v1',
        served_model_name='qwen35_9b',
        model_path='/server-only/model',
    )
    path = tmp_path / 'trace.jsonl'
    trace = TraceRecorder(path, model_cfg=model, code_revision='rev-1')
    trace.record(
        'validation_feedback',
        validation={'metrics': {'macro_f1': 0.8}, 'test_metrics': {'macro_f1': 0.1}},
        artifact_path='/users/fotile/secret.bin',
        authorization='Bearer secret-token',
        run={'run_id': 'run-1', 'rationale': 'hide test details'},
    )
    assert_trace_safe(path)
    raw = path.read_text(encoding='utf-8').lower()
    assert 'test_metrics' not in raw
    assert 'artifact_path' not in raw
    assert 'secret-token' not in raw
    assert '/users/fotile' not in raw
    assert 'hide test details' not in raw

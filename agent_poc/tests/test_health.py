from __future__ import annotations

from types import SimpleNamespace

from agent_poc.health import probe_agent_stack
from agent_poc.state import ModelConfig


def _model(key='qwen35_9b') -> ModelConfig:
    return ModelConfig(
        key=key,
        base_url='http://127.0.0.1:8101/v1',
        served_model_name=key,
        model_path='/must/not/appear',
    )


def _model_report(key):
    return {
        'model_key': key,
        'served_model_name': key,
        'models_api': {'ok': True, 'advertised': True},
        'inference': {'attempted': False, 'ok': None},
    }


def _autoai(models, *, runtime='agent-runtime-health-v1', agent='agent-session-v1'):
    return SimpleNamespace(
        health=lambda: {
            'status': 'ok',
            'deployment_mode': 'server',
            'worker': {'available': True, 'compatible': True},
        },
        agent_health=lambda **_: {
            'status': 'ready',
            'contract_version': runtime,
            'agent_contract': agent,
            'models': models,
        },
    )


def test_stack_probe_requires_worker_contracts_and_exact_model_set():
    report = probe_agent_stack(
        _autoai([_model_report('qwen35_9b')]),
        [_model()],
    )
    assert report['status'] == 'ready'
    assert all(report['contract_checks'].values())


def test_stack_probe_rejects_wrong_contract_or_model_set():
    report = probe_agent_stack(
        _autoai(
            [_model_report('different')],
            runtime='wrong-runtime',
            agent='wrong-agent',
        ),
        [_model('expected_a'), _model('expected_b')],
    )
    assert report['status'] == 'degraded'
    assert report['contract_checks'] == {
        'runtime_contract': False,
        'agent_contract': False,
        'model_keys_match': False,
        'model_keys_unique': True,
    }


def test_stack_probe_rejects_duplicate_returned_model_keys():
    report = probe_agent_stack(
        _autoai([_model_report('qwen35_9b'), _model_report('qwen35_9b')]),
        [_model()],
    )
    assert report['status'] == 'degraded'
    assert report['contract_checks']['model_keys_unique'] is False

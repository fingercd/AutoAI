from __future__ import annotations

import pytest

from agent_poc.state import load_agent_config


def test_agent_config_serializes_session_module_flags(tmp_path):
    config = tmp_path / 'agent.toml'
    config.write_text(
        '[agent]\n'
        'autoai_base_url = "http://127.0.0.1:8000"\n'
        '[agent.modules]\n'
        'evidence_card = true\n'
        'budget_control = true\n'
        '[agent.context]\n'
        'source_role = "domain"\n'
        'case_write = true\n',
        encoding='utf-8',
    )
    loaded = load_agent_config(config, dataset_id='dataset-1')
    payload = loaded.session_payload()
    assert payload['modules']['evidence_card'] is True
    assert payload['modules']['budget_control'] is True
    assert payload['context_policy'] == {
        'source_role': 'domain',
        'case_write': True,
    }


def test_default_agent_config_keeps_legacy_v1_payload_shape(tmp_path):
    config = tmp_path / 'agent.toml'
    config.write_text(
        '[agent]\n'
        'autoai_base_url = "http://127.0.0.1:8000"\n',
        encoding='utf-8',
    )
    payload = load_agent_config(config, dataset_id='dataset-1').session_payload()
    assert 'modules' not in payload
    assert 'context_policy' not in payload


def test_agent_config_rejects_unknown_module_flag(tmp_path):
    config = tmp_path / 'agent.toml'
    config.write_text(
        '[agent]\n'
        'autoai_base_url = "http://127.0.0.1:8000"\n'
        '[agent.modules]\n'
        'invented = true\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='unknown agent module flags'):
        load_agent_config(config, dataset_id='dataset-1')


def test_agent_config_rejects_unknown_context_field(tmp_path):
    config = tmp_path / 'agent.toml'
    config.write_text(
        '[agent]\n'
        'autoai_base_url = "http://127.0.0.1:8000"\n'
        '[agent.context]\n'
        'source_roel = "benchmark"\n'
        'case_write = true\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='unknown agent context fields'):
        load_agent_config(config, dataset_id='dataset-1')

from __future__ import annotations

import pytest

from agent_poc.state import action_config_hash, load_agent_config


def test_agent_config_serializes_session_module_flags(tmp_path):
    config = tmp_path / 'agent.toml'
    config.write_text(
        '[agent]\n'
        'autoai_base_url = "http://127.0.0.1:8000"\n'
        '[agent.modules]\n'
        'evidence_card = true\n'
        'bounded_hpo = true\n'
        'fail_fast_guard = true\n'
        'budget_control = true\n'
        '[agent.context]\n'
        'source_role = "domain"\n'
        'case_write = true\n'
        '[agent.budget]\n'
        'max_model_fits = 20\n'
        'max_llm_calls = 3\n'
        'max_api_calls = 100\n'
        'max_wall_clock_seconds = 300\n'
        'max_retry_attempts = 2\n',
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
    assert payload['budget'] == {
        'max_model_fits': 20,
        'max_llm_calls': 3,
        'max_api_calls': 100,
        'max_wall_clock_seconds': 300,
        'max_retry_attempts': 2,
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


def test_budget_control_requires_all_five_strict_fields(tmp_path):
    config = tmp_path / 'agent.toml'
    config.write_text(
        '[agent]\n'
        'autoai_base_url = "http://127.0.0.1:8000"\n'
        '[agent.modules]\n'
        'bounded_hpo = true\n'
        'fail_fast_guard = true\n'
        'budget_control = true\n'
        '[agent.budget]\n'
        'max_model_fits = 20\n'
        'max_llm_calls = 3\n'
        'max_api_calls = 100\n'
        'max_wall_clock_seconds = 300\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='must contain exactly'):
        load_agent_config(config, dataset_id='dataset-1')


@pytest.mark.parametrize('value', ['true', '1.5', '10001'])
def test_budget_control_rejects_non_integer_or_out_of_range_api_limit(
    tmp_path, value
):
    config = tmp_path / 'agent.toml'
    config.write_text(
        '[agent]\n'
        'autoai_base_url = "http://127.0.0.1:8000"\n'
        '[agent.modules]\n'
        'bounded_hpo = true\n'
        'fail_fast_guard = true\n'
        'budget_control = true\n'
        '[agent.budget]\n'
        'max_model_fits = 20\n'
        'max_llm_calls = 3\n'
        f'max_api_calls = {value}\n'
        'max_wall_clock_seconds = 300\n'
        'max_retry_attempts = 2\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError):
        load_agent_config(config, dataset_id='dataset-1')


def test_disabled_budget_control_rejects_budget_table(tmp_path):
    config = tmp_path / 'agent.toml'
    config.write_text(
        '[agent]\n'
        'autoai_base_url = "http://127.0.0.1:8000"\n'
        '[agent.budget]\n'
        'max_model_fits = 20\n'
        'max_llm_calls = 3\n'
        'max_api_calls = 100\n'
        'max_wall_clock_seconds = 300\n'
        'max_retry_attempts = 2\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='requires agent.modules.budget_control'):
        load_agent_config(config, dataset_id='dataset-1')


def test_budget_control_requires_bounded_hpo_and_fail_fast_guard(tmp_path):
    config = tmp_path / 'agent.toml'
    config.write_text(
        '[agent]\n'
        'autoai_base_url = "http://127.0.0.1:8000"\n'
        '[agent.modules]\n'
        'budget_control = true\n'
        '[agent.budget]\n'
        'max_model_fits = 20\n'
        'max_llm_calls = 3\n'
        'max_api_calls = 100\n'
        'max_wall_clock_seconds = 300\n'
        'max_retry_attempts = 2\n',
        encoding='utf-8',
    )
    with pytest.raises(
        ValueError,
        match='requires bounded_hpo and fail_fast_guard',
    ):
        load_agent_config(config, dataset_id='dataset-1')


@pytest.mark.parametrize(
    'line',
    [
        'max_runs = 1.9',
        'max_runs = true',
        'split_train = 8.9',
        'seed = true',
        'selection_metric = "accuracy"',
        'split_mode = "leave_one_sample_id_cv"',
        'allowed_models = ["cnn1d"]',
        'selection_metic = "macro_f1"',
        'poll_interval_seconds = nan',
        'poll_interval_seconds = -1.0',
        'run_timeout_seconds = inf',
        'llm_timeout_seconds = -inf',
        'temperature = nan',
        'temperature = 2.1',
        'seed = -1',
    ],
)
def test_agent_config_rejects_ambiguous_or_unsupported_values(tmp_path, line):
    config = tmp_path / 'agent.toml'
    config.write_text(
        '[agent]\n'
        'autoai_base_url = "http://127.0.0.1:8000"\n'
        f'{line}\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError):
        load_agent_config(config, dataset_id='dataset-1')


def test_effective_action_hash_ignores_provenance_only_fields():
    first = action_config_hash({
        'model_type': 'svm',
        'normalization': 'zscore',
        'class_balance': 'none',
        'proposal_id': 'p_0123456789abcdef',
        'parent_run_id': 'run-a',
        'rationale': 'baseline',
    })
    second = action_config_hash({
        'model_type': 'svm',
        'normalization': 'zscore',
        'class_balance': 'none',
        'proposal_id': 'p_fedcba9876543210',
        'parent_run_id': 'run-b',
        'rationale': 'retry',
    })
    assert first == second

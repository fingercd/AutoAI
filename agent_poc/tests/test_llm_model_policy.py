import pytest
from agent_poc.llm_policy import approved_models, approved_model_key, LLMModelNotApproved
from agent_poc.orchestration.llm import LLMConfig
from agent_poc.orchestration.runtime import main


def test_actual_three_models_are_the_entire_default_catalog():
    assert set(approved_models()) == {'qwen35_9b', 'qwen35_27b', 'qwen38_27b'}


@pytest.mark.parametrize('model', ['qwen35_9b', 'Qwen/Qwen3.5-9B',
    'Qwen3.5-27B', approved_models()['qwen38_27b']['model_path']])
def test_registered_identity_forms_are_accepted(model):
    assert approved_model_key(model) is not None
    config = LLMConfig('http://localhost:8101/v1', model)
    assert 'model_policy_exception' not in config.public_config()


@pytest.mark.parametrize('model', ['qwen3-4b', 'Qwen/Qwen3-4B',
    '/users/fotile/models/Qwen3-4B', 'qwen3-8b', 'qwen38_9b', '/unregistered/Qwen3.5-9B'])
def test_unapproved_backbones_are_rejected_without_explicit_authorization(model):
    with pytest.raises(LLMModelNotApproved):
        LLMConfig('http://localhost:18762/v1', model)


def test_explicit_exception_is_visible_in_frozen_config():
    config = LLMConfig('http://localhost/v1', 'user-selected-model', allow_unapproved_model=True)
    assert config.public_config()['model_policy_exception'] == 'explicit_user_authorization'


def test_cli_rejects_4b_before_creating_task_storage(tmp_path, capsys):
    root = tmp_path/'checkpoint'
    result = main(['start', '--dataset-id', 'unused', '--storage', str(root),
        '--principal-scope', 'local', '--llm-base-url', 'http://localhost:18762/v1',
        '--llm-model', 'qwen3-4b'], environ={})
    assert result == 2
    assert 'llm_model_not_approved' in capsys.readouterr().err
    assert not root.exists()


def test_cli_exception_requires_typed_model_not_only_an_environment_value(tmp_path, capsys):
    result = main(['start', '--dataset-id', 'unused', '--allow-unapproved-llm'], environ={
        'AUTOAI_LLM_MODEL':'other', 'AUTOAI_LLM_BASE_URL':'http://localhost/v1',
        'AUTOAI_PRINCIPAL_SCOPE':'local'})
    assert result == 2
    assert 'llm_model_override_requires_explicit_argument' in capsys.readouterr().err

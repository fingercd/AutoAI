import os

import pytest


@pytest.fixture
def budget_config(tmp_path):
    """Offline real tokenizer; acceptance can select the deployed Qwen tokenizer."""
    path = os.environ.get('AUTOAI_TEST_TOKENIZER_PATH')
    if path is None:
        tokenizers = pytest.importorskip('tokenizers')
        transformers = pytest.importorskip('transformers')
        tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
        local = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]')
        local.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}{% if tools is defined %}{{ tools | tojson }}{% endif %}{% if add_generation_prompt %}assistant: {% endif %}"
        path = str(tmp_path / 'tokenizer')
        local.save_pretrained(path)
    return dict(tokenizer_path=path, context_window=32768)

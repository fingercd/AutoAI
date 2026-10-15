"""User-approved inference identities for new Agent experiments."""
from functools import lru_cache
from pathlib import Path
import tomllib

REGISTRY = Path(__file__).parent / 'config' / 'models.toml'


class LLMModelNotApproved(ValueError):
    def __init__(self):
        super().__init__('llm_model_not_approved: select a registered model; '
                         'other models require explicit user authorization')


@lru_cache(maxsize=1)
def approved_models():
    with REGISTRY.open('rb') as handle:
        return tomllib.load(handle)['models']


def approved_model_key(model):
    name = model.replace('\\', '/').rstrip('/').casefold()
    for key, item in approved_models().items():
        aliases = (key, item['served_model_name'], item['display_name'],
                   'Qwen/' + item['display_name'], item['model_path'], item['original_model_path'])
        if name in {alias.casefold() for alias in aliases}:
            return key
    return None


def require_approved_model(model, *, user_authorized=False):
    if type(user_authorized) is not bool:
        raise ValueError('LLM model authorization must be an explicit bool')
    key = approved_model_key(model)
    if key is None and not user_authorized:
        raise LLMModelNotApproved()
    return key

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.registry import SUPPORTED_MODEL_TYPES


def test_model_catalog_matches_supported_training_models():
    response = TestClient(app).get('/api/models')

    assert response.status_code == 200
    models = response.json()['models']
    required = {
        'id',
        'display_name',
        'family',
        'available',
        'unavailable_reason',
        'explainability_method',
    }
    assert models
    assert {model['id'] for model in models} == SUPPORTED_MODEL_TYPES
    assert len({model['id'] for model in models}) == len(models)
    assert all(set(model) == required for model in models)


def test_missing_dscarnet_dependency_does_not_break_catalog(monkeypatch):
    from backend.app.routers import catalog

    original_find_spec = catalog.importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == 'aggmap':
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(catalog.importlib.util, 'find_spec', fake_find_spec)
    response = TestClient(app).get('/api/models')

    assert response.status_code == 200
    by_id = {model['id']: model for model in response.json()['models']}
    assert by_id['dscarnet']['available'] is False
    assert 'aggmap' in by_id['dscarnet']['unavailable_reason']
    assert all(model['available'] for model_id, model in by_id.items() if model_id != 'dscarnet')


def test_model_catalog_initialization_waits_for_modules_and_preserves_fallback():
    content = Path('static/index.html').read_text(encoding='utf-8')
    fallback = content.split('function renderModelCatalogFallback', 1)[1].split('async function loadModelCatalog', 1)[0]

    assert 'window.addEventListener("autoai:modules-ready", initializeApiBackedUi' in content
    assert 'window.AutoAIRequest && window.AutoAITrainingStore' in content
    assert 'api("/api/models")' in content
    assert 'select.replaceChildren()' not in fallback
    assert '继续使用页面内置的稳定模型' in fallback
    assert '<option value="pls_da">PLS-DA</option>' in content
    assert '<option value="dscarnet">DSCARNet</option>' in content


def test_training_store_announces_module_readiness():
    content = Path('static/js/training-store.js').read_text(encoding='utf-8')

    assert "window.dispatchEvent(new CustomEvent('autoai:modules-ready'))" in content

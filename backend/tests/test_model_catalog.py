from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.registry import TARGET_MODEL_TYPES


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
    assert {model['id'] for model in models} == TARGET_MODEL_TYPES
    assert len({model['id'] for model in models}) == len(models)
    assert all(set(model) == required for model in models)


def test_retired_dependency_is_not_probed(monkeypatch):
    from backend.app import model_catalog
    original = model_catalog.importlib.util.find_spec
    def probe(name, *args, **kwargs):
        assert name != 'aggmap'
        return original(name, *args, **kwargs)
    monkeypatch.setattr(model_catalog.importlib.util, 'find_spec', probe)
    response = TestClient(app).get('/api/models')
    assert response.status_code == 200
    assert 'dscarnet' not in {m['id'] for m in response.json()['models']}


def test_catalog_does_not_import_model_runtime(monkeypatch):
    from backend.app.routers import catalog
    monkeypatch.setattr(catalog.importlib, 'import_module',
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('runtime import')))
    response = TestClient(app).get('/api/models')
    assert response.status_code == 200
    assert 'dscarnet' not in {m['id'] for m in response.json()['models']}


def test_model_catalog_initialization_has_module_and_dom_ready_paths():
    content = Path('static/index.html').read_text(encoding='utf-8')
    fallback = content.split('function renderModelCatalogFallback', 1)[1].split('async function loadModelCatalog', 1)[0]

    assert 'window.addEventListener("specautoai:modules-ready", initializeApiBackedUi' in content
    assert 'apiBackedUiInitialized' in content
    assert 'window.setTimeout(initializeApiBackedUi, 1000)' in content
    assert 'apiWithTimeout("/api/models", 30000)' in content
    assert 'window.SpecAutoAIRequest && window.SpecAutoAITrainingStore' in content
    assert 'select.replaceChildren()' in fallback
    assert 'select.disabled = true' in fallback
    assert '<option value="pls_da">PLS-DA</option>' not in content
    assert '<option value="dscarnet">DSCARNet</option>' not in content


def test_model_option_visibility_tolerates_removed_optional_blocks():
    content = Path('static/index.html').read_text(encoding='utf-8')

    assert '$("trainTimeBlock")?.classList.toggle("hidden", traditional)' in content


def test_training_store_announces_module_readiness():
    content = Path('static/js/training-store.js').read_text(encoding='utf-8')

    assert "window.dispatchEvent(new CustomEvent('specautoai:modules-ready'))" in content

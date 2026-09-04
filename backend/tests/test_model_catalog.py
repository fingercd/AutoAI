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
        'ui_visible',
        'visibility_reason',
    }
    assert models
    assert {model['id'] for model in models} == TARGET_MODEL_TYPES
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
    assert all(model['available'] for model_id, model in by_id.items() if model_id not in {'dscarnet', 'cnn_mamba1d'})
    assert by_id['cnn_mamba1d']['available'] is False


def test_dscarnet_catalog_does_not_import_heavy_aggmap_runtime(monkeypatch):
    from backend.app import dscarnet_mapping
    from backend.app.routers import catalog

    original_find_spec = catalog.importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == 'aggmap':
            return object()
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(catalog.importlib.util, 'find_spec', fake_find_spec)
    monkeypatch.setattr(
        catalog.importlib,
        'import_module',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('catalog must not import model runtimes')),
    )
    monkeypatch.setattr(
        dscarnet_mapping,
        '_load_aggmap_class',
        lambda: (_ for _ in ()).throw(AssertionError('catalog must not import AggMap')),
    )

    assert catalog._dscarnet_capability() == (True, None)


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

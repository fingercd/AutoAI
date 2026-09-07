from fastapi.testclient import TestClient


CANONICAL_MODELS = {
    "pls_da",
    "spls_da",
    "pca_lda",
    "logistic_regression",
    "svm",
    "pca_svm",
    "random_forest",
    "xgboost",
    "pca_mlp",
    "cnn1d",
    "cnn1d_se",
    "resnet1d",
    "inception1d",
    "tcn1d",
    "cnn_transformer1d",
    "cnn_mamba1d",
    "dscarnet",
}


def _catalog_models() -> list[dict[str, object]]:
    from backend.app.main import app

    response = TestClient(app).get("/api/models")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"models", "training_scheme"}
    assert payload['training_scheme']['version'] == 'word-0904'
    return payload["models"]


def test_models_catalog_exposes_all_backend_models_with_explicit_ui_visibility():
    models = _catalog_models()

    assert {item["id"] for item in models} == CANONICAL_MODELS
    assert len(models) == 17
    assert len({item["id"] for item in models}) == len(models)
    for item in models:
        assert set(item) == {
            "id",
            "display_name",
            "family",
            "available",
            "unavailable_reason",
            "ui_visible",
            "visibility_reason",
        }
        assert item["display_name"]
        assert item["family"]
        assert isinstance(item["available"], bool)
        assert isinstance(item["ui_visible"], bool)
        assert (item["visibility_reason"] is None) is item["ui_visible"]
        assert (item["unavailable_reason"] is None) is item["available"]


def test_models_catalog_degrades_optional_dependencies_without_server_error(monkeypatch):
    from backend.app.routers import catalog

    monkeypatch.setattr(catalog, "_mamba_capability", lambda: (False, "mamba test dependency missing"), raising=False)
    monkeypatch.setattr(catalog, "_dscarnet_capability", lambda: (False, "aggmap test dependency missing"), raising=False)

    by_id = {item["id"]: item for item in _catalog_models()}

    assert by_id["cnn_mamba1d"] == {
        "id": "cnn_mamba1d",
        "display_name": "CNN-Mamba",
        "family": "long_range",
        "available": False,
        "unavailable_reason": "mamba test dependency missing",
        "ui_visible": False,
        "visibility_reason": "temporarily_hidden_from_ui",
    }
    assert by_id["dscarnet"]["available"] is False
    assert by_id["dscarnet"]["unavailable_reason"] == "aggmap test dependency missing"

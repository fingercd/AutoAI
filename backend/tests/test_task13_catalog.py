from fastapi.testclient import TestClient


CANONICAL_MODELS = {
    "pls_da",
    "pca_lda",
    "logistic_regression",
    "svm",
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
}


def _catalog_models() -> list[dict[str, object]]:
    from backend.app.main import app

    response = TestClient(app).get("/api/models")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"models"}
    return payload["models"]


def test_models_catalog_exposes_exactly_fourteen_canonical_models():
    models = _catalog_models()

    assert {item["id"] for item in models} == CANONICAL_MODELS
    assert len(models) == 14
    assert len({item["id"] for item in models}) == len(models)
    for item in models:
        assert set(item) == {
            "id",
            "display_name",
            "family",
            "available",
            "unavailable_reason",
            "explainability_method",
        }
        assert item["display_name"]
        assert item["family"]
        assert isinstance(item["available"], bool)
        assert item["explainability_method"]
        assert (item["unavailable_reason"] is None) is item["available"]


def test_models_catalog_degrades_optional_dependencies_without_server_error(monkeypatch):
    from backend.app.routers import catalog

    monkeypatch.setattr(catalog, "_mamba_capability", lambda: (False, "mamba test dependency missing"), raising=False)

    by_id = {item["id"]: item for item in _catalog_models()}

    assert by_id["cnn_mamba1d"] == {
        "id": "cnn_mamba1d",
        "display_name": "CNN-Mamba",
        "family": "long_range",
        "available": False,
        "unavailable_reason": "mamba test dependency missing",
        "explainability_method": "window_occlusion_log_loss",
    }
    assert "dscarnet" not in by_id

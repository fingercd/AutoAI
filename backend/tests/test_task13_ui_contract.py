from pathlib import Path


def test_official_ui_loads_model_catalog_without_hard_coded_legacy_options():
    content = Path("static/index.html").read_text(encoding="utf-8")

    assert 'async function loadModelCatalog()' in content
    assert 'api("/api/models")' in content
    assert 'function renderModelCatalog(models' in content
    assert 'const MODEL_CATALOG_GROUPS' in content
    assert '传统模型' in content
    assert '基础深度模型' in content
    assert '卷积改进模型' in content
    assert '长距离模型' in content
    assert '二维映射模型' in content
    assert '<option value="pls_da">PLS-DA</option>' not in content
    assert '<option value="transformer1d">1D-Transformer</option>' not in content


def test_official_ui_preserves_catalog_ids_and_explains_unavailable_models():
    content = Path("static/index.html").read_text(encoding="utf-8")

    assert 'option.value = model.id' in content
    assert 'model.display_name' in content
    assert 'option.disabled = !available' in content
    assert 'model.unavailable_reason' in content
    assert 'function renderModelCatalogFallback' in content
    assert 'loadModelCatalog().catch' in content

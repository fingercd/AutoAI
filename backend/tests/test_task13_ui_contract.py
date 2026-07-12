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


def test_official_ui_renders_training_audit_for_traditional_deep_and_explainability():
    content = Path("static/index.html").read_text(encoding="utf-8")

    assert 'id="trainingAudit"' in content
    assert 'function trainingAuditFor(run)' in content
    assert 'function renderTrainingAudit(run)' in content
    assert 'traditional.best_params_by_fold' in content
    assert 'valid_balanced_accuracy' in content
    assert 'hyperparameter_search_csv' in content
    assert 'audit.deep_training' in content
    assert 'best_valid_loss' in content
    assert 'min_learning_rate' in content
    assert 'declared_method' in content
    assert 'artifact_method' in content
    assert 'importance_metric' in content


def test_official_ui_external_audit_never_renders_cv_or_fold_progress():
    content = Path("static/index.html").read_text(encoding="utf-8")

    assert 'function isExternalTestHoldout(run)' in content
    assert 'const isExternal = isExternalTestHoldout(run);' in content
    assert '独立测试集最终评估' in content
    assert 'const progressMarkup = isExternal' in content
    assert 'renderTrainingAudit(run);' in content

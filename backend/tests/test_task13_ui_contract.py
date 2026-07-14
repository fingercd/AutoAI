from pathlib import Path


def test_official_ui_loads_model_catalog_without_hard_coded_legacy_options():
    content = Path("static/index.html").read_text(encoding="utf-8")

    assert 'async function loadModelCatalog()' in content
    assert 'apiWithTimeout("/api/models", 30000)' in content
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
    assert '当前不可用模型：' not in content
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
    assert '参数选择记录' in content
    assert '相同配置已自动合并' in content
    assert 'auditParamsForModel' in content
    assert '"oob_balanced_accuracy"' in content
    assert "平均 OOB BA" in content
    assert 'audit.deep_training' not in content
    assert '声明方法' not in content
    assert '产物方法' not in content


def test_official_ui_external_audit_never_renders_cv_or_fold_progress():
    content = Path("static/index.html").read_text(encoding="utf-8")

    assert 'function isExternalTestHoldout(run)' in content
    assert 'const isExternal = isExternalTestHoldout(run);' in content
    assert '独立测试集最终评估' in content
    assert 'const progressMarkup = isExternal' in content
    assert 'renderTrainingAudit(run);' in content


def test_official_ui_exposes_dscarnet_input_modes_only_for_dscarnet():
    content = Path("static/index.html").read_text(encoding="utf-8")

    assert 'id="dscarnetInputMode"' in content
    assert '<option value="sar">SAR</option>' in content
    assert '<option value="car">CAR</option>' in content
    assert '<option value="dual">SAR + CAR</option>' in content
    assert 'dscarnet_input_mode: $("dscarnetInputMode").value' in content
    assert '$("dscarnetOptions").classList.toggle("hidden", modelType !== "dscarnet")' in content


def test_official_ui_hides_mamba_and_traditional_epoch_progress():
    content = Path("static/index.html").read_text(encoding="utf-8")

    assert 'const HIDDEN_MODEL_IDS = new Set(["cnn_mamba1d"])' in content
    assert '!HIDDEN_MODEL_IDS.has(model.id)' in content
    assert 'const epochMetric = traditionalModel' in content
    assert '训练 Epoch<strong>${run.actual_epochs || "-"}/' not in content
    assert '实际训练 Epoch' in content


def test_training_spinner_is_stable_across_polling():
    content = Path("static/index.html").read_text(encoding="utf-8")

    assert 'data-active-training-run' in content
    assert 'if (pollingRunId === requestedRunId) return;' in content
    assert 'if (currentRunId === requestedRunId) renderRun(run);' in content
    assert 'background: conic-gradient' in content


def test_official_ui_exposes_oob_random_forest_budget_and_terminal_run_delete():
    content = Path("static/index.html").read_text(encoding="utf-8")

    assert 'id="rfEstimators" type="number" value="200"' in content
    assert 'id="rfSearchIterations" type="number" value="10"' in content
    assert 'random_forest_search_iterations: Number($("rfSearchIterations").value)' in content
    assert 'id="rfMaxDepth"' not in content
    assert 'id="rfMinSamplesLeaf"' not in content
    assert 'window.deleteRun = async (runId)' in content
    assert 'method: "DELETE"' in content
    assert '删除此训练记录' in content
    assert '删除失败：${error.message}' in content
    assert '模型、指标、预测结果和其他产物都将一并删除' in content
    assert '["succeeded", "failed", "cancelled"].includes(run.state)' in content


def test_modeling_view_refetches_run_before_redrawing_hidden_canvases():
    content = Path("static/index.html").read_text(encoding="utf-8")

    assert 'if (rect.width < 1 || rect.height < 1) return;' in content
    assert 'if (viewName === "modeling" && currentRunId)' in content
    assert 'pollCurrentRun().catch((error) => window.alert(error.message));' in content
    assert 'drawSampleFeatureImportanceCanvas(data, sample);' in content
    assert 'button.addEventListener("click", () => showView(button.dataset.view));' in content


def test_official_ui_only_renders_sample_feature_importance():
    content = Path("static/index.html").read_text(encoding="utf-8")

    assert "async function renderGlobalFeatureImportance" not in content
    assert "function drawFeatureImportanceCanvas" not in content
    assert "feature-view-switch" not in content
    assert 'data-feature-view="global"' not in content
    assert 'data-feature-view="sample"' not in content
    assert "combinedLogLossView" not in content
    assert "run.feature_importance" not in content
    assert "暂无单样本重要性" in content
    assert "await renderSampleFeatureImportance(run, sampleSummary, target);" in content
    assert 'id="sampleFeatureImportanceCanvas" class="sample-feature-canvas"' in content
    assert 'id="featureSampleSelect"' in content
    assert 'const windows = source.windows || [];' in content
    assert 'span <= 1e-12) return 0;' in content
    assert 'function divergentImportanceColor' in content
    assert 'sample_occlusion_log_loss' in content
    assert 'const windows = (source?.windows || []).filter(isPositive);' in content

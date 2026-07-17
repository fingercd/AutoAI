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
    html = Path("static/index.html").read_text(encoding="utf-8")
    results = Path("static/js/run-results.js").read_text(encoding="utf-8")

    assert 'id="resultAnalysis"' in html
    assert 'function renderTrainingAudit(result)' in results
    assert 'result.analysis?.training_audit' in results
    assert 'const traditional = audit.traditional;' in results
    assert 'const deep = audit.deep_training;' in results
    assert 'traditional.best_params_by_fold' in results
    assert 'traditional.valid_balanced_accuracy' in results
    assert '完整参数搜索 CSV 如已生成' in results
    assert '最佳验证 Loss' in results
    assert '实际训练 Epoch' in results
    assert 'function renderTrainingAudit(run)' not in html


def test_official_ui_external_audit_never_renders_cv_or_fold_progress():
    html = Path("static/index.html").read_text(encoding="utf-8")
    results = Path("static/js/run-results.js").read_text(encoding="utf-8")

    assert "external_test_holdout: '独立测试集最终评估'" in results
    assert "leave_one_sample_id_cv: '按 Sample_ID 留一交叉验证'" in results
    assert "pooled_oof: '合并 OOF 预测'" in results
    assert 'result.evaluation?.primary_aggregation' in results
    assert 'result.metrics?.splits' in results
    assert 'const progressMarkup = isExternal' not in html


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
    results = Path("static/js/run-results.js").read_text(encoding="utf-8")

    assert 'const HIDDEN_MODEL_IDS = new Set(["cnn_mamba1d"])' in content
    assert '!HIDDEN_MODEL_IDS.has(model.id)' in content
    assert 'const epochMetric = traditionalModel' not in content
    assert '训练 Epoch<strong>${run.actual_epochs || "-"}/' not in content
    assert 'result.analysis?.history?.reason' in results
    assert 'function renderHistory(result)' in results
    assert "'traditional_ml'" in results
    assert '实际训练 Epoch' in results


def test_training_spinner_is_stable_across_polling():
    content = Path("static/index.html").read_text(encoding="utf-8")
    results = Path("static/js/run-results.js").read_text(encoding="utf-8")

    assert 'data-active-training-run' in content
    assert 'if (currentRunId === requestedRunId) renderRun(run);' in content
    assert 'export class RunPollController' in results
    assert 'this.controller = new AbortController();' in results
    assert 'if (generation !== this.generation)' in results
    assert 'this.baseDelay * (2 ** Math.min(this.failures, 4))' in results
    assert 'window.SpecAutoAIResults.watchTrainingRun(' in content
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
    assert '确定永久删除训练记录' in content
    assert '删除失败：${error.message}' in content
    assert '模型、指标、预测结果和其他产物都将一并删除' in content
    assert '["succeeded", "success", "failed", "cancelled", "paused"].includes(canonicalState)' in content


def test_modeling_completion_uses_refreshable_dedicated_result_route():
    html = Path("static/index.html").read_text(encoding="utf-8")
    results = Path("static/js/run-results.js").read_text(encoding="utf-8")

    assert 'data-view="results"' in html
    assert 'id="view-results"' in html
    assert 'id="resultMetrics"' in html
    assert 'id="resultAnalysis"' in html
    assert 'id="resultArtifacts"' in html
    assert 'window.SpecAutoAIResults?.showNewRunSuccess(run)' in html
    for element_id in (
        'trainingResultDialog',
        'trainingResultDialogTitle',
        'trainingResultRunId',
        'trainingResultCountdown',
        'trainingResultNow',
        'trainingResultStay',
    ):
        assert f'id="{element_id}"' in html
    assert 'export const AUTO_REDIRECT_DELAY_MS = 3000;' in results
    assert "text: '立即查看结果'" in results
    assert '留在当前页' in html
    assert 'window.location.hash = destination;' in results
    assert 'return await request(`/api/training/runs/${encoded}/result`, { signal });' in results
    assert 'function backendSupportsResultContract' in results
    assert 'function shouldUseLegacyResultFallback' in results
    assert "if (route.view === 'results') loadResult(route.runId);" in results
    assert 'window.SpecAutoAIResults.navigateToView(button.dataset.view)' in html


def test_official_ui_only_renders_sample_feature_importance():
    content = Path("static/index.html").read_text(encoding="utf-8")
    results = Path("static/js/run-results.js").read_text(encoding="utf-8")

    assert "async function renderGlobalFeatureImportance" not in content
    assert "function drawFeatureImportanceCanvas" not in content
    assert "feature-view-switch" not in content
    assert 'data-feature-view="global"' not in content
    assert 'data-feature-view="sample"' not in content
    assert "combinedLogLossView" not in content
    assert "run.feature_importance" not in content
    assert 'function renderExplainability(result)' in results
    assert 'renderGlobalImportance' not in results
    assert "explainabilityPayload(result, 'global'" not in results
    assert '全局特征重要性' not in results
    assert '加载单样品解释' in results
    assert '正在自动加载单样品解释' in results
    assert "const samplePayload = await explainabilityPayload(result, generation);" in results
    assert 'if (generation !== resultRenderGeneration) return;' in results
    assert 'id: \'resultSampleExplanationSelect\'' in results
    assert 'function drawSampleExplanationChart' in results
    assert 'function drawImportanceHeatmap' in results
    assert 'sample-explanation-chart' in results
    assert 'sample-importance-heatmap' in results
    assert '重要性低' in results
    assert '重要性高' in results
    assert 'function triggerArtifactDownload' in results
    assert 'artifact.suggested_filename || response.filename' in results
    assert 'artifact?.downloadable === true' in results
    assert 'safeDownloadUrl(artifact?.download_url)' in results


def test_official_result_ui_renders_three_split_analysis_and_vertical_distributions():
    html = Path("static/index.html").read_text(encoding="utf-8")
    results = Path("static/js/run-results.js").read_text(encoding="utf-8")

    assert '.analysis-split-grid' in html
    assert '.result-split-card' in html
    assert '.distribution-chart' in html
    assert '.distribution-column' in html
    assert 'function analysisSplitEntries' in results
    assert 'function renderSplitConfusion' in results
    assert 'function renderSplitClassMetrics' in results
    assert 'function renderSplitDistribution' in results
    for split in ('train', 'valid', 'test'):
        assert f"'{split}'" in results
    assert 'pooled_cross_fold' in results
    assert 'pooled_oof' in results


def test_modeling_layout_and_recent_result_landing_use_new_summary_fields():
    html = Path("static/index.html").read_text(encoding="utf-8")
    results = Path("static/js/run-results.js").read_text(encoding="utf-8")

    assert '.modeling-primary-grid' in html
    assert 'class="grid modeling-primary-grid"' in html
    assert 'dataset_name' in results
    assert 'started_at' in results
    assert 'duration_seconds' in results
    assert '训练时间' in results
    assert '耗时' in results


def test_result_download_cards_use_fixed_four_column_grid():
    html = Path("static/index.html").read_text(encoding="utf-8")

    assert 'grid-template-columns: repeat(4, minmax(0, 1fr));' in html
    assert '@media (max-width: 1180px)' in html
    assert '@media (max-width: 600px)' in html

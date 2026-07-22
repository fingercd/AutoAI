"""static/v2 独立前端的轻量契约断言。

检查文件存在性、入口路由、api-client 复用、框架/CDN 禁令、令牌存储边界和
run-result-v1 / artifacts[] 消费边界；不启动真实监听端口，也不做浏览器自动化。
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "static" / "v2"

REQUIRED_FILES = [
    "index.html",
    "styles.css",
    "app.js",
    "api.js",
    "store.js",
    "lib/dom.js",
    "lib/format.js",
    "lib/poller.js",
    "lib/charts.js",
    "views/workbench.js",
    "views/modeling.js",
    "views/runs.js",
    "views/result.js",
    "views/manual.js",
    "components/auth-gate.js",
    "components/toast.js",
    "components/model-catalog.js",
    "components/run-list.js",
    "components/artifacts.js",
    "components/confusion-matrix.js",
    "components/explainability-panel.js",
    "tests/run-tests.mjs",
]


def _read(relative: str) -> str:
    return (V2 / relative).read_text(encoding="utf-8")


def _js_sources() -> dict[str, str]:
    sources: dict[str, str] = {}
    for path in sorted(V2.rglob("*")):
        if path.suffix in {".js", ".mjs"} and path.is_file():
            sources[str(path.relative_to(V2)).replace("\\", "/")] = path.read_text(encoding="utf-8")
    return sources


def test_v2_required_files_exist() -> None:
    missing = [name for name in REQUIRED_FILES if not (V2 / name).is_file()]
    assert not missing, f"v2 缺少文件: {missing}"


def test_v2_reuses_shared_api_client() -> None:
    api = _read("api.js")
    assert "../js/api-client.js" in api
    assert re.search(r"import\s*\{[^}]*\brequest\b[^}]*\bdownloadFile\b[^}]*\}\s*from\s*'\.\./js/api-client\.js'", api)
    # 不重复实现令牌存储键，统一走 api-client 的 sessionStorage 边界。
    assert "specautoai.serverToken" not in api


def test_v2_entry_redirect_preserves_static_module_base_path() -> None:
    client = TestClient(app, follow_redirects=False)
    response = client.get("/v2")
    assert response.status_code == 307
    assert response.headers["location"] == "/static/v2/index.html"
    assert TestClient(app).get("/static/v2/index.html").status_code == 200


def test_v2_has_no_framework_or_cdn() -> None:
    html = _read("index.html")
    for name, source in _js_sources().items():
        assert not re.search(r"\b(react|vue|vite)\b", source, re.IGNORECASE), f"{name} 出现框架/构建链关键字"
        for cdn in ("cdn.jsdelivr", "unpkg.com", "cdnjs.cloudflare", "esm.sh"):
            assert cdn not in source, f"{name} 引用了 CDN {cdn}"
    assert 'src="http' not in html and "src='http" not in html
    assert 'href="http' not in html and "href='http" not in html
    assert '<script type="module" src="./app.js"></script>' in html


def test_v2_hplc_preprocess_preserves_range_and_interpolation_controls() -> None:
    api = _read("api.js")
    workbench = _read("views/workbench.js")
    manual = _read("views/manual.js")

    assert "hplc_interpolate" in api
    assert "hplc_interpolate" in workbench
    assert "启用共同时间轴线性插值" in workbench
    assert "validateHplcRowRange" in workbench
    assert "inspectHplc" in api
    assert "hplcInspectionTable" in workbench
    assert "table-wrap hplc-inspection-scroll" in workbench
    assert ".hplc-inspection-scroll" in _read("styles.css")
    assert "common_point_count" in workbench
    assert "form.setPointCount" in workbench
    assert "最大终止行 ${detectedPointCount}" in workbench
    assert "7500" not in workbench
    assert workbench.index("params = form.collect()") < workbench.index("busy = true")
    assert "lastResult.x_axis_consistent" in workbench
    assert "lastResult.hplc_axis" in workbench
    assert "selected_start_row" in workbench
    assert "selected_end_row" in workbench
    assert "真实保留时间" in workbench
    assert "每个真实时间坐标逐列保存在 CSV 特征表头中" in workbench
    assert "lastResult.output_precision.format" in workbench
    assert "lastResult.output_precision.feature_count" in workbench
    assert "lastResult.output_precision.total_column_count" in workbench
    assert "保留时间下限（分钟）" in workbench
    assert "保留时间上限（分钟）" in workbench
    assert "保留时间（分钟）" in workbench
    assert workbench.count("下载统一建模 CSV") == 1
    for removed in ("linspace-slice-v1", "xxx_download_url", "downloadVisibleAxis", "下载可见 XXX 时间轴", "归一化 X 坐标"):
        assert removed not in workbench
    assert "wide-feature-v2 宽表" in manual
    assert "wide-feature-v1" in manual
    assert "Name" in manual
    assert "旧的六列数组/JSON CSV 不再可训练" in manual
    assert "16,384" in manual and "16,380" in manual
    assert "warning-panel" in workbench
    assert "不做面积归一化" in workbench
    for removed in ("hplc_subtract_min", "hplc_normalize_area", "减最小值"):
        assert removed not in api
        assert removed not in workbench


def test_v2_modeling_summary_and_business_tables_use_unified_terms() -> None:
    modeling = _read("views/modeling.js")
    metrics = modeling.split("const metrics = [", 1)[1].split("];", 1)[0]
    labels = ["数据量", "类别数", "样本数", "每样本测量数", "特征数"]
    positions = [metrics.index(f"['{label}'") for label in labels]

    assert positions == sorted(positions)
    assert "expected_repeats_per_group" in metrics
    assert "renderSampleIdList" in modeling
    assert "按样本分组" in modeling
    assert "类别分布" in modeling
    assert "el('th', { text: '类别'" in modeling
    assert "el('th', { text: '数据量'" in modeling
    assert "wide-feature-v2 CSV" in modeling
    assert "旧 wide-feature-v1 仍兼容" in modeling
    assert "前三列固定为 Index、Label、Sample_ID" in modeling
    assert "模型目录已加载" not in modeling
    for old_term in ("曲线数（行）", "Sample_ID 组数", "曲线长度", "标签分布"):
        assert old_term not in modeling


def test_v2_result_uses_one_training_time_and_user_facing_cv_terms() -> None:
    formatting = _read("lib/format.js")
    result = _read("views/result.js")
    modeling = _read("views/modeling.js")
    manual = _read("views/manual.js")

    assert "export function formatTrainingTime" in formatting
    assert "run?.started_at" in formatting
    assert "run?.created_at" in formatting
    assert "（任务创建）" in formatting
    assert "['训练时间', formatTrainingTime(run)]" in result
    assert "['创建时间'" not in result
    assert "['开始时间'" not in result
    assert "['数据量', dataset.curve_count]" in result
    assert "['样本数', dataset.sample_id_count]" in result
    assert "['独立测试数据量', dataset.test_curve_count]" in result
    assert "真实数据量" in result and "预测数据量" in result
    for source in (formatting, result, modeling, manual):
        assert "OOF" not in source
        assert "Support" not in source


def test_v2_run_list_shows_test_macro_f1_and_training_time() -> None:
    run_list = _read("components/run-list.js")

    assert "测试集 Macro F1" in run_list
    assert "formatMetric(item.test_macro_f1)" in run_list
    assert "formatTrainingTime(item)" in run_list
    assert "'训练时间'" in run_list
    assert "'创建时间'" not in run_list
    assert "run-summary-table" in run_list
    for action in ("view", "copy", "cancel", "delete"):
        assert f"action: '{action}'" in run_list


def test_v2_token_only_in_session_storage_via_api_client() -> None:
    for name, source in _js_sources().items():
        assert not re.search(r"localStorage\s*[.\[]", source), f"{name} 访问了 localStorage"
        assert not re.search(r"\.innerHTML\s*=", source), f"{name} 使用了 innerHTML 赋值"


def test_v2_result_view_consumes_run_result_v1_only() -> None:
    api = _read("api.js")
    assert "run-result-v1" in _read("views/result.js")
    assert re.search(r"/api/training/runs/\$\{encodeURIComponent\(runId\)\}/result", api)
    # v2 不自行拼接 artifact 下载地址，只使用 artifacts[] 返回的 download_url。
    for name, source in _js_sources().items():
        assert "/artifact/${" not in source, f"{name} 自行拼接了 artifact URL"
    artifacts = _read("components/artifacts.js")
    for forbidden in ("model.pkl", "model.pt", "status.json", "manifest.json"):
        assert forbidden in artifacts
    assert ".joblib" in artifacts
    assert "download_url" in artifacts


def test_v2_explainability_auto_loads_current_sample_series() -> None:
    view = _read("views/result.js")
    panel = _read("components/explainability-panel.js")
    assert "收起单样品解释" in view
    assert "解释数据默认自动加载" in view
    assert "renderExplainabilityPanel" in view
    assert "if (action.enabled) load();" in panel
    assert "sample?.curve" in panel
    assert "sample?.sample_x_axis" in panel
    assert "resolveSampleSeries" in panel


def test_v2_forbidden_artifact_names_only_in_guard_and_tests() -> None:
    # manual.js 仅在说明文档中解释这些文件不开放下载，属合法文案。
    allowed = {"components/artifacts.js", "tests/run-tests.mjs", "views/manual.js"}
    for name, source in _js_sources().items():
        if "model.pkl" in source or ".joblib" in source:
            assert name in allowed, f"{name} 不应引用内部产物文件名"

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

"""首页、健康检查、本地样例摘要和模型能力目录路由。

模型目录始终返回 15 个目标 ID，并用 capability 字段表达可选依赖是否可用；
前端据此禁用模型，而不是维护另一份硬编码清单。
"""

from __future__ import annotations

import importlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse

from ..parsers import summarize_modeling_csv
from ..paths import DEFAULT_DATA, PREPROCESSED_DIR, STATIC_DIR, UPLOADS_DIR
from ..version import WEB_CONTRACTS, WORKER_CONTRACT_VERSION

router = APIRouter()

# 顺序同时决定前端目录的稳定展示顺序；新增模型需同步 registry 与契约测试。
_MODEL_CATALOG: tuple[tuple[str, str, str], ...] = (
    ("pls_da", "PLS-DA", "traditional_ml"),
    ("pca_lda", "PCA-LDA", "traditional_ml"),
    ("logistic_regression", "Logistic Regression", "traditional_ml"),
    ("svm", "SVM", "traditional_ml"),
    ("random_forest", "Random Forest", "traditional_ml"),
    ("xgboost", "XGBoost", "traditional_ml"),
    ("pca_mlp", "PCA-MLP", "basic_deep"),
    ("cnn1d", "1D CNN", "basic_deep"),
    ("cnn1d_se", "1D CNN-SE", "convolutional"),
    ("resnet1d", "1D ResNet", "convolutional"),
    ("inception1d", "1D Inception", "convolutional"),
    ("tcn1d", "1D TCN", "convolutional"),
    ("cnn_transformer1d", "CNN-Transformer", "long_range"),
    ("cnn_mamba1d", "CNN-Mamba", "long_range"),
    ("dscarnet", "DSCARNet", "two_dimensional_mapping"),
)

_MODEL_MODULES = {
    "pls_da": "backend.app.models.pls_da",
    "pca_lda": "backend.app.models.pca_lda",
    "logistic_regression": "backend.app.models.logistic_regression",
    "svm": "backend.app.models.svm",
    "random_forest": "backend.app.models.random_forest",
    "xgboost": "backend.app.models.xgboost",
    "pca_mlp": "backend.app.models.pca_mlp",
    "cnn1d": "backend.app.models.cnn1d",
    "cnn1d_se": "backend.app.models.cnn_se1d",
    "resnet1d": "backend.app.models.resnet1d",
    "inception1d": "backend.app.models.inception1d",
    "tcn1d": "backend.app.models.tcn1d",
    "cnn_transformer1d": "backend.app.models.cnn_transformer1d",
}
_MODEL_ROOT = Path(__file__).resolve().parents[1] / "models"


def _import_capability(model_type: str, display_name: str) -> tuple[bool, str | None]:
    module_name = _MODEL_MODULES[model_type]
    module_file = _MODEL_ROOT / f"{module_name.rsplit('.', 1)[-1]}.py"
    if not module_file.is_file():
        return False, f"{display_name} 不可用：缺少内部模型模块 {module_name}"
    return True, None


def _mamba_capability() -> tuple[bool, str | None]:
    if importlib.util.find_spec("mamba_ssm") is None:
        return False, (
            "CNN-Mamba 需要可选依赖 mamba-ssm（Python 导入名为 mamba_ssm）；"
            "请安装与当前 PyTorch/CUDA 匹配的官方兼容版本。"
        )
    return True, None


def _dscarnet_capability() -> tuple[bool, str | None]:
    if not (_MODEL_ROOT / "dscarnet.py").is_file():
        return False, "DSCARNet/AggMap 不可用：缺少内部 DSCARNet 模块"
    if importlib.util.find_spec("aggmap") is None:
        return False, "DSCARNet/AggMap 不可用：缺少 aggmap"
    # Capability discovery runs in the Web request path. Importing AggMap
    # here also imports UMAP/Numba and can spend minutes compiling third-party
    # code. The Worker performs the authoritative import and compatibility
    # patch when a DSCARNet Run actually starts.
    return True, None


def _model_capability(model_type: str, display_name: str) -> tuple[bool, str | None]:
    if model_type == "cnn_mamba1d":
        return _mamba_capability()
    if model_type == "dscarnet":
        return _dscarnet_capability()
    return _import_capability(model_type, display_name)


def _curve_intensity_summary(curves: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for curve in curves:
        values = curve.get('processed_y') or curve.get('corrected_y') or curve.get('raw_y') or []
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        summary.append(
            {
                'name': str(curve.get('name', '')),
                'point_count': int(arr.size),
                'finite_count': int(finite.size),
                'min': float(np.min(finite)) if finite.size else None,
                'max': float(np.max(finite)) if finite.size else None,
                'mean': float(np.mean(finite)) if finite.size else None,
                'all_zero': bool(np.all(np.abs(finite) <= 1e-12)) if finite.size else True,
            }
        )
    return summary


@router.get('/')
def index() -> FileResponse:
    """返回正式静态前端入口。"""
    return FileResponse(STATIC_DIR / 'index.html')


@router.get('/v2', include_in_schema=False)
@router.get('/v2/', include_in_schema=False)
def v2_index() -> RedirectResponse:
    """把简洁入口重定向到 v2 的真实静态基路径，确保共享模块相对导入有效。"""
    return RedirectResponse(url='/static/v2/index.html', status_code=307)


@router.get('/health')
def health(request: Request) -> dict[str, object]:
    """报告 Web 存活与独立训练 worker 的最近心跳摘要。"""

    from ..paths import RUNS_DATABASE
    from ..runs.repository import RunRepository

    worker_summary: dict[str, object]
    try:
        repository = RunRepository(RUNS_DATABASE)
        repository.initialize()
        worker_health = repository.worker_health(
            now=datetime.now(timezone.utc),
            expected_contract_version=WORKER_CONTRACT_VERSION,
        )
        workers = list(worker_health.get('workers') or [])
        live_workers = [item for item in workers if item.get('live')]
        worker_summary = {
            'available': bool(worker_health.get('available')),
            'compatible': bool(worker_health.get('compatible')),
            'contract_version': worker_health.get('contract_version'),
            'live_count': len(live_workers),
            'last_seen_at': workers[0].get('last_seen_at') if workers else None,
            'active_run_count': sum(1 for item in live_workers if item.get('active_run_id')),
        }
    except Exception:
        # 健康检查不能把数据库路径或底层异常公开给未认证的探针。
        worker_summary = {
            'available': False,
            'compatible': False,
            'contract_version': None,
            'live_count': 0,
            'last_seen_at': None,
            'active_run_count': 0,
            'diagnostic': 'unavailable',
        }
    return {
        'status': 'ok',
        'deployment_mode': request.app.state.security_settings.mode,
        'contracts': dict(WEB_CONTRACTS),
        'worker': worker_summary,
    }


@router.get('/api/models')
def get_models_catalog() -> dict[str, list[dict[str, object]]]:
    """返回稳定的 15 项模型目录及当前环境 capability，不启动训练。"""

    from ..training_explainability import explainability_method

    models: list[dict[str, object]] = []
    for model_type, display_name, family in _MODEL_CATALOG:
        available, unavailable_reason = _model_capability(model_type, display_name)
        models.append(
            {
                "id": model_type,
                "display_name": display_name,
                "family": family,
                "available": available,
                "unavailable_reason": unavailable_reason,
                "explainability_method": explainability_method(model_type),
            }
        )
    return {"models": models}


@router.get('/api/sample/summary')
def sample_summary() -> dict[str, object]:
    """读取可选根目录 data.csv，仅供本地兼容与人工验证。"""
    if not DEFAULT_DATA.exists():
        raise HTTPException(status_code=404, detail='项目根目录未找到 data.csv')
    try:
        return summarize_modeling_csv(DEFAULT_DATA)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get('/api/files')
def get_file(path: str) -> FileResponse:
    """下载 uploads/preprocessed 内文件；Run 产物不经过此接口。"""
    target = Path(path).resolve()
    roots = [UPLOADS_DIR.resolve(), PREPROCESSED_DIR.resolve()]
    if not any(target == root or root in target.parents for root in roots):
        raise HTTPException(status_code=403, detail='不允许访问该路径')
    if not target.is_file():
        raise HTTPException(status_code=404, detail='文件不存在')
    return FileResponse(target, filename=target.name)

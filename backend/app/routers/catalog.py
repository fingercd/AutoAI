# ============================================================================
# 模块说明：backend/app/routers/catalog.py
#
# 职责：
#   与“目录/元信息”相关的只读 HTTP 端点：前端首页与 v2 入口、/health
#   健康检查、/api/models 模型能力目录、/api/sample/summary 本地样例摘要，
#   以及 /api/files 上传/预处理文件的受控下载。
#
# 在系统中的位置：
#   前端启动时调用 /api/models 渲染可选模型列表，依据 available /
#   unavailable_reason 禁用当前环境不可训练的模型。ui_visible 是产品展示
#   边界，不影响后端兼容训练能力。
#
# 关键设计约束：
#   - 模型目录返回所有后端能力；前端只能渲染 ui_visible=true 的十项；
#     可选依赖（mamba-ssm、aggmap）缺失时只标 available=false，
#     绝不用近似实现静默顶替。
#   - capability 探测运行在 Web 请求路径上，必须廉价：只查文件存在性与
#     importlib.util.find_spec，绝不真正 import 重依赖（AggMap/UMAP/Numba
#     首次编译可能耗时数分钟）。
#   - /health 对外不能泄露数据库路径或底层异常细节。
#   - /api/files 只允许下载 uploads 与 preprocessed 两个目录内的文件，
#     Run 产物（模型权重、status.json 等）不经过此接口。
# ============================================================================

"""首页、健康检查、本地样例摘要和模型能力目录路由。

模型目录返回所有后端能力，并用 capability 与显式 UI 可见性表达可用边界；
前端不维护第二份硬编码清单，也不得因 UI 收敛删除后端能力。
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
# 模型能力目录（全系统单一事实来源）：(模型 id, 展示名, 家族) 三元组。
_MODEL_CATALOG: tuple[tuple[str, str, str, bool], ...] = (
    ("pls_da", "PLS-DA", "traditional_ml", True),
    ("spls_da", "sPLS-DA", "traditional_ml", False),
    ("pca_lda", "PCA-LDA", "traditional_ml", False),
    ("logistic_regression", "Elastic Net", "traditional_ml", True),
    ("svm", "SVM", "traditional_ml", True),
    ("pca_svm", "PCA-SVM", "traditional_ml", False),
    ("random_forest", "Random Forest", "traditional_ml", True),
    ("xgboost", "XGBoost", "traditional_ml", True),
    ("pca_mlp", "PCA-MLP", "basic_deep", False),
    ("cnn1d", "1D-CNN", "basic_deep", True),
    ("cnn1d_se", "1D CNN-SE", "convolutional", False),
    ("resnet1d", "1D ResNet", "convolutional", False),
    ("inception1d", "1D Inception", "convolutional", False),
    ("tcn1d", "1D TCN", "convolutional", False),
    ("cnn_transformer1d", "CNN-Transformer", "long_range", False),
    ("cnn_mamba1d", "CNN-Mamba", "long_range", False),
    ("dscarnet", "DSCARNet", "two_dimensional_mapping", False),
)

# 各模型对应的内部实现模块路径，用于“模块文件是否存在”的廉价探测；
# cnn_mamba1d 与 dscarnet 不在此表——它们走专属的可选依赖探测函数。
_MODEL_MODULES = {
    "pls_da": "backend.app.models.pls_da",
    "spls_da": "backend.app.models.spls_da",
    "pca_lda": "backend.app.models.pca_lda",
    "logistic_regression": "backend.app.models.logistic_regression",
    "svm": "backend.app.models.svm",
    "pca_svm": "backend.app.models.pca_svm",
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
# backend/app/models 目录的绝对路径，供文件存在性检查使用。
_MODEL_ROOT = Path(__file__).resolve().parents[1] / "models"


def _import_capability(model_type: str, display_name: str) -> tuple[bool, str | None]:
    """通用 capability 探测：仅检查模型实现 .py 文件是否存在。

    返回 (available, unavailable_reason)。故意不做真正的 import，
    保证 /api/models 在 Web 请求路径上保持毫秒级响应。
    """
    module_name = _MODEL_MODULES[model_type]
    module_file = _MODEL_ROOT / f"{module_name.rsplit('.', 1)[-1]}.py"
    if not module_file.is_file():
        return False, f"{display_name} 不可用：缺少内部模型模块 {module_name}"
    return True, None


def _mamba_capability() -> tuple[bool, str | None]:
    """cnn_mamba1d 的 capability 探测：可选依赖 mamba-ssm 是否可导入。

    mamba-ssm 与 PyTorch/CUDA 版本强耦合，很多环境无法安装；缺失时返回
    available=false 与人类可读的安装提示，而不是拖到训练时才报错。
    """
    # find_spec 只解析不执行模块，代价远低于真正 import mamba_ssm。
    if importlib.util.find_spec("mamba_ssm") is None:
        return False, (
            "CNN-Mamba 需要可选依赖 mamba-ssm（Python 导入名为 mamba_ssm）；"
            "请安装与当前 PyTorch/CUDA 匹配的官方兼容版本。"
        )
    return True, None


def _dscarnet_capability() -> tuple[bool, str | None]:
    """dscarnet 的 capability 探测：内部模块文件 + aggmap 依赖双重检查。

    注意这里故意不 import aggmap：它会连带拉起 UMAP/Numba 并触发第三方
    代码编译，在 Web 请求里可能卡数分钟。真正权威的导入与兼容性 patch
    由 Worker 在 DSCARNet Run 实际启动时完成（见下方英文注释）。
    """
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
    """按模型类型分派到对应 capability 探测函数的统一入口。

    两个特殊模型（cnn_mamba1d、dscarnet）走可选依赖探测，
    其余 15 个走通用的“模块文件存在性”探测。
    """
    if model_type == "cnn_mamba1d":
        return _mamba_capability()
    if model_type == "dscarnet":
        return _dscarnet_capability()
    return _import_capability(model_type, display_name)


def _curve_intensity_summary(curves: list[dict[str, object]]) -> list[dict[str, object]]:
    """为曲线列表计算强度摘要的工具函数（min/max/mean/全零判定）。

    按 processed_y -> corrected_y -> raw_y 优先级取 Y 数组，剔除 NaN/Inf
    后统计；当前模块内暂无调用点，与 preprocess.py 中的同名实现保持一致，
    供需要曲线摘要的目录类端点复用。
    """
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
    # 正式前端入口 static/index.html，由后端直接静态托管。
    return FileResponse(STATIC_DIR / 'index.html')


@router.get('/v2', include_in_schema=False)
@router.get('/v2/', include_in_schema=False)
def v2_index() -> RedirectResponse:
    """把简洁入口重定向到 v2 的真实静态基路径，确保共享模块相对导入有效。"""
    # 307 临时重定向不改变请求方法；真实资源位于 /static/v2/ 下，
    # 这样 v2 页面里的相对路径导入（如 ../js/api-client.js）才有正确基路径。
    return RedirectResponse(url='/static/v2/index.html', status_code=307)


@router.get('/health')
def health(request: Request) -> dict[str, object]:
    """报告 Web 存活与独立训练 worker 的最近心跳摘要。"""

    # 延迟导入：避免在模块加载期就引入数据库依赖，
    # 让仅访问首页/目录的进程保持轻量。
    from ..paths import RUNS_DATABASE
    from ..runs.repository import RunRepository

    worker_summary: dict[str, object]
    try:
        # 查询训练 worker 心跳：直接读 runs 数据库，并与期望契约版本
        # （WORKER_CONTRACT_VERSION）比对，版本不符视为不兼容。
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
    # 任何底层故障都收敛为固定结构的“不可用”摘要——
    # 健康检查绝不能把数据库路径或堆栈泄露给未认证探针。
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
        # status=ok 仅表示 Web 进程存活；worker 可用性单独在 worker 字段表达。
        'status': 'ok',
        'deployment_mode': request.app.state.security_settings.mode,
        'contracts': dict(WEB_CONTRACTS),
        'worker': worker_summary,
    }


@router.get('/api/models')
def get_models_catalog() -> dict[str, list[dict[str, object]]]:
    """返回完整后端模型目录及当前 capability，不启动训练。"""

    models: list[dict[str, object]] = []
    # 按 _MODEL_CATALOG 的固定顺序逐项探测 capability 并组装响应；
    # 该顺序即前端展示顺序，新增模型必须同步 registry 与契约测试。
    for model_type, display_name, family, ui_visible in _MODEL_CATALOG:
        available, unavailable_reason = _model_capability(model_type, display_name)
        models.append(
            {
                "id": model_type,
                "display_name": display_name,
                "family": family,
                "available": available,
                "unavailable_reason": unavailable_reason,
                "ui_visible": ui_visible,
                "visibility_reason": None if ui_visible else "temporarily_hidden_from_ui",
            }
        )
    return {"models": models}


@router.get('/api/sample/summary')
def sample_summary() -> dict[str, object]:
    """读取可选根目录 data.csv，仅供本地兼容与人工验证。"""
    # data.csv 只是本地验证用的可选样例，不存在时明确 404 而非静默失败。
    if not DEFAULT_DATA.exists():
        raise HTTPException(status_code=404, detail='项目根目录未找到 data.csv')
    try:
        # 解析失败（ValueError）转成 400，把可读的错误信息交给前端。
        return summarize_modeling_csv(DEFAULT_DATA)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get('/api/files')
def get_file(path: str) -> FileResponse:
    """下载 uploads/preprocessed 内文件；Run 产物不经过此接口。"""
    # 先 resolve 归一化（消除 .. 与符号链接），再校验目标必须位于
    # UPLOADS_DIR 或 PREPROCESSED_DIR 之内——目录白名单防路径穿越。
    target = Path(path).resolve()
    roots = [UPLOADS_DIR.resolve(), PREPROCESSED_DIR.resolve()]
    if not any(target == root or root in target.parents for root in roots):
        raise HTTPException(status_code=403, detail='不允许访问该路径')
    # 通过白名单后再确认是真实存在的普通文件，否则 404。
    if not target.is_file():
        raise HTTPException(status_code=404, detail='文件不存在')
    return FileResponse(target, filename=target.name)

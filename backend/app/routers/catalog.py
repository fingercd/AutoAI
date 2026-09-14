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
#   unavailable_reason 禁用当前环境不可训练的模型；可解释性方法声明来自
#   backend/app/training_explainability.py，与训练侧保持一致。
#
# 关键设计约束：
#   - 模型目录固定返回 当前活跃目标 ID，前端不维护第二份硬编码清单；
#     可选依赖（mamba-ssm）缺失时只标 available=false，
#     绝不用近似实现静默顶替。
#   - capability 探测运行在 Web 请求路径上，必须廉价：只查文件存在性与
#     importlib.util.find_spec，绝不真正 import 重依赖。
#   - /health 对外不能泄露数据库路径或底层异常细节。
#   - /api/files 只允许下载 uploads 与 preprocessed 两个目录内的文件，
#     Run 产物（模型权重、status.json 等）不经过此接口。
# ============================================================================

"""首页、健康检查、本地样例摘要和模型能力目录路由。

模型目录始终返回 当前活跃目标 ID，并用 capability 字段表达可选依赖是否可用；
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

from ..model_catalog import MODEL_DECLARATIONS, model_availability

# Historical helper names remain available; all identity and probes are shared.
_MODEL_CATALOG = tuple((m.id, m.display_name, m.family) for m in MODEL_DECLARATIONS)


def _model_capability(model_type: str, display_name: str) -> tuple[bool, str | None]:
    if model_type == "cnn_mamba1d":
        return _mamba_capability()
    return _import_capability(model_type, display_name)


def _probe_capability(model_type: str, display_name: str) -> tuple[bool, str | None]:
    available, reason = model_availability(model_type)
    return available, None if available else f"{display_name} unavailable: {reason}"


def _import_capability(model_type: str, display_name: str) -> tuple[bool, str | None]:
    return _probe_capability(model_type, display_name)


def _mamba_capability() -> tuple[bool, str | None]:
    return _probe_capability('cnn_mamba1d', 'CNN-Mamba')




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
    """返回稳定的 15 项模型目录及当前环境 capability，不启动训练。"""

    # 延迟导入可解释性声明，避免目录接口拉起训练侧重依赖。
    from ..training_explainability import explainability_method

    models: list[dict[str, object]] = []
    # 按 _MODEL_CATALOG 的固定顺序逐项探测 capability 并组装响应；
    # 该顺序即前端展示顺序，新增模型必须同步 registry 与契约测试。
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

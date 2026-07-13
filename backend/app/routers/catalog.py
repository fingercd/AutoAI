from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..parsers import summarize_modeling_csv
from ..paths import DEFAULT_DATA, PREPROCESSED_DIR, STATIC_DIR, UPLOADS_DIR

router = APIRouter()

_MODEL_CATALOG: tuple[tuple[str, str, str, str, str], ...] = (
    ('pls_da', 'PLS-DA', 'traditional_ml', 'backend.app.models.pls_da', 'window_permutation'),
    ('svm', 'SVM', 'traditional_ml', 'backend.app.models.svm', 'window_permutation'),
    ('random_forest', 'Random Forest', 'traditional_ml', 'backend.app.models.random_forest', 'window_permutation'),
    ('xgboost', 'XGBoost', 'traditional_ml', 'backend.app.models.xgboost', 'window_permutation'),
    ('cnn1d', '1D-CNN', 'basic_deep', 'backend.app.models.cnn1d', 'gradcam_1d'),
    ('transformer1d', '1D-Transformer', 'long_range', 'backend.app.models.transformer', 'input_gradient_attribution'),
    ('resnet1d', '1D-ResNet', 'convolutional', 'backend.app.models.resnet1d', 'gradcam_1d'),
    ('inception1d', '1D-Inception', 'convolutional', 'backend.app.models.inception1d', 'gradcam_1d'),
    ('tcn1d', '1D-TCN', 'convolutional', 'backend.app.models.tcn1d', 'gradcam_1d'),
    ('dscarnet', 'DSCARNet', 'two_dimensional_mapping', 'backend.app.models.dscarnet', 'dscarnet_dual_2d_gradcam'),
)


def _model_capability(model_type: str, display_name: str, module_name: str) -> tuple[bool, str | None]:
    if model_type == 'dscarnet':
        for dependency in ('aggmap', 'lapjv'):
            if importlib.util.find_spec(dependency) is None:
                return False, f'{display_name} 不可用：缺少 {dependency}'
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        detail = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
        return False, f'{display_name} 不可用：{detail}'
    return True, None

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
    return FileResponse(STATIC_DIR / 'index.html')


@router.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@router.get('/api/models')
def get_models_catalog() -> dict[str, list[dict[str, object]]]:
    models: list[dict[str, object]] = []
    for model_type, display_name, family, module_name, explainability_method in _MODEL_CATALOG:
        available, unavailable_reason = _model_capability(model_type, display_name, module_name)
        models.append(
            {
                'id': model_type,
                'display_name': display_name,
                'family': family,
                'available': available,
                'unavailable_reason': unavailable_reason,
                'explainability_method': explainability_method,
            }
        )
    return {'models': models}


@router.get('/api/sample/summary')
def sample_summary() -> dict[str, object]:
    if not DEFAULT_DATA.exists():
        raise HTTPException(status_code=404, detail='项目根目录未找到 data.csv')
    return summarize_modeling_csv(DEFAULT_DATA)


@router.get('/api/files')
def get_file(path: str) -> FileResponse:
    target = Path(path).resolve()
    roots = [UPLOADS_DIR.resolve(), PREPROCESSED_DIR.resolve()]
    if not any(target == root or root in target.parents for root in roots):
        raise HTTPException(status_code=403, detail='不允许访问该路径')
    if not target.is_file():
        raise HTTPException(status_code=404, detail='文件不存在')
    return FileResponse(target, filename=target.name)

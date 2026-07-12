from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..parsers import summarize_modeling_csv
from ..paths import DEFAULT_DATA, PREPROCESSED_DIR, STATIC_DIR, UPLOADS_DIR

router = APIRouter()


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
    "cnn1d": "backend.app.models.cnn1d_v2",
    "cnn1d_se": "backend.app.models.cnn_se1d",
    "resnet1d": "backend.app.models.resnet1d_v2",
    "inception1d": "backend.app.models.inception1d_v2",
    "tcn1d": "backend.app.models.tcn1d_v2",
    "cnn_transformer1d": "backend.app.models.cnn_transformer1d",
}


def _capability_failure(model_name: str, exc: BaseException) -> tuple[bool, str]:
    detail = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
    return False, f"{model_name} 不可用：{detail}"


def _import_capability(model_type: str, display_name: str) -> tuple[bool, str | None]:
    try:
        importlib.import_module(_MODEL_MODULES[model_type])
    except Exception as exc:  # Catalog reporting must not break the landing page.
        return _capability_failure(display_name, exc)
    return True, None


def _mamba_capability() -> tuple[bool, str | None]:
    try:
        from ..models.cnn_mamba1d import (
            MAMBA_INSTALL_MESSAGE,
            ModelDependencyError,
            build_cnn_mamba1d,
            mamba_available,
        )

        if not mamba_available():
            return False, MAMBA_INSTALL_MESSAGE
        build_cnn_mamba1d(input_length=8, class_count=2, hidden_size=8, mamba_layers=1)
    except ModelDependencyError as exc:
        return _capability_failure("CNN-Mamba", exc)
    except Exception as exc:  # Optional native dependencies can fail after import.
        return _capability_failure("CNN-Mamba", exc)
    return True, None


def _dscarnet_capability() -> tuple[bool, str | None]:
    try:
        from ..models.dscarnet import dual_dscarnet

        # Do not invoke the legacy AggMap compatibility loader here: a broken
        # native lapjv import can terminate the interpreter before Python can
        # turn it into an exception.  The catalog must remain safe for the
        # landing page, so require the real runtime dependencies first.
        if importlib.util.find_spec("aggmap") is None:
            return False, "DSCARNet/AggMap 不可用：缺少 aggmap"
        if importlib.util.find_spec("lapjv") is None:
            return False, "DSCARNet/AggMap 不可用：缺少 lapjv"
        importlib.import_module("aggmap")
        dual_dscarnet((4, 4, 1), (4, 4, 1), n_outputs=2, last_avf=None)
    except Exception as exc:  # AggMap and its native optional dependencies are not guaranteed.
        return _capability_failure("DSCARNet/AggMap", exc)
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
    return FileResponse(STATIC_DIR / 'index.html')


@router.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@router.get('/api/models')
def get_models_catalog() -> dict[str, list[dict[str, object]]]:
    """Report model import/build capability without starting a training run."""

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

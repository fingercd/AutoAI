from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..parsers import summarize_modeling_csv
from ..paths import DEFAULT_DATA, PREPROCESSED_DIR, STATIC_DIR, UPLOADS_DIR

router = APIRouter()

UI_PAGES = {
    'workbench': 'ui-workbench.html',
    'wizard': 'ui-wizard.html',
    'dashboard': 'ui-dashboard.html',
    'console': 'ui-console.html',
    'minimal-lab': 'ui-minimal-lab.html',
    'swiss': 'ui-swiss.html',
    'dark-instrument': 'ui-dark-instrument.html',
    'warm-paper': 'ui-warm-paper.html',
}


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


@router.get('/ui')
def ui_gallery() -> FileResponse:
    return FileResponse(STATIC_DIR / 'ui-gallery.html')


@router.get('/ui/{name}')
def ui_variant(name: str) -> FileResponse:
    filename = UI_PAGES.get(name)
    if not filename:
        raise HTTPException(status_code=404, detail='UI 方案不存在')
    return FileResponse(STATIC_DIR / filename)


@router.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


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

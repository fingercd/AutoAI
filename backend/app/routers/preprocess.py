from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..hplc import preprocess_hplc_files_with_preview
from ..parsers import preprocess_raw_files_with_preview
from ..paths import PREPROCESSED_DIR, UPLOADS_DIR

router = APIRouter()


def _save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or 'upload.csv').suffix or '.csv'
    target = UPLOADS_DIR / f'{uuid.uuid4().hex}{suffix}'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('wb') as handle:
        file.file.seek(0)
        handle.write(file.file.read())
    return target


def _curve_intensity_summary(curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for curve in curves:
        values = curve.get('processed_y') or curve.get('corrected_y') or curve.get('raw_y') or []
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            min_value = float(np.min(finite))
            max_value = float(np.max(finite))
            mean_value = float(np.mean(finite))
            all_zero = bool(np.all(np.abs(finite) <= 1e-12))
        else:
            min_value = max_value = mean_value = None
            all_zero = True
        summary.append(
            {
                'name': str(curve.get('name', '')),
                'point_count': int(arr.size),
                'finite_count': int(finite.size),
                'min': min_value,
                'max': max_value,
                'mean': mean_value,
                'all_zero': all_zero,
            }
        )
    return summary


@router.post('/api/preprocess/{kind}')
def preprocess(
    kind: str,
    files: list[UploadFile] = File(...),
    start_row: int = Form(1),
    end_row: int | None = Form(None),
    range_mode: str = Form('row'),
    x_min: float | None = Form(None),
    x_max: float | None = Form(None),
    baseline_method: str = Form('arPLS'),
    hplc_interpolate: bool = Form(True),
    hplc_subtract_min: bool = Form(True),
    hplc_normalize_area: bool = Form(True),
) -> dict[str, Any]:
    if kind not in {'raman', 'chromatography', 'hplc'}:
        raise HTTPException(status_code=400, detail='kind 必须是 raman、chromatography 或 hplc')
    original_names = [Path(file.filename or f'sample_{idx}').stem for idx, file in enumerate(files, start=1)]
    saved_files = [_save_upload(file) for file in files]
    try:
        if kind == 'hplc':
            result = preprocess_hplc_files_with_preview(
                saved_files,
                start_row=start_row,
                end_row=end_row,
                range_mode=range_mode,
                x_min=x_min,
                x_max=x_max,
                display_names=original_names,
                interpolate=hplc_interpolate,
                subtract_min=hplc_subtract_min,
                normalize_area=hplc_normalize_area,
            )
        else:
            result = preprocess_raw_files_with_preview(
                saved_files,
                kind=kind,
                start_row=start_row,
                end_row=end_row,
                range_mode=range_mode,
                x_min=x_min,
                x_max=x_max,
                baseline_method=baseline_method,
                display_names=original_names,
            )
        frame = result['frame']
        output = PREPROCESSED_DIR / f'{kind}_{uuid.uuid4().hex[:10]}.csv'
        frame.to_csv(output, index=False, encoding='utf-8-sig')
        response: dict[str, Any] = {
            'output_path': str(output.resolve()),
            'download_url': f'/api/files?path={output.resolve()}',
            'rows': int(len(frame)),
            'range_mode': range_mode,
            'x_min': x_min,
            'x_max': x_max,
            'curves': result['curves'],
            'preview': frame.head(5).drop(columns=['XXX', 'Intensity']).to_dict(orient='records'),
            'intensity_summary': _curve_intensity_summary(result['curves']),
        }
        if kind == 'hplc':
            response.update(
                {
                    'baseline_method': None,
                    'hplc_interpolate': hplc_interpolate,
                    'hplc_subtract_min': hplc_subtract_min,
                    'hplc_normalize_area': hplc_normalize_area,
                    'common_time': result.get('common_time', []),
                }
            )
            if result.get('common_time'):
                npy_path = PREPROCESSED_DIR / f'common_time_{output.stem}.npy'
                np.save(npy_path, np.array(result['common_time'], dtype=np.float32))
                response['common_time_path'] = str(npy_path.resolve())
        elif kind == 'raman':
            response['baseline_method'] = baseline_method
        else:
            response['baseline_method'] = None
        return response
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

"""拉曼、简单色谱和 HPLC 预处理 HTTP 路由。

该层只负责 multipart 参数、上传落盘、算法分派和下载 URL 组装。拉曼与 HPLC
数学处理分别位于 parsers.py 和 hplc.py；主色谱界面默认使用 hplc 分支。
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..hplc import preprocess_hplc_files_with_preview
from ..parsers import modeling_metadata_preview, preprocess_raw_files_with_preview
from ..paths import PREPROCESSED_DIR, UPLOADS_DIR

router = APIRouter()


def _save_upload(file: UploadFile) -> Path:
    """把单个 UploadFile 复制到随机命名的受控上传目录。"""
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
) -> dict[str, Any]:
    """校验 kind 与范围参数，执行对应管线并返回预览及下载地址。"""
    if kind not in {'raman', 'chromatography', 'hplc'}:
        raise HTTPException(status_code=400, detail='kind 必须是 raman、chromatography 或 hplc')
    original_names = [Path(file.filename or f'sample_{idx}').stem for idx, file in enumerate(files, start=1)]
    saved_files = [_save_upload(file) for file in files]
    temporary_output: Path | None = None
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
        output_token = uuid.uuid4().hex[:10]
        output = PREPROCESSED_DIR / f'{kind}_{output_token}.csv'
        response: dict[str, Any] = {
            'output_path': str(output.resolve()),
            'download_url': f'/api/files?path={output.resolve()}',
            'rows': int(len(frame)),
            'range_mode': range_mode,
            'x_min': x_min,
            'x_max': x_max,
            'curves': result['curves'],
            'preview': modeling_metadata_preview(frame),
            'intensity_summary': _curve_intensity_summary(result['curves']),
            'output_precision': result['output_precision'],
            'warnings': result.get('warnings', []),
        }
        if kind == 'hplc':
            response.update(
                {
                    'baseline_method': None,
                    'hplc_interpolate': hplc_interpolate,
                    'common_time': result.get('common_time', []),
                    'hplc_axis': result.get('hplc_axis'),
                    'x_axis_consistent': result.get('x_axis_consistent', True),
                }
            )
        elif kind == 'raman':
            response['baseline_method'] = baseline_method
        else:
            response['baseline_method'] = None
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output.with_name(f'.{output.name}.{uuid.uuid4().hex}.tmp')
        frame.to_csv(temporary_output, index=False, encoding='utf-8-sig')
        temporary_output.replace(output)
        return response
    except Exception as exc:
        if temporary_output is not None:
            try:
                temporary_output.unlink(missing_ok=True)
            except OSError:
                pass
        raise HTTPException(status_code=400, detail=str(exc)) from exc

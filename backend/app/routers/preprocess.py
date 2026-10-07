# ============================================================================
# 模块说明：backend/app/routers/preprocess.py
#
# 职责：
#   拉曼（raman）、简单色谱（chromatography）与 HPLC 三条预处理管线的
#   HTTP 入口，外加 HPLC 上传前的逐文件解析检测接口。本模块属于 FastAPI
#   路由层，只做“HTTP 边界”工作：接收 multipart 表单、把上传文件复制到
#   受控上传目录、按 kind 分派到算法模块、把结果宽表写成 CSV 并组装下载 URL。
#
# 在系统中的位置：
#   前端预处理页（static/index.html） -> 本路由 ->
#   backend/app/parsers.py（拉曼/简单色谱数学处理）或
#   backend/app/hplc.py（HPLC 固定轴/插值处理） ->
#   storage/preprocessed 下的 wide-feature-v2 宽表 CSV，供训练接口读取。
#
# 关键设计约束：
#   - 预处理统一输出 wide-feature-v2 宽表（前四列 Index/Label/Sample_ID/Name，
#     第 5 列起为严格递增的真实 X 坐标表头，float64 可往返文本，不静默降精度）。
#   - 主色谱界面默认走 /api/preprocess/hplc；旧 chromatography 分支仅作兼容保留。
#   - 上传文件只写入 UPLOADS_DIR 下的随机文件名，不信任客户端文件名，防路径穿越。
#   - 所有算法层异常统一转成 HTTP 400 返回可读信息，不向前端泄露内部堆栈。
# ============================================================================

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

from ..hplc import inspect_hplc_files, preprocess_hplc_files_with_preview
from ..parsers import modeling_metadata_preview, preprocess_raw_files_with_preview
from ..paths import PREPROCESSED_DIR, UPLOADS_DIR

# FastAPI 路由器实例；由 backend/app/main.py 统一 include_router 挂载。
# 本模块所有端点共用此 router，完整路径由各装饰器自带。
router = APIRouter()


def _upload_display_name(file: UploadFile, index: int) -> str:
    """只保留客户端基础文件名，兼容浏览器提交的两种路径分隔符。"""
    # 浏览器可能提交带目录的名字（C:\fakepath\x.csv 或 a/b.csv），
    # 统一把反斜杠归一为正斜杠后取 Path.name，只保留基础文件名；
    # 文件名缺失时回退为 sample_{index}.csv，保证响应中始终有展示名。
    raw_name = str(file.filename or f'sample_{index}.csv').replace('\\', '/')
    return Path(raw_name).name


def _save_upload(file: UploadFile) -> Path:
    """把单个 UploadFile 复制到随机命名的受控上传目录。"""
    # 只取原始扩展名（默认 .csv），目标文件名用 uuid4 随机生成：
    # 客户端文件名不可信，直接拼接会造成路径穿越或互相覆盖。
    suffix = Path(file.filename or 'upload.csv').suffix or '.csv'
    target = UPLOADS_DIR / f'{uuid.uuid4().hex}{suffix}'
    target.parent.mkdir(parents=True, exist_ok=True)
    # 先 seek(0) 再整体读出：UploadFile 底层的 SpooledTemporaryFile
    # 可能已被 FastAPI 读取过，复位指针保证拷贝内容完整。
    with target.open('wb') as handle:
        file.file.seek(0)
        handle.write(file.file.read())
    return target


def _curve_intensity_summary(curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """为每条预览曲线计算强度摘要，供前端快速判断数据质量。

    按 processed_y -> corrected_y -> raw_y 的优先级取“最能代表当前处理
    阶段”的 Y 数组，统计总点数、有限值点数、min/max/mean 与是否全零。

    参数:
        curves: parsers/hplc 返回的曲线字典列表，每项至少含 name 及若干 Y 数组。
    返回:
        与输入等长的字典列表；NaN/Inf 会被剔除后统计，全部无效时
        min/max/mean 为 None 且 all_zero 为 True。
    """
    summary = []
    for curve in curves:
        # 三级回退：优先展示最终处理结果，其次基线校正结果，最后原始强度。
        values = curve.get('processed_y') or curve.get('corrected_y') or curve.get('raw_y') or []
        arr = np.asarray(values, dtype=float)
        # 只统计有限值，避免个别 NaN/Inf 把 min/max/mean 污染成 NaN。
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


@router.post('/api/preprocess/hplc/inspect')
def inspect_hplc_uploads(files: list[UploadFile] = File(...)) -> dict[str, object]:
    """按正式解析规则返回逐文件点数；检测文件不会保留在上传目录。"""
    # 先记录客户端展示名（1 基 index 兜底），落盘后的随机文件名不面向用户。
    original_names = [
        _upload_display_name(file, index)
        for index, file in enumerate(files, start=1)
    ]
    # 检测是临时性的：文件落盘仅为满足解析器“按本地路径读取”的接口要求。
    saved_files = [_save_upload(file) for file in files]
    try:
        # 委托 hplc 模块按正式解析规则逐文件读取点数/时间范围/状态；
        # finally 块会无条件删除临时文件，保证 inspect 不在上传目录留痕。
        return inspect_hplc_files(saved_files, display_names=original_names)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        for path in saved_files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


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
    # 路径参数 kind 白名单校验：三者之外直接 400，
    # 防止未知分支误入错误的算法管线。
    if kind not in {'raman', 'chromatography', 'hplc'}:
        raise HTTPException(status_code=400, detail='kind 必须是 raman、chromatography 或 hplc')
    # 与 inspect 相同：展示名与落盘路径分离，响应里始终用原始文件名。
    original_names = [
        _upload_display_name(file, index)
        for index, file in enumerate(files, start=1)
    ]
    # 上传内容复制到 UPLOADS_DIR 下的随机命名文件，算法层只认本地路径。
    saved_files = [_save_upload(file) for file in files]
    temporary_output: Path | None = None
    try:
        if kind == 'hplc':
            # HPLC 走独立管线：固定轴业务值集中在 HplcGridConfig，
            # hplc_interpolate 开关决定输出固定目标轴插值结果还是所选原始轴导出；
            # 范围选择（行号或 X 轴范围）在插值模式下作用于固定目标轴。
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
            # 拉曼/简单色谱走 parsers 管线：处理顺序固定为先选数据范围
            # （行号或 X 轴范围），再执行基线校正（baseline_method，如 arPLS）。
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
        # frame 即算法层返回的 wide-feature-v2 宽表 DataFrame，后续统一由它写盘。
        frame = result['frame']
        # 输出文件名加随机 token，避免并发请求写同一个 CSV 相互覆盖。
        output_token = uuid.uuid4().hex[:10]
        output = PREPROCESSED_DIR / f'{kind}_{output_token}.csv'
        # 组装响应：下载地址走 /api/files（catalog.py 中按目录白名单校验）；
        # 同时返回曲线预览、建模元数据预览、强度摘要、精度声明与 warnings。
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
                    # HPLC 不做基线校正，固定回显 None；另附公共时间轴、轴一致性与
                    # 逐文件 inspection 结果，供前端展示点数/时间范围/状态并排错。
                    'baseline_method': None,
                    'hplc_interpolate': hplc_interpolate,
                    'common_time': result.get('common_time', []),
                    'hplc_axis': result.get('hplc_axis'),
                    'x_axis_consistent': result.get('x_axis_consistent', True),
                    'inspection': result.get('inspection'),
                }
            )
        # raman 回显实际使用的基线方法；chromatography 无基线概念，回显 None。
        elif kind == 'raman':
            response['baseline_method'] = baseline_method
        else:
            response['baseline_method'] = None
        # 先写隐藏临时文件再 replace 原子替换：进程崩溃或并发时
        # 下载端永远看到完整 CSV，不会读到写了一半的文件。
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output.with_name(f'.{output.name}.{uuid.uuid4().hex}.tmp')
        frame.to_csv(temporary_output, index=False, encoding='utf-8-sig')
        temporary_output.replace(output)
        return response
    # 失败路径：清掉可能已写出的临时文件，再把算法异常统一转成 400。
    except Exception as exc:
        if temporary_output is not None:
            try:
                temporary_output.unlink(missing_ok=True)
            except OSError:
                pass
        raise HTTPException(status_code=400, detail=str(exc)) from exc

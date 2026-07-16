"""HPLC 色谱预处理管线。

处理顺序：
    Step 1: 线性插值到统一时间轴 (解决文件间轻微时间偏移)
    Step 2: 减最小值消负 (逐条曲线)
    Step 3: 按真实时间轴梯形积分面积归一化 (保留峰形比例)

主前端的色谱页面默认调用 `/api/preprocess/hplc` 并开启三步处理。
旧 `/api/preprocess/chromatography` 仍保留为简单范围截取兼容接口。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .parsers import (
    MODELING_COLUMNS,
    _normalize_numeric_array,
    _range_indexer,
    _serialize_modeling_array,
    read_raw_spectrum,
)


def compute_common_time_axis(x_axes: list[np.ndarray]) -> np.ndarray:
    """从多个时间轴中计算统一时间轴。

    策略：
      1. 取重叠区域: start = max(min(x)), end = min(max(x))
      2. 点数取中位数: n_points = median(len(x))
      3. 返回 linspace(start, end, n_points)

    不硬编码点数（如 7500），自动从数据推断。
    """
    if len(x_axes) == 1:
        return x_axes[0].astype(np.float32).copy()
    # 只在所有曲线共同覆盖的交集内插值，避免 np.interp 在边界做常数外推。
    starts = [x.min() for x in x_axes]
    ends = [x.max() for x in x_axes]
    start = max(starts)
    end = min(ends)
    if start >= end:
        raise ValueError(
            f"各文件时间轴无重叠区域: 起始范围 [{min(starts):.6f}, {max(starts):.6f}], "
            f"结束范围 [{min(ends):.6f}, {max(ends):.6f}]"
        )
    # 中位数长度不被单个异常长/短文件主导，同时保持与原采样密度接近。
    lengths = [len(x) for x in x_axes]
    n_points = int(np.median(lengths))
    return np.linspace(start, end, n_points, dtype=np.float32)


def hplc_interpolate(
    files_data: list[tuple[np.ndarray, np.ndarray]],
    common_x: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Step 1: 线性插值到统一时间轴。

    Args:
        files_data: [(x_i, y_i), ...] 各文件的 (时间, 强度) 对
        common_x: 共用时间轴。None 时自动通过 compute_common_time_axis 计算

    Returns:
        aligned: (n_files, n_points) float32 对齐后的强度矩阵
        common_x: (n_points,) float32 统一时间轴
    """
    if common_x is None:
        common_x = compute_common_time_axis([x for x, _ in files_data])
    aligned = np.empty((len(files_data), len(common_x)), dtype=np.float32)
    for i, (x, y) in enumerate(files_data):
        aligned[i] = np.interp(common_x, x, y).astype(np.float32)
    return aligned, common_x


def hplc_subtract_min(matrix: np.ndarray) -> np.ndarray:
    """Step 2: 逐行减最小值，消除负值。

    每行 matrix[i] -= matrix[i].min()，保证所有值 ≥ 0。
    """
    mins = matrix.min(axis=1, keepdims=True)
    return (matrix - mins).astype(np.float32)


def hplc_normalize_area(
    matrix: np.ndarray,
    x_axis: np.ndarray | None = None,
) -> np.ndarray:
    """Step 3: 逐行除以总面积（梯形积分法）。

    默认按点序号积分；传入 x_axis 时按真实时间轴积分。
    零面积保护: 除以 max(area, 1e-12)。
    """
    if x_axis is not None:
        areas = np.trapezoid(matrix, x=x_axis, axis=1)
    else:
        areas = np.trapezoid(matrix, axis=1)
    areas = np.maximum(areas, 1e-12)
    return (matrix / areas[:, np.newaxis]).astype(np.float32)


def preprocess_hplc_files_with_preview(
    files: Iterable[str | Path],
    start_row: int = 1,
    end_row: int | None = None,
    range_mode: str = "row",
    x_min: float | None = None,
    x_max: float | None = None,
    display_names: list[str] | None = None,
    interpolate: bool = True,
    subtract_min: bool = True,
    normalize_area: bool = True,
) -> dict:
    """HPLC 预处理管线入口，生成统一建模 CSV 和预览曲线。

    处理顺序:
      1. 读取原始 CSV + 范围选择
      2. [可选] 线性插值到统一时间轴
      3. [可选] 减最小值消负
      4. [可选] 按 common_x 面积归一化

    Args:
        files: 原始 CSV 文件路径列表
        start_row, end_row: 行号范围（1-based）
        range_mode: "row" 或 "x_value"
        x_min, x_max: X 轴数值范围
        display_names: 显示名称列表
        interpolate: 是否启用 Step 1 线性插值
        subtract_min: 是否启用 Step 2 消负
        normalize_area: 是否启用 Step 3 面积归一化

    Returns:
        {
            "frame": pd.DataFrame,
            "curves": [{"name", "x", "raw_y", "processed_y"}, ...],
            "common_time": list[float],
        }
    """
    if not files:
        raise ValueError("至少需要上传一个文件")
    files = list(files)

    # ---- 阶段 1: 逐文件读取 + 范围选择 ----
    raw_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    names: list[str] = []
    for idx, file_path in enumerate(files):
        path = Path(file_path)
        name = (
            display_names[idx]
            if display_names and idx < len(display_names)
            else path.stem
        )
        names.append(name)
        full_x, full_y = read_raw_spectrum(path, kind="hplc")
        indexer, _ = _range_indexer(full_x, start_row, end_row, range_mode, x_min, x_max)
        x_i = full_x[indexer]
        y_i = full_y[indexer]
        if len(x_i) == 0:
            raise ValueError(f"{path.name} 在所选范围内没有数据")
        raw_pairs.append((x_i, y_i))

    # ---- 阶段 2: 统一处理 ----
    if interpolate:
        matrix, common_x = hplc_interpolate(raw_pairs)
    else:
        # 验证所有 x 长度一致且时间轴一致，否则同一 common_x 会错配强度。
        lengths = {len(x) for x, _ in raw_pairs}
        if len(lengths) != 1:
            raise ValueError(
                f"关闭插值时要求所有文件曲线长度一致，当前长度: {sorted(lengths)}"
            )
        reference_x = raw_pairs[0][0]
        for x_i, _ in raw_pairs[1:]:
            if not np.allclose(x_i, reference_x, rtol=1e-5, atol=1e-8):
                raise ValueError("关闭插值时要求所有文件时间轴一致，请开启线性插值")
        common_x = raw_pairs[0][0].astype(np.float32)
        matrix = np.stack([y for _, y in raw_pairs], axis=0).astype(np.float32)

    # 保存插值后/原始强度（消负和归一化前的中间态）
    raw_interpolated = matrix.copy()

    if subtract_min:
        matrix = hplc_subtract_min(matrix)

    if normalize_area:
        matrix = hplc_normalize_area(matrix, common_x)

    # ---- 阶段 3: 组装输出 ----
    n_files = len(files)
    records = []
    common_x_values, common_x_serialized = _serialize_modeling_array(
        common_x, "XXX", names[0]
    )
    processed_values: list[list[float]] = []
    for i in range(n_files):
        intensity_values, intensity_serialized = _serialize_modeling_array(
            matrix[i], "Intensity", names[i]
        )
        processed_values.append(intensity_values)
        records.append(
            {
                "Index": i + 1,
                "Name": names[i],
                "XXX": common_x_serialized,
                "Intensity": intensity_serialized,
                "Label": "",
                "Sample_ID": "",
            }
        )

    frame = pd.DataFrame.from_records(
        records, columns=MODELING_COLUMNS
    )

    curves = []
    for i in range(n_files):
        curve_data: dict = {
            "name": names[i],
            "x": common_x_values,
            "raw_y": _normalize_numeric_array(raw_interpolated[i], "raw_y"),
            "processed_y": processed_values[i],
        }
        curves.append(curve_data)

    return {
        "frame": frame,
        "curves": curves,
        "common_time": common_x_values,
    }

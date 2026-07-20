"""HPLC 色谱固定时间轴线性映射管线。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .parsers import (
    EXCEL_CELL_CHARACTER_LIMIT,
    MAX_OUTPUT_DECIMAL_PLACES,
    MODELING_COLUMNS,
    _output_precision_metadata,
    _range_indexer,
    _serialize_modeling_arrays,
    read_raw_spectrum,
)


@dataclass(frozen=True)
class HplcGridConfig:
    """HPLC 固定轴配置；算法函数只消费配置，不内嵌业务数值。"""

    start_minutes: float
    stop_minutes: float
    point_count: int
    tolerance_minutes: float
    unit: str = "minute"

    @property
    def step_minutes(self) -> float:
        return (self.stop_minutes - self.start_minutes) / (self.point_count - 1)


DEFAULT_HPLC_GRID = HplcGridConfig(
    start_minutes=0.0,
    stop_minutes=50.0,
    point_count=7500,
    tolerance_minutes=1e-8,
)


def build_hplc_target_axis(config: HplcGridConfig = DEFAULT_HPLC_GRID) -> np.ndarray:
    """根据注入配置构造包含首尾端点的固定 float64 时间轴。"""
    if config.point_count < 2:
        raise ValueError("HPLC 固定轴点数必须至少为 2")
    if not np.isfinite(config.start_minutes) or not np.isfinite(config.stop_minutes):
        raise ValueError("HPLC 固定轴范围必须是有限数值")
    if config.start_minutes >= config.stop_minutes:
        raise ValueError("HPLC 固定轴起点必须小于终点")
    if config.tolerance_minutes < 0 or not np.isfinite(config.tolerance_minutes):
        raise ValueError("HPLC 固定轴容差必须是非负有限数值")
    return np.linspace(
        config.start_minutes,
        config.stop_minutes,
        config.point_count,
        dtype=np.float64,
    )


def _validated_hplc_source(
    source_x: np.ndarray,
    source_y: np.ndarray,
    source_name: str,
    config: HplcGridConfig,
    *,
    require_point_count: bool = True,
    require_coverage: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(source_x, dtype=np.float64).copy()
    y = np.asarray(source_y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError(f"{source_name} 的时间轴和强度必须是一维且长度一致")
    if require_point_count and len(x) != config.point_count:
        raise ValueError(
            f"{source_name} 解析到 {len(x)} 个有效色谱点，HPLC 固定流程要求恰好 "
            f"{config.point_count} 个点；未生成结果文件"
        )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"{source_name} 含有 NaN 或无穷值；未生成结果文件")
    differences = np.diff(x)
    invalid = np.flatnonzero(differences <= 0)
    if invalid.size:
        index = int(invalid[0])
        raise ValueError(
            f"{source_name} 的时间轴在第 {index + 1}、{index + 2} 个点不严格递增"
            f"（{x[index]:.12g}、{x[index + 1]:.12g} {config.unit}）；未生成结果文件"
        )
    if abs(x[0] - config.start_minutes) <= config.tolerance_minutes:
        x[0] = config.start_minutes
    if abs(x[-1] - config.stop_minutes) <= config.tolerance_minutes:
        x[-1] = config.stop_minutes
    if np.any(np.diff(x) <= 0):
        raise ValueError(
            f"{source_name} 的端点在按容差对齐后不再严格递增；请检查时间轴起止点"
        )
    if require_coverage and (x[0] > config.start_minutes or x[-1] < config.stop_minutes):
        raise ValueError(
            f"{source_name} 的时间范围为 {x[0]:.12g}–{x[-1]:.12g} {config.unit}，"
            f"不能覆盖固定目标范围 {config.start_minutes:g}–{config.stop_minutes:g} "
            f"{config.unit}；禁止外推"
        )
    return x, y


def map_hplc_intensity(
    source_x: np.ndarray,
    source_y: np.ndarray,
    source_name: str,
    config: HplcGridConfig = DEFAULT_HPLC_GRID,
    target_x: np.ndarray | None = None,
) -> np.ndarray:
    """把源强度映射到固定轴或其范围切片，边界仅允许一个采样间隔内线性延伸。"""
    x, y = _validated_hplc_source(
        source_x,
        source_y,
        source_name,
        config,
        require_coverage=False,
    )
    target = build_hplc_target_axis(config) if target_x is None else np.asarray(target_x, dtype=np.float64)
    if target.ndim != 1 or len(target) < 1 or not np.all(np.isfinite(target)):
        raise ValueError("HPLC 目标时间轴必须是一维非空有限数组")
    if len(target) > 1 and np.any(np.diff(target) <= 0):
        raise ValueError("HPLC 目标时间轴必须严格递增")

    source_step = float(np.median(np.diff(x)))
    target_step = float(np.median(np.diff(target))) if len(target) > 1 else config.step_minutes
    max_edge_gap = max(source_step, target_step) + config.tolerance_minutes
    left_gap = max(0.0, float(x[0] - target[0]))
    right_gap = max(0.0, float(target[-1] - x[-1]))
    if left_gap > max_edge_gap or right_gap > max_edge_gap:
        raise ValueError(
            f"{source_name} 的时间范围为 {x[0]:.12g}–{x[-1]:.12g} {config.unit}，"
            f"与所选固定目标范围 {target[0]:.12g}–{target[-1]:.12g} {config.unit} "
            "相差超过一个采样间隔，无法安全线性映射"
        )

    mapped = np.interp(target, x, y)
    left = target < x[0]
    if np.any(left):
        slope = (y[1] - y[0]) / (x[1] - x[0])
        mapped[left] = y[0] + (target[left] - x[0]) * slope
    right = target > x[-1]
    if np.any(right):
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        mapped[right] = y[-1] + (target[right] - x[-1]) * slope
    return mapped


def hplc_x_axes_consistent(x_axes: list[np.ndarray]) -> bool:
    """判断关闭插值后各文件导出的 X 轴长度和坐标是否一致。"""
    if len(x_axes) <= 1:
        return True
    reference = x_axes[0]
    return all(
        len(axis) == len(reference)
        and np.allclose(axis, reference, rtol=1e-5, atol=1e-8)
        for axis in x_axes[1:]
    )


def _axis_descriptor(target_x: np.ndarray, config: HplcGridConfig) -> str:
    return json.dumps(
        {
            "type": "linspace-v1",
            "start": float(target_x[0]),
            "stop": float(target_x[-1]),
            "count": int(len(target_x)),
            "unit": config.unit,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _axis_metadata(target_x: np.ndarray, config: HplcGridConfig) -> dict[str, int | float | str]:
    return {
        "start": float(target_x[0]),
        "stop": float(target_x[-1]),
        "unit": config.unit,
        "point_count": int(len(target_x)),
        "step_minutes": config.step_minutes,
        "mapping": "piecewise_linear",
        "input_point_count_required": config.point_count,
        "encoding": "linspace-v1",
    }


def preprocess_hplc_files_with_preview(
    files: Iterable[str | Path],
    start_row: int = 1,
    end_row: int | None = None,
    range_mode: str = "row",
    x_min: float | None = None,
    x_max: float | None = None,
    display_names: list[str] | None = None,
    interpolate: bool = True,
    config: HplcGridConfig = DEFAULT_HPLC_GRID,
) -> dict:
    """保留既有范围/开关交互；开启时映射到配置轴，关闭时导出所选原轴。"""
    if not files:
        raise ValueError("至少需要上传一个文件")
    files = list(files)

    full_target_x = build_hplc_target_axis(config)
    target_indexer, target_range_label = _range_indexer(
        full_target_x,
        start_row,
        end_row,
        range_mode,
        x_min,
        x_max,
    )
    target_x = full_target_x[target_indexer]
    if interpolate and len(target_x) < 2:
        raise ValueError(f"所选{target_range_label}在固定时间轴上少于 2 个点，无法线性插值")
    full_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    selected_pairs: list[tuple[np.ndarray, np.ndarray]] = []
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
        validated_x, validated_y = _validated_hplc_source(
            full_x,
            full_y,
            path.name,
            config,
            require_coverage=False,
        )
        full_pairs.append((validated_x, validated_y))
        indexer, range_label = _range_indexer(
            validated_x,
            start_row,
            end_row,
            range_mode,
            x_min,
            x_max,
        )
        selected_x = validated_x[indexer]
        selected_y = validated_y[indexer]
        if len(selected_x) == 0:
            raise ValueError(f"{path.name} 在所选{range_label}内没有数据")
        selected_pairs.append((selected_x, selected_y))

    warnings: list[str] = []
    if interpolate:
        mapped_intensities = [
            map_hplc_intensity(
                full_x,
                full_y,
                Path(file_path).name,
                config,
                target_x,
            )
            for (full_x, full_y), file_path in zip(full_pairs, files)
        ]
        x_arrays = [target_x] * len(selected_pairs)
        intensity_arrays = mapped_intensities
        x_axis_consistent = True
    else:
        x_arrays = [pair[0] for pair in selected_pairs]
        intensity_arrays = [pair[1] for pair in selected_pairs]
        x_axis_consistent = hplc_x_axes_consistent(x_arrays)
        if not x_axis_consistent:
            warnings.append(
                "已关闭线性插值，检测到各文件 X 轴或点数不一致；CSV 已按各文件所选原始 X 轴正常生成，"
                "后续建模前请确认这些曲线可以直接比较。"
            )

    n_files = len(files)
    records = []
    intensity_result = _serialize_modeling_arrays(
        intensity_arrays, "Intensity", names
    )
    axis_descriptor = _axis_descriptor(target_x, config) if interpolate else ""
    x_result = None if interpolate else _serialize_modeling_arrays(x_arrays, "XXX", names)
    processed_values: list[list[int | float]] = []
    for i in range(n_files):
        intensity_item = intensity_result.arrays[i]
        processed_values.append(intensity_item.values)
        records.append(
            {
                "Index": i + 1,
                "Name": names[i],
                "XXX": axis_descriptor if interpolate else x_result.arrays[i].serialized,
                "Intensity": intensity_item.serialized,
                "Label": "",
                "Sample_ID": "",
            }
        )

    frame = pd.DataFrame.from_records(
        records, columns=MODELING_COLUMNS
    )

    curves = []
    target_values = target_x.tolist()
    for i in range(n_files):
        curve_x = target_values if interpolate else x_result.arrays[i].values
        curve_data: dict = {
            "name": names[i],
            "x": curve_x,
            "raw_y": processed_values[i],
            "processed_y": processed_values[i],
        }
        curves.append(curve_data)

    return {
        "frame": frame,
        "curves": curves,
        "common_time": target_values if interpolate else [],
        "hplc_axis": _axis_metadata(target_x, config) if interpolate else None,
        "x_axis_consistent": x_axis_consistent,
        "warnings": warnings,
        "output_precision": (
            {
                "adaptive": True,
                "max_decimal_places": MAX_OUTPUT_DECIMAL_PLACES,
                "xxx_encoding": "linspace-v1",
                "xxx_max_characters": len(axis_descriptor),
                "intensity_decimal_places": intensity_result.decimal_places,
                "intensity_max_characters": intensity_result.max_characters,
                "excel_cell_character_limit": EXCEL_CELL_CHARACTER_LIMIT,
            }
            if interpolate
            else _output_precision_metadata(x_result, intensity_result)
        ),
    }

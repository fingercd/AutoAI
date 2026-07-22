"""HPLC 色谱固定时间轴线性映射管线。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .parsers import (
    WIDE_AXIS_ENCODING,
    build_wide_modeling_frame,
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


@dataclass(frozen=True)
class HplcAxisSelection:
    """固定 HPLC 网格上的连续范围选择结果。"""

    full_axis: np.ndarray
    indices: np.ndarray
    target_x: np.ndarray
    offset: int
    length: int
    selected_start_row: int
    selected_end_row: int
    range_label: str


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


def _validated_hplc_row_range(
    start_row: int,
    end_row: int | None,
    config: HplcGridConfig,
) -> tuple[int, int]:
    """校验 HPLC 专用的 1 基、首尾包含行号范围。"""

    def validate(value: object, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(
                f"HPLC {label}必须是 1 到 {config.point_count} 的整数；当前为 {value!r}"
            )
        row = int(value)
        if not 1 <= row <= config.point_count:
            raise ValueError(
                f"HPLC {label}必须在 1 到 {config.point_count} 之间；当前为 {row}"
            )
        return row

    start = validate(start_row, "起始行")
    end = config.point_count if end_row is None else validate(end_row, "终止行")
    if start > end:
        raise ValueError(
            f"HPLC 起始行不能大于终止行；当前为 {start}–{end}，"
            f"允许范围为 1–{config.point_count}"
        )
    return start, end


def _validated_hplc_time_bounds(
    x_min: float | None,
    x_max: float | None,
) -> tuple[float, float]:
    """把可选的真实保留时间边界规范化为有限闭区间边界。"""

    def validate(value: object, label: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"HPLC 保留时间{label}必须是有限数值；当前为 {value!r}")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"HPLC 保留时间{label}必须是有限数值；当前为 {value!r}"
            ) from exc
        if not np.isfinite(number):
            raise ValueError(f"HPLC 保留时间{label}必须是有限数值；当前为 {value!r}")
        return number

    lower = float("-inf") if x_min is None else validate(x_min, "下限")
    upper = float("inf") if x_max is None else validate(x_max, "上限")
    if lower > upper:
        raise ValueError(
            f"HPLC 保留时间下限不能大于上限；当前为 {lower:.12g}–{upper:.12g} 分钟"
        )
    return lower, upper


def _select_hplc_target_axis(
    *,
    start_row: int = 1,
    end_row: int | None = None,
    range_mode: str = "row",
    x_min: float | None = None,
    x_max: float | None = None,
    config: HplcGridConfig = DEFAULT_HPLC_GRID,
) -> HplcAxisSelection:
    """在完整固定分钟轴上严格选择连续点，并保留原始整数位置。"""

    full_axis = build_hplc_target_axis(config)
    positions = np.arange(config.point_count, dtype=np.int64)
    if range_mode == "row":
        selected_start, selected_end = _validated_hplc_row_range(
            start_row, end_row, config
        )
        indices = positions[selected_start - 1 : selected_end]
        range_label = "行范围"
    elif range_mode == "x_value":
        lower, upper = _validated_hplc_time_bounds(x_min, x_max)
        indices = positions[(full_axis >= lower) & (full_axis <= upper)]
        range_label = "保留时间范围"
    else:
        raise ValueError("HPLC range_mode 必须是 row 或 x_value")

    if indices.size == 0:
        raise ValueError(f"所选{range_label}在固定 HPLC 时间轴上没有数据")
    if indices.size > 1 and np.any(np.diff(indices) != 1):
        raise ValueError(f"所选{range_label}在固定 HPLC 时间轴上不是连续范围")

    offset = int(indices[0])
    length = int(indices.size)
    target_x = full_axis[indices]
    return HplcAxisSelection(
        full_axis=full_axis,
        indices=indices,
        target_x=target_x,
        offset=offset,
        length=length,
        selected_start_row=offset + 1,
        selected_end_row=offset + length,
        range_label=range_label,
    )


def _select_hplc_original_axis(
    source_x: np.ndarray,
    *,
    start_row: int,
    end_row: int | None,
    range_mode: str,
    x_min: float | None,
    x_max: float | None,
    config: HplcGridConfig,
) -> tuple[np.ndarray, str]:
    """关闭插值时，在每条完整原始轴上应用同一严格请求范围。"""

    if range_mode == "row":
        selected_start, selected_end = _validated_hplc_row_range(
            start_row, end_row, config
        )
        return np.arange(selected_start - 1, selected_end, dtype=np.int64), "行范围"
    if range_mode == "x_value":
        lower, upper = _validated_hplc_time_bounds(x_min, x_max)
        return np.flatnonzero((source_x >= lower) & (source_x <= upper)), "保留时间范围"
    raise ValueError("HPLC range_mode 必须是 row 或 x_value")


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


def _axis_metadata(
    selection: HplcAxisSelection,
    config: HplcGridConfig,
) -> dict[str, int | float | str]:
    target_x = selection.target_x
    return {
        "start": float(target_x[0]),
        "stop": float(target_x[-1]),
        "unit": config.unit,
        "point_count": int(len(target_x)),
        "step_minutes": config.step_minutes,
        "mapping": "piecewise_linear",
        "input_point_count_required": config.point_count,
        "encoding": WIDE_AXIS_ENCODING,
        "grid_start": float(config.start_minutes),
        "grid_stop": float(config.stop_minutes),
        "grid_point_count": int(config.point_count),
        "selected_start_row": selection.selected_start_row,
        "selected_end_row": selection.selected_end_row,
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

    if interpolate:
        selection = _select_hplc_target_axis(
            start_row=start_row,
            end_row=end_row,
            range_mode=range_mode,
            x_min=x_min,
            x_max=x_max,
            config=config,
        )
        if selection.length < 2:
            raise ValueError(
                f"所选{selection.range_label}在固定时间轴上少于 2 个点，无法线性插值"
            )
    else:
        # 关闭插值时不套用固定轴边界，但配置和请求本身仍需先严格校验。
        build_hplc_target_axis(config)
        if range_mode == "row":
            _validated_hplc_row_range(start_row, end_row, config)
        elif range_mode == "x_value":
            _validated_hplc_time_bounds(x_min, x_max)
        else:
            raise ValueError("HPLC range_mode 必须是 row 或 x_value")
        selection = None
    full_pairs: list[tuple[np.ndarray, np.ndarray]] = []
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

    warnings: list[str] = []
    if interpolate:
        assert selection is not None
        target_x = selection.target_x
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
        x_arrays = [target_x] * len(full_pairs)
        intensity_arrays = mapped_intensities
        x_axis_consistent = True
    else:
        selected_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for (validated_x, validated_y), file_path in zip(full_pairs, files):
            indices, range_label = _select_hplc_original_axis(
                validated_x,
                start_row=start_row,
                end_row=end_row,
                range_mode=range_mode,
                x_min=x_min,
                x_max=x_max,
                config=config,
            )
            selected_x = validated_x[indices]
            selected_y = validated_y[indices]
            if len(selected_x) == 0:
                raise ValueError(f"{Path(file_path).name} 在所选{range_label}内没有数据")
            selected_pairs.append((selected_x, selected_y))
        x_arrays = [pair[0] for pair in selected_pairs]
        intensity_arrays = [pair[1] for pair in selected_pairs]
        x_axis_consistent = hplc_x_axes_consistent(x_arrays)

    n_files = len(files)
    try:
        wide = build_wide_modeling_frame(
            indices=list(range(1, n_files + 1)),
            x_arrays=x_arrays,
            intensity_arrays=intensity_arrays,
            source_names=names,
        )
    except ValueError as exc:
        if not interpolate and "XXX 与" in str(exc):
            raise ValueError(
                f"{exc}；当前已关闭 HPLC 线性插值，请开启插值后重试"
            ) from exc
        raise
    frame = wide.frame
    processed_values = wide.intensity
    # 能成功写成一张宽表就必然只有一条共享真实轴。
    x_axis_consistent = True

    curves = []
    target_values = wide.x_axis if selection is not None else []
    for i in range(n_files):
        curve_x = wide.x_axis
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
        "hplc_axis": _axis_metadata(selection, config) if selection is not None else None,
        "x_axis_consistent": x_axis_consistent,
        "warnings": warnings,
        "output_precision": wide.output_precision,
    }

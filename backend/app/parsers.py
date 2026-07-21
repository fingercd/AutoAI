"""建模 CSV 与原始拉曼/色谱 CSV 的解析和统一导出。

统一建模行固定为 ``Index, Name, XXX, Intensity, Label, Sample_ID``。
数组序列化采用紧凑 JSON，并在最多 5 位小数内自动选择满足 Excel 32,767
字符单元格上限的最高批次统一精度；无法安全容纳时明确拒绝，绝不静默截断或
降采样。拉曼流程固定先选择范围，再执行基线校正。
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MODELING_COLUMNS = ("Index", "Name", "XXX", "Intensity", "Label", "Sample_ID")
REQUIRED_MODELING_COLUMNS = set(MODELING_COLUMNS)
LEGACY_SAMPLE_ID_COLUMN = "Repeat_index"
EXCEL_CELL_CHARACTER_LIMIT = 32_767
MAX_OUTPUT_DECIMAL_PLACES = 5
MAX_AXIS_DESCRIPTOR_POINTS = 1_000_000
# 兼容既有内部引用；新代码应使用语义更明确的 MAX_OUTPUT_DECIMAL_PLACES。
OUTPUT_DECIMAL_PLACES = MAX_OUTPUT_DECIMAL_PLACES


@dataclass
class ModelingDataset:
    """解析后的表格元数据、二维强度矩阵和逐行 X 轴。"""
    frame: pd.DataFrame
    x_axis: list[list[float]]
    intensity: np.ndarray
    labels: list[str]
    sample_id: list[str]


@dataclass(frozen=True)
class SerializedModelingArray:
    """一个数组在指定小数精度下的最终建模 JSON 表示。"""

    values: list[int | float]
    serialized: str
    decimal_places: int
    character_count: int


@dataclass(frozen=True)
class BatchSerializationResult:
    """同一字段在一个预处理批次内采用统一精度后的结果。"""

    arrays: list[SerializedModelingArray]
    decimal_places: int
    max_characters: int
    max_source_name: str


def _parse_array(value: object, field: str, row_number: int) -> list[float]:
    if isinstance(value, list):
        raw = value
    else:
        try:
            raw = ast.literal_eval(str(value))
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"第 {row_number} 行 {field} 不是有效数组") from exc
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"第 {row_number} 行 {field} 必须是非空数组")
    try:
        return [float(item) for item in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"第 {row_number} 行 {field} 含有非数字值") from exc


def _parse_modeling_axis(value: object, row_number: int) -> list[float]:
    """读取旧式数值数组、历史等距轴或固定分钟网格切片描述。"""
    if isinstance(value, dict):
        raw = value
    elif isinstance(value, (list, tuple)):
        return _parse_array(value, "XXX", row_number)
    else:
        try:
            raw = ast.literal_eval(str(value))
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"第 {row_number} 行 XXX 不是有效数组或轴描述") from exc
    if not isinstance(raw, dict):
        return _parse_array(raw, "XXX", row_number)
    descriptor_type = raw.get("type")
    if descriptor_type == "linspace-slice-v1":
        expected_fields = {
            "type",
            "grid_start",
            "grid_stop",
            "grid_count",
            "offset",
            "length",
            "unit",
        }
        actual_fields = set(raw)
        if actual_fields != expected_fields:
            missing = sorted(expected_fields - actual_fields)
            extra = sorted(str(item) for item in actual_fields - expected_fields)
            details = []
            if missing:
                details.append(f"缺少字段 {', '.join(missing)}")
            if extra:
                details.append(f"包含不支持字段 {', '.join(extra)}")
            raise ValueError(
                f"第 {row_number} 行 XXX 的 linspace-slice-v1 描述无效："
                + "；".join(details)
            )

        def finite_number(field: str) -> float:
            field_value = raw[field]
            if isinstance(field_value, bool) or not isinstance(
                field_value, (int, float, np.integer, np.floating)
            ):
                raise ValueError(
                    f"第 {row_number} 行 XXX 的 {field} 必须是有限数值"
                )
            number = float(field_value)
            if not np.isfinite(number):
                raise ValueError(
                    f"第 {row_number} 行 XXX 的 {field} 必须是有限数值"
                )
            return number

        def true_integer(field: str) -> int:
            field_value = raw[field]
            if isinstance(field_value, bool) or not isinstance(
                field_value, (int, np.integer)
            ):
                raise ValueError(f"第 {row_number} 行 XXX 的 {field} 必须是整数")
            return int(field_value)

        grid_start = finite_number("grid_start")
        grid_stop = finite_number("grid_stop")
        grid_count = true_integer("grid_count")
        offset = true_integer("offset")
        length = true_integer("length")
        if grid_start >= grid_stop:
            raise ValueError(
                f"第 {row_number} 行 XXX 的 grid_start/grid_stop 必须严格递增"
            )
        if not 2 <= grid_count <= MAX_AXIS_DESCRIPTOR_POINTS:
            raise ValueError(
                f"第 {row_number} 行 XXX 的 grid_count 必须在 2 到 "
                f"{MAX_AXIS_DESCRIPTOR_POINTS} 之间"
            )
        if offset < 0:
            raise ValueError(f"第 {row_number} 行 XXX 的 offset 不能小于 0")
        if length < 1:
            raise ValueError(f"第 {row_number} 行 XXX 的 length 必须至少为 1")
        if offset + length > grid_count:
            raise ValueError(
                f"第 {row_number} 行 XXX 的 offset + length 不能超过 grid_count"
            )
        if raw["unit"] != "minute":
            raise ValueError(
                f"第 {row_number} 行 XXX 的 linspace-slice-v1 unit 必须是 minute"
            )

        with np.errstate(over="ignore", invalid="ignore"):
            full_axis = np.linspace(
                grid_start, grid_stop, grid_count, dtype=np.float64
            )
        if not np.all(np.isfinite(full_axis)) or np.any(np.diff(full_axis) <= 0):
            raise ValueError(
                f"第 {row_number} 行 XXX 的完整固定轴必须有限且严格递增"
            )
        selected_axis = full_axis[offset : offset + length]
        if (
            selected_axis.size == 0
            or not np.all(np.isfinite(selected_axis))
            or (selected_axis.size > 1 and np.any(np.diff(selected_axis) <= 0))
        ):
            raise ValueError(
                f"第 {row_number} 行 XXX 展开的时间轴必须非空、有限且严格递增"
            )
        return selected_axis.tolist()

    if descriptor_type != "linspace-v1":
        raise ValueError(f"第 {row_number} 行 XXX 使用了不支持的轴描述类型")
    try:
        start = float(raw["start"])
        stop = float(raw["stop"])
        count_raw = raw["count"]
        count = int(count_raw)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"第 {row_number} 行 XXX 的等距轴描述参数无效") from exc
    if isinstance(count_raw, bool) or float(count_raw) != count:
        raise ValueError(f"第 {row_number} 行 XXX 的 count 必须是整数")
    if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
        raise ValueError(f"第 {row_number} 行 XXX 的 start/stop 必须是有限递增数值")
    if not 2 <= count <= MAX_AXIS_DESCRIPTOR_POINTS:
        raise ValueError(
            f"第 {row_number} 行 XXX 的 count 必须在 2 到 {MAX_AXIS_DESCRIPTOR_POINTS} 之间"
        )
    unit = raw.get("unit")
    if unit is not None and (not isinstance(unit, str) or not unit.strip()):
        raise ValueError(f"第 {row_number} 行 XXX 的 unit 必须是非空字符串")
    return np.linspace(start, stop, count, dtype=np.float64).tolist()


def _normalise_sample_id_column(frame: pd.DataFrame) -> pd.DataFrame:
    """把旧分组列收敛为 Sample_ID；冲突双列拒绝猜测。"""
    has_current = "Sample_ID" in frame.columns
    has_legacy = LEGACY_SAMPLE_ID_COLUMN in frame.columns
    if has_current and has_legacy:
        current = frame["Sample_ID"].astype("string").fillna("").str.strip()
        legacy = frame[LEGACY_SAMPLE_ID_COLUMN].astype("string").fillna("").str.strip()
        mismatch = current.ne(legacy)
        if bool(mismatch.any()):
            rows = ", ".join(str(int(index) + 2) for index in frame.index[mismatch][:5])
            raise ValueError(f"Sample_ID 与旧分组列内容不一致，请检查第 {rows} 行")
        return frame.drop(columns=[LEGACY_SAMPLE_ID_COLUMN])
    if has_legacy:
        return frame.rename(columns={LEGACY_SAMPLE_ID_COLUMN: "Sample_ID"})
    return frame


def load_modeling_csv(path: str | Path) -> ModelingDataset:
    """解析并严格校验统一建模 CSV，返回可直接训练的数据结构。"""
    path = Path(path)
    frame = _read_csv_flexible(path)
    frame = _normalise_sample_id_column(frame)
    if "Index" not in frame.columns and REQUIRED_MODELING_COLUMNS.difference({"Index"}).issubset(frame.columns):
        frame = frame.rename(columns={frame.columns[0]: "Index"})
    missing = REQUIRED_MODELING_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"建模数据缺少字段: {', '.join(sorted(missing))}")

    x_axis: list[list[float]] = []
    y_values: list[list[float]] = []
    labels: list[str] = []
    sample_ids: list[str] = []
    for idx, row in frame.iterrows():
        row_number = idx + 2
        x = _parse_modeling_axis(row["XXX"], row_number)
        y = _parse_array(row["Intensity"], "Intensity", row_number)
        if len(x) != len(y):
            raise ValueError(f"第 {row_number} 行 XXX 和 Intensity 长度不一致")
        label = str(row["Label"]).strip()
        if not label or label.lower() == "nan":
            raise ValueError(f"第 {row_number} 行 Label 为空，建模前请补充标签")
        sample_id = str(row["Sample_ID"]).strip()
        if not sample_id or sample_id.lower() == "nan":
            raise ValueError(f"第 {row_number} 行 Sample_ID 为空，建模前请补充样品编号")
        x_axis.append(x)
        y_values.append(y)
        labels.append(label)
        sample_ids.append(sample_id)

    lengths = {len(values) for values in y_values}
    if len(lengths) != 1:
        raise ValueError(f"当前训练版本要求曲线长度一致，检测到长度: {sorted(lengths)}")

    frame = frame.copy()
    frame["Label"] = labels
    frame["Sample_ID"] = sample_ids
    sample_summary = _sample_id_summary(frame)
    if sample_summary["inconsistent_labels"]:
        details = ", ".join(f"{item['sample_id']}={item['labels']}" for item in sample_summary["inconsistent_labels"])
        raise ValueError(f"同一个 Sample_ID 内出现多个 Label，请检查: {details}")
    if sample_summary["incomplete_groups"]:
        expected = sample_summary["expected_repeats_per_group"]
        details = ", ".join(f"{item['sample_id']}={item['count']}" for item in sample_summary["incomplete_groups"])
        raise ValueError(f"Sample_ID 重复测量次数不一致，期望每组 {expected} 条，异常分组: {details}")

    return ModelingDataset(
        frame=frame,
        x_axis=x_axis,
        intensity=np.asarray(y_values, dtype=np.float32),
        labels=labels,
        sample_id=sample_ids,
    )


def natural_sort_key(value: object) -> tuple[tuple[int, object], ...]:
    """生成稳定自然排序键，使 2 排在 10 前，并兼容 S2/S10。"""
    parts = re.split(r"(\d+)", str(value).strip())
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in parts
        if part
    )


def _sample_id_summary(frame: pd.DataFrame) -> dict:
    grouped = frame.groupby("Sample_ID", sort=False)
    group_rows = []
    inconsistent_labels = []
    for sample_id in sorted(grouped.groups, key=natural_sort_key):
        group = grouped.get_group(sample_id)
        labels = sorted(
            (str(item) for item in group["Label"].dropna().astype(str).unique()),
            key=natural_sort_key,
        )
        count = int(len(group))
        group_rows.append({"sample_id": str(sample_id), "count": count, "label": labels[0] if len(labels) == 1 else " / ".join(labels)})
        if len(labels) > 1:
            inconsistent_labels.append({"sample_id": str(sample_id), "labels": labels})

    counts = [item["count"] for item in group_rows]
    expected = int(pd.Series(counts).mode().iloc[0]) if counts else 0
    incomplete_groups = [item for item in group_rows if item["count"] != expected]
    return {
        "group_count": int(len(group_rows)),
        "expected_repeats_per_group": expected,
        "groups": group_rows,
        "inconsistent_labels": inconsistent_labels,
        "incomplete_groups": incomplete_groups,
    }


def summarize_modeling_csv(path: str | Path) -> dict:
    """生成前端所需的类别、Sample_ID、长度与曲线预览摘要。"""
    dataset = load_modeling_csv(path)
    labels = pd.Series(dataset.labels)
    lengths = [len(item) for item in dataset.x_axis]
    sample_summary = _sample_id_summary(dataset.frame)
    return {
        "path": str(Path(path).resolve()),
        "samples": int(len(dataset.labels)),
        "classes": int(labels.nunique()),
        "label_counts": {str(k): int(v) for k, v in labels.value_counts().sort_index().items()},
        "sample_id": sample_summary,
        "curve_length": int(lengths[0]) if lengths else 0,
        "curve_lengths": {str(k): int(v) for k, v in pd.Series(lengths).value_counts().sort_index().items()},
        "columns": list(dataset.frame.columns),
        "preview": dataset.frame.head(8).drop(columns=["XXX", "Intensity"]).to_dict(orient="records"),
        "curves": [
            {
                "index": int(dataset.frame.iloc[i]["Index"]) if str(dataset.frame.iloc[i]["Index"]).isdigit() else str(dataset.frame.iloc[i]["Index"]),
                "name": str(dataset.frame.iloc[i]["Name"]),
                "label": dataset.labels[i],
                "sample_id": dataset.sample_id[i],
                "x": dataset.x_axis[i],
                "y": dataset.intensity[i].astype(float).tolist(),
            }
            for i in range(len(dataset.labels))
        ],
    }


def _read_csv_flexible(path: str | Path) -> pd.DataFrame:
    encodings = ("utf-8-sig", "utf-8", "gbk", "gb18030")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"无法读取 CSV 文件 {Path(path).name}: {last_error}")


def _read_csv_no_header_flexible(path: str | Path) -> pd.DataFrame:
    encodings = ("utf-8-sig", "utf-8", "gbk", "gb18030")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, header=None, sep=None, engine="python", encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"无法读取 CSV 文件 {Path(path).name}: {last_error}")


def read_raw_spectrum(path: str | Path, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """兼容常见编码/表头读取单个拉曼、色谱或 HPLC 二列文件。"""
    path = Path(path)
    frame = _read_csv_no_header_flexible(path)
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.shape[1] < 2 or numeric.iloc[:, :2].dropna().empty:
        frame = _read_csv_flexible(path)
        numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.iloc[:, :2].dropna()
    if numeric.empty:
        raise ValueError(f"{path.name} 没有可解析的两列数值数据")
    # X 轴保留输入精度，避免 450.82 先量化为 float32 后被输出成
    # 450.82000732421875 之类的长文本。强度仍沿用现有 float32 计算契约。
    x = numeric.iloc[:, 0].to_numpy(dtype=np.float64)
    # HPLC 的强度需要在原始时间轴上做 float64 线性映射，不能在插值前先量化。
    y_dtype = np.float64 if kind == "hplc" else np.float32
    y = numeric.iloc[:, 1].to_numpy(dtype=y_dtype)
    if kind not in {"raman", "chromatography", "hplc"}:
        raise ValueError("kind 必须是 raman、chromatography 或 hplc")
    return x, y


def _normalize_numeric_array(
    values: np.ndarray,
    field_name: str,
    decimal_places: int = MAX_OUTPUT_DECIMAL_PLACES,
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{field_name} 必须是一维数组")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} 含有 NaN 或无穷值")
    if not 0 <= decimal_places <= MAX_OUTPUT_DECIMAL_PLACES:
        raise ValueError(
            f"{field_name} 小数位数必须在 0 到 {MAX_OUTPUT_DECIMAL_PLACES} 之间"
        )
    rounded = np.round(array, decimals=decimal_places)
    rounded[rounded == 0.0] = 0.0
    if array.size > 1 and float(np.ptp(array)) > 0.0 and float(np.ptp(rounded)) == 0.0:
        raise ValueError(
            f"{field_name} 保留 {decimal_places} 位小数后失去全部有效变化；"
            "请缩小数值缩放范围，或改用更高精度后重试"
        )
    return rounded.astype(float).tolist()


def _compact_json_numbers(values: list[float]) -> list[int | float]:
    """删除不必要的 ``.0``，同时保留更短的合法指数表示。"""
    compact: list[int | float] = []
    for value in values:
        if value == 0.0:
            compact.append(0)
            continue
        if value.is_integer():
            integer_value = int(value)
            # 对普通整数去掉 .0；极大数使用更短的浮点指数表示。
            if len(str(integer_value)) <= len(repr(value)):
                compact.append(integer_value)
                continue
        compact.append(value)
    return compact


def _serialize_modeling_array_at_precision(
    values: np.ndarray,
    field_name: str,
    decimal_places: int,
) -> SerializedModelingArray:
    normalized = _normalize_numeric_array(values, field_name, decimal_places)
    compact_values = _compact_json_numbers(normalized)
    serialized = json.dumps(
        compact_values,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    # 从最终文本回读，保证 API 预览与实际落盘 JSON 使用完全相同的数值源。
    final_values = json.loads(serialized)
    return SerializedModelingArray(
        values=final_values,
        serialized=serialized,
        decimal_places=decimal_places,
        character_count=len(serialized),
    )


def _serialize_modeling_arrays(
    arrays: list[np.ndarray],
    field_name: str,
    source_names: list[str],
    max_decimal_places: int = MAX_OUTPUT_DECIMAL_PLACES,
    character_limit: int | None = None,
) -> BatchSerializationResult:
    """为一个批次字段选择 Excel 可容纳的最高统一小数精度。"""
    if not arrays:
        raise ValueError(f"{field_name} 没有可序列化的数据")
    if len(arrays) != len(source_names):
        raise ValueError(f"{field_name} 数组数量与样本名数量不一致")
    if not 0 <= max_decimal_places <= MAX_OUTPUT_DECIMAL_PLACES:
        raise ValueError(
            f"{field_name} 最大小数位数必须在 0 到 {MAX_OUTPUT_DECIMAL_PLACES} 之间"
        )
    limit = EXCEL_CELL_CHARACTER_LIMIT if character_limit is None else character_limit
    if limit <= 0:
        raise ValueError("Excel 单元格字符上限必须为正整数")

    shortest_results: list[SerializedModelingArray] | None = None
    for decimal_places in range(max_decimal_places, -1, -1):
        candidate: list[SerializedModelingArray] = []
        for values, source_name in zip(arrays, source_names):
            try:
                item = _serialize_modeling_array_at_precision(
                    values,
                    field_name,
                    decimal_places,
                )
            except ValueError as exc:
                if "失去全部有效变化" in str(exc):
                    if decimal_places == max_decimal_places:
                        raise ValueError(
                            f"{source_name} 的 {field_name} 保留 {decimal_places} 位小数后"
                            "失去全部有效变化；请缩小数值缩放范围，或改用更高精度后重试"
                        ) from exc
                    raise ValueError(
                        f"{source_name} 的 {field_name} 为满足 Excel 单元格上限需要降低精度，"
                        f"但保留 {decimal_places} 位小数会失去全部有效变化；"
                        "请缩小行号范围或 X 轴数值范围，或改用降采样/非 Excel 格式"
                    ) from exc
                raise
            candidate.append(item)
        shortest_results = candidate
        if all(item.character_count <= limit for item in candidate):
            max_index = max(
                range(len(candidate)),
                key=lambda index: candidate[index].character_count,
            )
            return BatchSerializationResult(
                arrays=candidate,
                decimal_places=decimal_places,
                max_characters=candidate[max_index].character_count,
                max_source_name=source_names[max_index],
            )

    assert shortest_results is not None
    max_index = max(
        range(len(shortest_results)),
        key=lambda index: shortest_results[index].character_count,
    )
    longest = shortest_results[max_index]
    source_name = source_names[max_index]
    point_count = len(longest.values)
    raise ValueError(
        f"{source_name} 的 {field_name} 含 {point_count} 个点，即使保留 0 位小数并紧凑序列化后仍有 "
        f"{longest.character_count} 个字符，超过 Excel 单元格上限 {limit}；"
        "请缩小行号范围或 X 轴数值范围，或改用降采样/非 Excel 格式"
    )


def _serialize_modeling_array(
    values: np.ndarray,
    field_name: str,
    source_name: str,
) -> tuple[list[int | float], str]:
    """兼容单数组调用；内部同样使用自适应最高精度算法。"""
    result = _serialize_modeling_arrays([values], field_name, [source_name])
    item = result.arrays[0]
    return item.values, item.serialized


def _output_precision_metadata(
    x_result: BatchSerializationResult,
    intensity_result: BatchSerializationResult,
) -> dict[str, int | bool]:
    return {
        "adaptive": True,
        "max_decimal_places": MAX_OUTPUT_DECIMAL_PLACES,
        "xxx_decimal_places": x_result.decimal_places,
        "intensity_decimal_places": intensity_result.decimal_places,
        "xxx_max_characters": x_result.max_characters,
        "intensity_max_characters": intensity_result.max_characters,
        "excel_cell_character_limit": EXCEL_CELL_CHARACTER_LIMIT,
    }


def _simple_baseline_correct(y: np.ndarray) -> np.ndarray:
    if len(y) < 8:
        return y - float(np.min(y))
    xs = np.arange(len(y), dtype=np.float32)
    anchor_count = max(8, len(y) // 20)
    anchor_idx = np.r_[np.arange(anchor_count), np.arange(len(y) - anchor_count, len(y))]
    degree = 2 if len(anchor_idx) >= 6 else 1
    coeff = np.polyfit(xs[anchor_idx], y[anchor_idx], degree)
    baseline = np.polyval(coeff, xs)
    corrected = y - baseline
    corrected -= np.min(corrected)
    return corrected.astype(np.float32)


def _baseline_correct(x: np.ndarray, y: np.ndarray, method: str) -> np.ndarray:
    try:
        import rampy
    except Exception:
        return _simple_baseline_correct(y)

    try:
        corrected, _baseline = rampy.baseline(x.astype(float), y.astype(float), method=method)
        return np.asarray(corrected, dtype=np.float32).reshape(-1)
    except Exception as exc:
        raise ValueError(f"Baseline method {method} failed: {exc}") from exc


def _range_indexer(
    x: np.ndarray,
    start_row: int,
    end_row: int | None,
    range_mode: str,
    x_min: float | None,
    x_max: float | None,
) -> tuple[slice | np.ndarray, str]:
    if range_mode == "row":
        start = max(0, start_row - 1)
        end = end_row if end_row and end_row > 0 else len(x)
        return slice(start, end), "行范围"
    if range_mode != "x_value":
        raise ValueError("range_mode must be row or x_value")
    if x_min is None and x_max is None:
        raise ValueError("按 X 轴数值取范围时，请至少填写下限或上限")
    lower = float("-inf") if x_min is None else float(x_min)
    upper = float("inf") if x_max is None else float(x_max)
    if lower > upper:
        raise ValueError("X 轴范围下限不能大于上限")
    return (x >= lower) & (x <= upper), "X 轴数值范围"


def preprocess_raw_files(
    files: Iterable[str | Path],
    kind: str,
    start_row: int = 1,
    end_row: int | None = None,
    range_mode: str = "row",
    x_min: float | None = None,
    x_max: float | None = None,
    baseline_method: str = "arPLS",
    display_names: list[str] | None = None,
) -> pd.DataFrame:
    """批量处理原始文件并生成 Excel 可编辑的统一建模表。"""
    prepared: list[dict[str, object]] = []
    for index, file_path in enumerate(files, start=1):
        path = Path(file_path)
        display_name = display_names[index - 1] if display_names and index - 1 < len(display_names) else path.stem
        x, y = read_raw_spectrum(path, kind=kind)
        indexer, range_label = _range_indexer(x, start_row, end_row, range_mode, x_min, x_max)
        x = x[indexer]
        y = y[indexer]
        if len(x) == 0:
            raise ValueError(f"{path.name} 在所选{range_label}内没有数据")
        if kind == "raman":
            y = _baseline_correct(x, y, baseline_method)
        prepared.append(
            {"index": index, "name": display_name, "x": x, "processed_y": y}
        )

    names = [str(item["name"]) for item in prepared]
    x_result = _serialize_modeling_arrays(
        [np.asarray(item["x"]) for item in prepared], "XXX", names
    )
    intensity_result = _serialize_modeling_arrays(
        [np.asarray(item["processed_y"]) for item in prepared], "Intensity", names
    )
    records = []
    for item, x_item, intensity_item in zip(
        prepared, x_result.arrays, intensity_result.arrays
    ):
        records.append(
            {
                "Index": item["index"],
                "Name": item["name"],
                "XXX": x_item.serialized,
                "Intensity": intensity_item.serialized,
                "Label": "",
                "Sample_ID": "",
            }
        )
    return pd.DataFrame.from_records(records, columns=MODELING_COLUMNS)


def preprocess_raw_files_with_preview(
    files: Iterable[str | Path],
    kind: str,
    start_row: int = 1,
    end_row: int | None = None,
    range_mode: str = "row",
    x_min: float | None = None,
    x_max: float | None = None,
    baseline_method: str = "arPLS",
    display_names: list[str] | None = None,
) -> dict:
    """在统一表之外返回前端曲线预览和实际范围元数据。"""
    prepared: list[dict[str, object]] = []
    for index, file_path in enumerate(files, start=1):
        path = Path(file_path)
        display_name = display_names[index - 1] if display_names and index - 1 < len(display_names) else path.stem
        full_x, full_y = read_raw_spectrum(path, kind=kind)
        indexer, range_label = _range_indexer(full_x, start_row, end_row, range_mode, x_min, x_max)
        x = full_x[indexer]
        raw_y = full_y[indexer]
        if kind == "raman":
            corrected_y = _baseline_correct(x, raw_y.copy(), baseline_method)
        else:
            corrected_y = raw_y.copy()
        if len(x) == 0:
            raise ValueError(f"{path.name} 在所选{range_label}内没有数据")
        prepared.append(
            {
                "index": index,
                "name": display_name,
                "x": x,
                "raw_y": raw_y,
                "processed_y": corrected_y,
            }
        )

    names = [str(item["name"]) for item in prepared]
    x_result = _serialize_modeling_arrays(
        [np.asarray(item["x"]) for item in prepared], "XXX", names
    )
    intensity_result = _serialize_modeling_arrays(
        [np.asarray(item["processed_y"]) for item in prepared], "Intensity", names
    )
    records = []
    curves = []
    for item, x_item, intensity_item in zip(
        prepared, x_result.arrays, intensity_result.arrays
    ):
        raw_values = (
            _normalize_numeric_array(np.asarray(item["raw_y"]), "raw_y")
            if kind == "raman"
            else intensity_item.values
        )
        records.append(
            {
                "Index": item["index"],
                "Name": item["name"],
                "XXX": x_item.serialized,
                "Intensity": intensity_item.serialized,
                "Label": "",
                "Sample_ID": "",
            }
        )
        curves.append(
            ({
                "name": item["name"],
                "x": x_item.values,
                "raw_y": raw_values,
                "corrected_y": intensity_item.values,
            }
            if kind == "raman"
            else {
                "name": item["name"],
                "x": x_item.values,
                "raw_y": raw_values,
            })
        )
    frame = pd.DataFrame.from_records(records, columns=MODELING_COLUMNS)
    return {
        "frame": frame,
        "curves": curves,
        "output_precision": _output_precision_metadata(x_result, intensity_result),
    }

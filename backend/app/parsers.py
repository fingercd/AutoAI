from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_MODELING_COLUMNS = {"Index", "Name", "XXX", "Intensity", "Label", "Repeat_index"}
EXCEL_CELL_CHARACTER_LIMIT = 32_767
OUTPUT_SIGNIFICANT_DIGITS = 9


@dataclass
class ModelingDataset:
    frame: pd.DataFrame
    x_axis: list[list[float]]
    intensity: np.ndarray
    labels: list[str]
    repeat_index: list[str]


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


def load_modeling_csv(path: str | Path) -> ModelingDataset:
    path = Path(path)
    frame = _read_csv_flexible(path)
    if "Index" not in frame.columns and REQUIRED_MODELING_COLUMNS.difference({"Index"}).issubset(frame.columns):
        frame = frame.rename(columns={frame.columns[0]: "Index"})
    missing = REQUIRED_MODELING_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"建模数据缺少字段: {', '.join(sorted(missing))}")

    x_axis: list[list[float]] = []
    y_values: list[list[float]] = []
    labels: list[str] = []
    repeat_indices: list[str] = []
    for idx, row in frame.iterrows():
        row_number = idx + 2
        x = _parse_array(row["XXX"], "XXX", row_number)
        y = _parse_array(row["Intensity"], "Intensity", row_number)
        if len(x) != len(y):
            raise ValueError(f"第 {row_number} 行 XXX 和 Intensity 长度不一致")
        label = str(row["Label"]).strip()
        if not label or label.lower() == "nan":
            raise ValueError(f"第 {row_number} 行 Label 为空，建模前请补充标签")
        repeat = str(row["Repeat_index"]).strip()
        if not repeat or repeat.lower() == "nan":
            raise ValueError(f"第 {row_number} 行 Repeat_index 为空，建模前请补充样品分组编号")
        x_axis.append(x)
        y_values.append(y)
        labels.append(label)
        repeat_indices.append(repeat)

    lengths = {len(values) for values in y_values}
    if len(lengths) != 1:
        raise ValueError(f"当前训练版本要求曲线长度一致，检测到长度: {sorted(lengths)}")

    frame = frame.copy()
    frame["Label"] = labels
    frame["Repeat_index"] = repeat_indices
    repeat_summary = _repeat_index_summary(frame)
    if repeat_summary["inconsistent_labels"]:
        details = ", ".join(f"{item['repeat_index']}={item['labels']}" for item in repeat_summary["inconsistent_labels"])
        raise ValueError(f"同一个 Repeat_index 内出现多个 Label，请检查: {details}")
    if repeat_summary["incomplete_groups"]:
        expected = repeat_summary["expected_repeats_per_group"]
        details = ", ".join(f"{item['repeat_index']}={item['count']}" for item in repeat_summary["incomplete_groups"])
        raise ValueError(f"Repeat_index 重复测量次数不一致，期望每组 {expected} 条，异常分组: {details}")

    return ModelingDataset(frame=frame, x_axis=x_axis, intensity=np.asarray(y_values, dtype=np.float32), labels=labels, repeat_index=repeat_indices)


def _repeat_index_summary(frame: pd.DataFrame) -> dict:
    grouped = frame.groupby("Repeat_index", sort=True)
    group_rows = []
    inconsistent_labels = []
    for repeat, group in grouped:
        labels = sorted(str(item) for item in group["Label"].dropna().astype(str).unique())
        count = int(len(group))
        group_rows.append({"repeat_index": str(repeat), "count": count, "label": labels[0] if len(labels) == 1 else " / ".join(labels)})
        if len(labels) > 1:
            inconsistent_labels.append({"repeat_index": str(repeat), "labels": labels})

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
    dataset = load_modeling_csv(path)
    labels = pd.Series(dataset.labels)
    lengths = [len(item) for item in dataset.x_axis]
    repeat_summary = _repeat_index_summary(dataset.frame)
    return {
        "path": str(Path(path).resolve()),
        "samples": int(len(dataset.labels)),
        "classes": int(labels.nunique()),
        "label_counts": {str(k): int(v) for k, v in labels.value_counts().sort_index().items()},
        "repeat_index": repeat_summary,
        "curve_length": int(lengths[0]) if lengths else 0,
        "curve_lengths": {str(k): int(v) for k, v in pd.Series(lengths).value_counts().sort_index().items()},
        "columns": list(dataset.frame.columns),
        "preview": dataset.frame.head(8).drop(columns=["XXX", "Intensity"]).to_dict(orient="records"),
        "curves": [
            {
                "index": int(dataset.frame.iloc[i]["Index"]) if str(dataset.frame.iloc[i]["Index"]).isdigit() else str(dataset.frame.iloc[i]["Index"]),
                "name": str(dataset.frame.iloc[i]["Name"]),
                "label": dataset.labels[i],
                "repeat_index": dataset.repeat_index[i],
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
    y = numeric.iloc[:, 1].to_numpy(dtype=np.float32)
    if kind not in {"raman", "chromatography", "hplc"}:
        raise ValueError("kind 必须是 raman、chromatography 或 hplc")
    return x, y


def _normalize_numeric_array(values: np.ndarray, field_name: str) -> list[float]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{field_name} 必须是一维数组")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} 含有 NaN 或无穷值")
    return [float(format(float(value), f".{OUTPUT_SIGNIFICANT_DIGITS}g")) for value in array]


def _serialize_modeling_array(
    values: np.ndarray,
    field_name: str,
    source_name: str,
) -> tuple[list[float], str]:
    normalized = _normalize_numeric_array(values, field_name)
    serialized = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(serialized) > EXCEL_CELL_CHARACTER_LIMIT:
        raise ValueError(
            f"{source_name} 的 {field_name} 含 {len(normalized)} 个点，紧凑序列化后仍有 "
            f"{len(serialized)} 个字符，超过 Excel 单元格上限 {EXCEL_CELL_CHARACTER_LIMIT}；"
            "请缩小行号范围或 X 轴数值范围后重试"
        )
    return normalized, serialized


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
    records = []
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
        _x_values, x_serialized = _serialize_modeling_array(x, "XXX", display_name)
        _y_values, y_serialized = _serialize_modeling_array(y, "Intensity", display_name)
        records.append(
            {
                "Index": index,
                "Name": display_name,
                "XXX": x_serialized,
                "Intensity": y_serialized,
                "Label": "",
                "Repeat_index": "",
            }
        )
    return pd.DataFrame.from_records(records, columns=["Index", "Name", "XXX", "Intensity", "Label", "Repeat_index"])


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
    records = []
    curves = []
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
        x_values, x_serialized = _serialize_modeling_array(x, "XXX", display_name)
        corrected_values, corrected_serialized = _serialize_modeling_array(
            corrected_y, "Intensity", display_name
        )
        raw_values = _normalize_numeric_array(raw_y, "raw_y")
        records.append(
            {
                "Index": index,
                "Name": display_name,
                "XXX": x_serialized,
                "Intensity": corrected_serialized,
                "Label": "",
                "Repeat_index": "",
            }
        )
        curves.append(
            ({
                "name": display_name,
                "x": x_values,
                "raw_y": raw_values,
                "corrected_y": corrected_values,
            }
            if kind == "raman"
            else {
                "name": display_name,
                "x": x_values,
                "raw_y": raw_values,
            })
        )
    frame = pd.DataFrame.from_records(records, columns=["Index", "Name", "XXX", "Intensity", "Label", "Repeat_index"])
    return {"frame": frame, "curves": curves}

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_MODELING_COLUMNS = {"Index", "Name", "XXX", "Intensity", "Label", "Repeat_index"}


@dataclass
class ModelingDataset:
    frame: pd.DataFrame
    x_axis: list[list[float]]
    intensity: np.ndarray
    labels: list[str]


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
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = REQUIRED_MODELING_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"建模数据缺少字段: {', '.join(sorted(missing))}")

    x_axis: list[list[float]] = []
    y_values: list[list[float]] = []
    labels: list[str] = []
    for idx, row in frame.iterrows():
        row_number = idx + 2
        x = _parse_array(row["XXX"], "XXX", row_number)
        y = _parse_array(row["Intensity"], "Intensity", row_number)
        if len(x) != len(y):
            raise ValueError(f"第 {row_number} 行 XXX 和 Intensity 长度不一致")
        label = str(row["Label"]).strip()
        if not label or label.lower() == "nan":
            raise ValueError(f"第 {row_number} 行 Label 为空，建模前请补充标签")
        x_axis.append(x)
        y_values.append(y)
        labels.append(label)

    lengths = {len(values) for values in y_values}
    if len(lengths) != 1:
        raise ValueError(f"当前训练版本要求曲线长度一致，检测到长度: {sorted(lengths)}")

    return ModelingDataset(frame=frame, x_axis=x_axis, intensity=np.asarray(y_values, dtype=np.float32), labels=labels)


def summarize_modeling_csv(path: str | Path) -> dict:
    dataset = load_modeling_csv(path)
    labels = pd.Series(dataset.labels)
    lengths = [len(item) for item in dataset.x_axis]
    return {
        "path": str(Path(path).resolve()),
        "samples": int(len(dataset.labels)),
        "classes": int(labels.nunique()),
        "label_counts": {str(k): int(v) for k, v in labels.value_counts().sort_index().items()},
        "curve_length": int(lengths[0]) if lengths else 0,
        "curve_lengths": {str(k): int(v) for k, v in pd.Series(lengths).value_counts().sort_index().items()},
        "columns": list(dataset.frame.columns),
        "preview": dataset.frame.head(8).drop(columns=["XXX", "Intensity"]).to_dict(orient="records"),
        "curves": [
            {
                "name": str(dataset.frame.iloc[i]["Name"]),
                "label": dataset.labels[i],
                "x": dataset.x_axis[i],
                "y": dataset.intensity[i].astype(float).tolist(),
            }
            for i in range(min(3, len(dataset.labels)))
        ],
    }


def _read_csv_flexible(path: str | Path) -> pd.DataFrame:
    encodings = ("utf-8-sig", "utf-8", "gbk")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"无法读取 CSV 文件 {Path(path).name}: {last_error}")


def read_raw_spectrum(path: str | Path, kind: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    frame = _read_csv_flexible(path)
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.shape[1] < 2 or numeric.iloc[:, :2].dropna().empty:
        frame = pd.read_csv(path, header=None, sep=None, engine="python", encoding="utf-8-sig")
        numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.iloc[:, :2].dropna()
    if numeric.empty:
        raise ValueError(f"{path.name} 没有可解析的两列数值数据")
    x = numeric.iloc[:, 0].to_numpy(dtype=np.float32)
    y = numeric.iloc[:, 1].to_numpy(dtype=np.float32)
    if kind not in {"raman", "chromatography"}:
        raise ValueError("kind 必须是 raman 或 chromatography")
    return x, y


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

        corrected, _baseline = rampy.baseline(x.astype(float), y.astype(float), method=method)
        return np.asarray(corrected, dtype=np.float32).reshape(-1)
    except Exception:
        return _simple_baseline_correct(y)


def preprocess_raw_files(
    files: Iterable[str | Path],
    kind: str,
    start_row: int = 1,
    end_row: int | None = None,
    baseline_method: str = "arPLS",
) -> pd.DataFrame:
    records = []
    for index, file_path in enumerate(files, start=1):
        path = Path(file_path)
        x, y = read_raw_spectrum(path, kind=kind)
        start = max(0, start_row - 1)
        end = end_row if end_row and end_row > 0 else len(x)
        x = x[start:end]
        y = y[start:end]
        if len(x) == 0:
            raise ValueError(f"{path.name} 在所选行范围内没有数据")
        if kind == "raman":
            y = _baseline_correct(x, y, baseline_method)
        records.append(
            {
                "Index": index,
                "Name": path.stem,
                "XXX": json.dumps(x.astype(float).tolist(), ensure_ascii=False),
                "Intensity": json.dumps(y.astype(float).tolist(), ensure_ascii=False),
                "Label": "",
                "Repeat_index": "",
            }
        )
    return pd.DataFrame.from_records(records, columns=["Index", "Name", "XXX", "Intensity", "Label", "Repeat_index"])

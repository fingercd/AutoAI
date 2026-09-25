"""建模 CSV 与原始拉曼/色谱 CSV 的解析和统一导出。

新建模文件使用 ``wide-feature-v2``：前三列固定为 ``Index, Label,
Sample_ID``，第 4 列 ``Name`` 保存原文件名，后续列名是共享的真实 XXX
坐标，每个单元格保存一个强度标量。旧 ``wide-feature-v1``（无 Name）
继续只读兼容。
宽表无法表达逐行不同的坐标轴，因此预处理导出前必须确认批次内所有曲线共享
同一轴；不一致时明确拒绝，绝不静默套用首条曲线的坐标。
"""

# =============================================================================
# 模块级说明（教学注释）
# -----------------------------------------------------------------------------
# 本模块是整个 AutoAI 数据层的"地基"，承担两类职责：
#   1. 读取方向：load_modeling_csv / summarize_modeling_csv 解析建模宽表
#      （wide-feature-v2 为主、v1 只读兼容），供训练入口与前端摘要使用；
#      read_raw_spectrum 解析仪器导出的二列原始拉曼/色谱/HPLC 文件。
#   2. 写出方向：build_wide_modeling_frame 把预处理后的多条曲线组装成
#      wide-feature-v2 宽表；preprocess_raw_files* 是拉曼/普通色谱的预处理
#      出口（HPLC 走 hplc.py 的专用管线，但同样复用这里的宽表构造）。
# 协作模块：backend.app.hplc（HPLC 固定轴管线）、训练入口（读取 ModelingDataset）、
# 预处理路由层（调用 preprocess_* 与 summarize_*）。
# 关键设计约束：
#   - wide-feature-v2 契约：前四列固定为 Index, Label, Sample_ID, Name，
#     第 5 列起列名是真实、有限、唯一、严格递增的 XXX 坐标文本（float64 可
#     往返），单元格是最多 5 位小数的有限强度标量，adaptive=false，绝不为了
#     适应单元格字符限制而静默降精度或删点。
#   - Excel 总列数上限 16,384：v2 扣除 4 个元数据列后最多 16,380 个特征。
#   - 批次内所有曲线必须共享公共轴；宽表写不出两条不同的轴，所以比对失败时
#     必须拒绝，禁止自动取首条轴迁移。
#   - 旧六列数组格式（XXX/Intensity 单元格存数组）、linspace-v1 等历史格式
#     不再可训练，读取时明确报错而不是尝试兼容。
# =============================================================================

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# 元数据列契约常量：前三列是所有建模表的固定前缀
MODELING_METADATA_COLUMNS = ("Index", "Label", "Sample_ID")
SOURCE_NAME_COLUMN = "Name"
# v2 在 v1 基础上追加 Name 列，保留浏览器端原始文件名
WIDE_MODELING_METADATA_COLUMNS = (*MODELING_METADATA_COLUMNS, SOURCE_NAME_COLUMN)
WIDE_MODELING_FORMAT = "wide-feature-v2"
LEGACY_WIDE_MODELING_FORMAT = "wide-feature-v1"
# 轴编码方式：真实坐标直接写在列名里（而不是另存一行/一个文件）
WIDE_AXIS_ENCODING = "column_headers"
# 旧六列数组格式的标志列，出现即拒绝训练
LEGACY_MODELING_ARRAY_COLUMNS = ("XXX", "Intensity")
EXCEL_WORKSHEET_MAX_COLUMNS = 16_384
# v2 可用特征数上限 = Excel 上限 - 4 个元数据列 = 16,380
MAX_WIDE_FEATURE_COUNT = EXCEL_WORKSHEET_MAX_COLUMNS - len(WIDE_MODELING_METADATA_COLUMNS)
# 强度输出最多 5 位小数（adaptive=false 契约的一部分）
MAX_OUTPUT_DECIMAL_PLACES = 5


@dataclass
class ModelingDataset:
    """解析后的表格元数据、二维强度矩阵和逐行 X 轴。"""
    # 训练入口消费的标准结构：
    #   frame       仅元数据列的 DataFrame（Index/Label/Sample_ID[/Name]）
    #   x_axis      逐行的 X 坐标列表（宽表下每行内容相同，但结构保持逐行，
    #               兼容历史上逐行不同轴的数据形态）
    #   intensity   (n_samples, n_features) 的 float32 强度矩阵
    #   labels/sample_id  逐行标签与样品编号（分类任务按类别名处理，不回归）
    #   data_format "wide-feature-v2" 或 "wide-feature-v1"
    frame: pd.DataFrame
    x_axis: list[list[float]]
    intensity: np.ndarray
    labels: list[str]
    sample_id: list[str]
    data_format: str


@dataclass(frozen=True)
class WideModelingFrameResult:
    """预处理宽表以及与落盘内容完全一致的轴和强度值。"""

    # 构造宽表的一次性产物；x_axis / intensity 直接来自最终表头与单元格，
    # 保证 API 预览、下载文件、训练加载器三方看到的数值完全一致（所见即所存）。
    frame: pd.DataFrame
    x_axis: list[float]
    intensity: list[list[float]]
    output_precision: dict[str, int | str | bool]


def _read_modeling_csv_flexible(path: str | Path, *, content: bytes | None = None) -> tuple[pd.DataFrame, list[str]]:
    """读取宽表并保留原始表头，避免 pandas 静默改写重复列名。"""

    # 参数：建模 CSV 路径。返回：(DataFrame[全字符串], 原始表头列表)。
    # 异常：空文件、无法识别分隔符、表头列数与数据列数不一致、所有编码都
    #       失败时抛 ValueError。
    # 为什么要手工嗅探：pandas 对重复列名会自动追加 .1/.2，而"重复 XXX 坐标"
    # 正是我们要检测并拒绝的错误，必须先拿到用户真实写入的表头。
    path = Path(path)
    encodings = ("utf-8-sig", "utf-8", "gbk", "gb18030")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            import io
            with (io.StringIO(content.decode(encoding)) if content is not None
                  else path.open("r", encoding=encoding, newline="")) as handle:
                header_line = handle.readline()
            if not header_line:
                raise ValueError(f"建模 CSV {path.name} 为空")
            try:
                dialect = csv.Sniffer().sniff(header_line, delimiters=",;\t|")
            except csv.Error as exc:
                raise ValueError(f"建模 CSV {path.name} 无法识别列分隔符") from exc
            raw_header = next(csv.reader([header_line], dialect=dialect))
            normalized_header = [str(item).strip() for item in raw_header]
            # dtype=str + keep_default_na=False：全部按原文本读入，
            # 数值解析与空值判定留到后面的显式校验，避免 pandas 抢先转换
            frame = pd.read_csv(
                io.BytesIO(content) if content is not None else path,
                sep=dialect.delimiter,
                engine="python",
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
            )
            if len(frame.columns) != len(normalized_header):
                raise ValueError(
                    f"建模 CSV 表头解析得到 {len(normalized_header)} 列，但数据表解析得到 "
                    f"{len(frame.columns)} 列"
                )
            # 在检查重复表头后才会按这些原始名称访问数据。这里先覆盖 pandas 为重复
            # 列自动附加的 .1/.2，确保后续检查面对用户实际写入的表头。
            frame.columns = normalized_header
            return frame, normalized_header
        except UnicodeDecodeError as exc:
            # 解码失败才值得尝试下一种编码
            last_error = exc
        except (OSError, csv.Error, pd.errors.ParserError, ValueError) as exc:
            last_error = exc
            # 编码已经成功解码时，结构错误不应继续用其他编码掩盖原始原因。
            if not isinstance(exc, UnicodeDecodeError):
                break
    if isinstance(last_error, ValueError):
        raise last_error
    raise ValueError(f"无法读取 CSV 文件 {path.name}: {last_error}")


def _parse_wide_axis_headers(
    feature_headers: list[str],
    *,
    metadata_column_count: int,
) -> list[float]:
    # 把宽表第 N 列起的特征列名解析为共享 XXX 坐标轴。
    # 参数：feature_headers —— 元数据列之后的全部列名文本；
    #       metadata_column_count —— 3（v1）或 4（v2），用于 Excel 上限核算。
    # 返回：float 坐标列表（保证非空、有限、严格递增）。
    # 异常：无特征列、超出 Excel 上限、列名为空/非数值/非有限、坐标重复或
    #       未严格递增时抛 ValueError。
    if not feature_headers:
        raise ValueError("建模 CSV 至少需要 1 个真实 XXX 特征列")
    max_feature_count = EXCEL_WORKSHEET_MAX_COLUMNS - metadata_column_count
    if len(feature_headers) > max_feature_count:
        raise ValueError(
            f"建模 CSV 含 {len(feature_headers)} 个特征，连同 {metadata_column_count} "
            f"个元数据列共 {len(feature_headers) + metadata_column_count} 列，超过 Excel 上限 "
            f"{EXCEL_WORKSHEET_MAX_COLUMNS}；请先按业务要求缩小数据范围"
        )

    axis: list[float] = []
    for position, header in enumerate(feature_headers, start=1):
        if not header:
            raise ValueError(f"第 {position} 个 XXX 特征列名为空")
        try:
            value = float(header)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"第 {position} 个特征列名 {header!r} 不是有效的真实 XXX 数值"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(f"第 {position} 个 XXX 特征列名必须是有限数值，当前为 {header!r}")
        axis.append(value)

    # 严格递增检查：重复坐标与非单调分别给出不同的错误文案，便于用户定位
    differences = np.diff(np.asarray(axis, dtype=np.float64))
    invalid = np.flatnonzero(differences <= 0)
    if invalid.size:
        index = int(invalid[0])
        if axis[index] == axis[index + 1]:
            raise ValueError(
                f"XXX 特征坐标数值重复：第 {index + 1}、{index + 2} 个特征列 "
                f"{feature_headers[index]!r} 与 {feature_headers[index + 1]!r} 表示同一坐标"
            )
        raise ValueError(
            f"XXX 特征坐标必须按列严格递增；第 {index + 1}、{index + 2} 个坐标为 "
            f"{axis[index]:.17g}、{axis[index + 1]:.17g}"
        )
    return axis


def load_modeling_csv(path: str | Path, *, content: bytes | None = None) -> ModelingDataset:
    """解析 v2 带 Name 宽表或兼容的 v1 宽表，返回训练数据结构。"""
    # 建模数据读取主入口。校验链：表头唯一 → 拒绝旧六列数组格式 → 固定前缀
    # → 判定 v1/v2 → 解析共享轴 → 元数据非空/Index 唯一/Name 非空 → Label 与
    # Sample_ID 逐行非空 → 强度全部有限 → Sample_ID 分组一致性。
    # 异常：任一环节不满足契约即抛 ValueError，训练入口不做二次猜测。
    path = Path(path)
    raw_frame, raw_header = _read_modeling_csv_flexible(path, content=content)
    stripped_header = [item.strip() for item in raw_header]
    if len(set(stripped_header)) != len(stripped_header):
        duplicates = sorted(
            {item for item in stripped_header if stripped_header.count(item) > 1}
        )
        raise ValueError(f"建模 CSV 表头包含重复列名: {', '.join(repr(item) for item in duplicates)}")
    if set(LEGACY_MODELING_ARRAY_COLUMNS).issubset(stripped_header):
        raise ValueError(
            "检测到旧六列数组格式（XXX/Intensity 单元格存数组）；新训练只接受 "
            "wide-feature-v2（含 Name）或兼容的 wide-feature-v1：前三列为 "
            "Index, Label, Sample_ID，元数据列之后为真实 XXX 坐标"
        )
    actual_prefix = tuple(stripped_header[: len(MODELING_METADATA_COLUMNS)])
    if actual_prefix != MODELING_METADATA_COLUMNS:
        raise ValueError(
            "建模 CSV 前三列及顺序必须是 Index, Label, Sample_ID；"
            f"当前为 {', '.join(actual_prefix) if actual_prefix else '空表头'}"
        )

    # 第 4 列恰为 "Name" 则按 v2 解析，否则按 v1 只读兼容
    has_source_name = (
        len(stripped_header) > len(MODELING_METADATA_COLUMNS)
        and stripped_header[len(MODELING_METADATA_COLUMNS)] == SOURCE_NAME_COLUMN
    )
    metadata_columns = (
        WIDE_MODELING_METADATA_COLUMNS if has_source_name else MODELING_METADATA_COLUMNS
    )
    data_format = WIDE_MODELING_FORMAT if has_source_name else LEGACY_WIDE_MODELING_FORMAT
    feature_headers = stripped_header[len(metadata_columns) :]
    shared_axis = _parse_wide_axis_headers(
        feature_headers,
        metadata_column_count=len(metadata_columns),
    )
    if raw_frame.empty:
        raise ValueError("建模 CSV 没有数据行")

    # 元数据统一转成去空白字符串；缺失值先填空串，再由下面的显式检查拒绝
    metadata = raw_frame.loc[:, list(metadata_columns)].copy()
    for column in metadata_columns:
        metadata[column] = (
            metadata[column].where(metadata[column].notna(), "").astype(str).str.strip()
        )
    empty_index = metadata["Index"].eq("")
    if bool(empty_index.any()):
        # +2 换算到用户视角的行号：1 行表头 + 0 基转 1 基
        row_number = int(np.flatnonzero(empty_index.to_numpy())[0]) + 2
        raise ValueError(f"第 {row_number} 行 Index 为空")
    duplicate_index = metadata["Index"].duplicated(keep=False)
    if bool(duplicate_index.any()):
        value = str(metadata.loc[duplicate_index, "Index"].iloc[0])
        raise ValueError(f"Index 必须唯一，检测到重复值 {value!r}")
    if has_source_name:
        # v2 契约：Name 必须保留原始文件名，空串和字面 "nan" 都算缺失
        empty_name = metadata[SOURCE_NAME_COLUMN].eq("") | metadata[SOURCE_NAME_COLUMN].str.lower().eq("nan")
        if bool(empty_name.any()):
            row_number = int(np.flatnonzero(empty_name.to_numpy())[0]) + 2
            raise ValueError(f"第 {row_number} 行 Name 为空，必须保留原始文件名")

    labels = metadata["Label"].tolist()
    sample_ids = metadata["Sample_ID"].tolist()
    for idx, (label, sample_id) in enumerate(zip(labels, sample_ids), start=2):
        if not label or label.lower() == "nan":
            raise ValueError(f"第 {idx} 行 Label 为空，建模前请补充标签")
        if not sample_id or sample_id.lower() == "nan":
            raise ValueError(f"第 {idx} 行 Sample_ID 为空，建模前请补充样品编号")

    # 强度矩阵：逐单元格转数值，任何无法转换或非有限的值都定位到行/列报错
    feature_text = raw_frame.loc[:, feature_headers]
    numeric_features = feature_text.apply(pd.to_numeric, errors="coerce")
    numeric_array = numeric_features.to_numpy(dtype=np.float64)
    invalid = ~np.isfinite(numeric_array)
    if bool(invalid.any()):
        row_index, column_index = np.argwhere(invalid)[0]
        raw_value = feature_text.iloc[int(row_index), int(column_index)]
        raise ValueError(
            f"第 {int(row_index) + 2} 行、XXX={feature_headers[int(column_index)]} 的 "
            f"Intensity 必须是有限数值，当前为 {raw_value!r}"
        )

    # Sample_ID 分组一致性：同一样品多标签、或各组重复测量次数不一致，
    # 都会破坏按 Sample_ID 整组划分 train/valid/test 的评估口径，必须前置拒绝
    sample_summary = _sample_id_summary(metadata)
    if sample_summary["inconsistent_labels"]:
        details = ", ".join(f"{item['sample_id']}={item['labels']}" for item in sample_summary["inconsistent_labels"])
        raise ValueError(f"同一个 Sample_ID 内出现多个 Label，请检查: {details}")
    if sample_summary["incomplete_groups"]:
        expected = sample_summary["expected_repeats_per_group"]
        details = ", ".join(f"{item['sample_id']}={item['count']}" for item in sample_summary["incomplete_groups"])
        raise ValueError(f"Sample_ID 重复测量次数不一致，期望每组 {expected} 条，异常分组: {details}")

    return ModelingDataset(
        frame=metadata,
        x_axis=[list(shared_axis) for _ in range(len(metadata))],
        intensity=np.asarray(numeric_array, dtype=np.float32),
        labels=labels,
        sample_id=sample_ids,
        data_format=data_format,
    )


def natural_sort_key(value: object) -> tuple[tuple[int, object], ...]:
    """生成稳定自然排序键，使 2 排在 10 前，并兼容 S2/S10。"""
    # 把文本切成"数字段/非数字段"交替序列：数字段按整数值比（标记 0 优先），
    # 非数字段按 casefold 文本比（标记 1 靠后），元组逐段比较即得自然序。
    parts = re.split(r"(\d+)", str(value).strip())
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in parts
        if part
    )


def _sample_id_summary(frame: pd.DataFrame) -> dict:
    # 汇总 Sample_ID 分组信息，同时服务于"数据摘要展示"和"训练前一致性校验"。
    # 返回：group_count（样品数）、expected_repeats_per_group（期望重复次数，
    # 取各组行数的众数）、groups（逐组明细）、inconsistent_labels（同组多标签）、
    # incomplete_groups（行数偏离众数的组）。
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
    # 建模数据"摘要接口"：在 load_modeling_csv 的全量校验之上，组装前端
    # 数据卡片与曲线预览所需的全部字段；curves 逐行携带 x/y，供前端直接绘图。
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
        "data_format": dataset.data_format,
        "columns": list(dataset.frame.columns),
        "preview": dataset.frame.head(8).to_dict(orient="records"),
        "curves": [
            {
                "index": int(dataset.frame.iloc[i]["Index"]) if str(dataset.frame.iloc[i]["Index"]).isdigit() else str(dataset.frame.iloc[i]["Index"]),
                "name": str(
                    dataset.frame.iloc[i].get(
                        SOURCE_NAME_COLUMN,
                        dataset.frame.iloc[i]["Index"],
                    )
                ),
                "label": dataset.labels[i],
                "sample_id": dataset.sample_id[i],
                "x": dataset.x_axis[i],
                "y": dataset.intensity[i].astype(float).tolist(),
            }
            for i in range(len(dataset.labels))
        ],
    }


def _read_csv_flexible(path: str | Path) -> pd.DataFrame:
    # 原始文件读取（带表头假设）：依次尝试 utf-8-sig / utf-8 / gbk / gb18030，
    # sep=None 让 python 引擎自动嗅探分隔符；全部按文本读入。
    encodings = ("utf-8-sig", "utf-8", "gbk", "gb18030")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
            )
        except Exception as exc:
            last_error = exc
    raise ValueError(f"无法读取 CSV 文件 {Path(path).name}: {last_error}")


def _read_csv_no_header_flexible(path: str | Path) -> pd.DataFrame:
    # 原始文件读取（无表头假设）：与上面成对存在，先按"无表头纯数据"尝试，
    # 失败再由调用方退回带表头解析，兼容两类仪器导出格式。
    encodings = ("utf-8-sig", "utf-8", "gbk", "gb18030")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                header=None,
                sep=None,
                engine="python",
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
            )
        except Exception as exc:
            last_error = exc
    raise ValueError(f"无法读取 CSV 文件 {Path(path).name}: {last_error}")


def read_raw_spectrum(path: str | Path, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """兼容常见编码/表头读取单个拉曼、色谱或 HPLC 二列文件。"""
    # 参数：kind ∈ {"raman", "chromatography", "hplc"}，决定强度的 dtype 策略。
    # 返回：(x, y)；x 恒为 float64，y 在 HPLC 下为 float64、其余为 float32。
    # 异常：解析不出两列数值、或 kind 非法时抛 ValueError。
    path = Path(path)

    def numeric_values(frame: pd.DataFrame) -> pd.DataFrame:
        def parse_cell(value: object) -> float:
            try:
                return float(str(value).strip())
            except (TypeError, ValueError, OverflowError):
                return float("nan")

        # 不使用 pandas.to_numeric：其快速转换器会把部分 17 位十进制先舍入
        # 一个 ULP，破坏真实 XXX 表头的 float64 往返契约。
        return frame.map(parse_cell)

    # 先按"无表头"解析；若前两列没有任何数值，再按"带表头"重试。
    # 表头行在第一次尝试中会变成 NaN 行，dropna 后即被剔除。
    frame = _read_csv_no_header_flexible(path)
    numeric = numeric_values(frame)
    if numeric.shape[1] < 2 or numeric.iloc[:, :2].dropna().empty:
        frame = _read_csv_flexible(path)
        numeric = numeric_values(frame)
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
    # 把一维数值数组规范化为"最多 decimal_places 位小数"的 float 列表，
    # 是强度落盘前的统一精度闸门（adaptive=false：固定位数，不自动降级）。
    # 异常：非一维、含 NaN/inf、位数越界、或舍入后丢失全部有效变化时抛
    #       ValueError——最后一条防止把小信号 silently 舍成全零。
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
    # 消除 -0.0，避免 CSV 里出现 "-0" 这类易引发误会的文本
    rounded[rounded == 0.0] = 0.0
    if array.size > 1 and float(np.ptp(array)) > 0.0 and float(np.ptp(rounded)) == 0.0:
        raise ValueError(
            f"{field_name} 保留 {decimal_places} 位小数后失去全部有效变化；"
            "请缩小数值缩放范围，或改用更高精度后重试"
        )
    return rounded.astype(float).tolist()


def _format_wide_axis_header(value: float) -> str:
    # 单个 XXX 坐标 → 列名文本。".17g" 是 float64 可往返的最短精度保证：
    # 读回 float(header) 必然得到与原值完全相同的 double。
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("XXX 坐标必须是有限数值")
    if number == 0.0:
        return "0"
    return format(number, ".17g")


def _wide_axis_headers(values: np.ndarray, source_name: str) -> tuple[list[float], list[str]]:
    # 校验一条 X 轴并生成宽表特征列名。
    # 返回：(从最终表头回读的 float64 坐标列表, 列名文本列表)。
    # 异常：非一维/空、非有限、非严格递增、超出 Excel 特征上限、格式化后
    #       列名重复（两个不同 double 落到同一文本）时抛 ValueError。
    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError(f"{source_name} 的 XXX 必须是一维非空数组")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{source_name} 的 XXX 含有 NaN 或无穷值")
    invalid = np.flatnonzero(np.diff(axis) <= 0)
    if invalid.size:
        index = int(invalid[0])
        raise ValueError(
            f"{source_name} 的 XXX 必须严格递增；第 {index + 1}、{index + 2} 个坐标为 "
            f"{axis[index]:.17g}、{axis[index + 1]:.17g}"
        )
    if axis.size > MAX_WIDE_FEATURE_COUNT:
        raise ValueError(
            f"{source_name} 含 {axis.size} 个特征，宽表连同 4 个元数据列共 "
            f"{axis.size + len(WIDE_MODELING_METADATA_COLUMNS)} 列，超过 Excel 上限 "
            f"{EXCEL_WORKSHEET_MAX_COLUMNS}；请先缩小行号或 X 轴范围"
        )
    headers = [_format_wide_axis_header(item) for item in axis]
    if len(set(headers)) != len(headers):
        raise ValueError(f"{source_name} 的真实 XXX 坐标格式化后出现重复列名")
    # 从最终表头回读，确保 API 预览与训练加载器看到完全相同的 float64 坐标。
    return [float(item) for item in headers], headers


def build_wide_modeling_frame(
    *,
    indices: list[int | str],
    x_arrays: list[np.ndarray],
    intensity_arrays: list[np.ndarray],
    source_names: list[str],
) -> WideModelingFrameResult:
    """构造以真实 XXX 为表头的统一宽表，并强制一个批次共享公共轴。"""

    # wide-feature-v2 宽表唯一构造入口：拉曼、普通色谱、HPLC 三条预处理管线
    # 最终都汇到这里，保证输出契约只有一份实现。
    # 参数：indices（Index 列）、x_arrays（逐曲线 X 轴）、intensity_arrays
    # （逐曲线强度）、source_names（Name 列，原始文件名），四者必须等长。
    # 返回：WideModelingFrameResult，Label/Sample_ID 留空由用户后续填写。
    # 异常：任一曲线轴与首条曲线表头逐点不一致时抛 ValueError——宽表物理上
    #       只能保存一条公共轴，这里绝不静默取首条轴套用（静默迁移会导致
    #       其他曲线的强度被标到错误坐标上）。
    count = len(indices)
    if count == 0:
        raise ValueError("没有可生成宽表的曲线")
    if not (len(x_arrays) == len(intensity_arrays) == len(source_names) == count):
        raise ValueError("宽表的索引、坐标轴、强度和文件名数量不一致")
    if any(not str(name).strip() for name in source_names):
        raise ValueError("宽表 Name 必须保留每条曲线的原始文件名")

    # 首条曲线的轴即公共轴基准；其余曲线逐一与它比表头文本
    reference_axis, feature_headers = _wide_axis_headers(x_arrays[0], source_names[0])
    reference_header_tuple = tuple(feature_headers)
    normalized_intensities: list[list[float]] = []
    for axis_values, intensity_values, source_name in zip(
        x_arrays, intensity_arrays, source_names
    ):
        candidate_axis, candidate_headers = _wide_axis_headers(axis_values, source_name)
        if tuple(candidate_headers) != reference_header_tuple:
            # 定位第一个不同点：能给出坐标值差异就给坐标，否则说明是点数不同
            first_difference = next(
                (
                    index
                    for index, (left, right) in enumerate(
                        zip(reference_header_tuple, candidate_headers)
                    )
                    if left != right
                ),
                min(len(reference_header_tuple), len(candidate_headers)),
            )
            if first_difference < min(len(reference_axis), len(candidate_axis)):
                detail = (
                    f"第 {first_difference + 1} 个坐标分别为 "
                    f"{reference_axis[first_difference]:.17g} 与 "
                    f"{candidate_axis[first_difference]:.17g}"
                )
            else:
                detail = (
                    f"点数分别为 {len(reference_axis)} 与 {len(candidate_axis)}"
                )
            raise ValueError(
                f"{source_name} 的 XXX 与 {source_names[0]} 不一致（{detail}）；"
                "宽表只能保存一条公共真实轴，请先统一采样轴或开启 HPLC 插值"
            )

        intensity = np.asarray(intensity_values, dtype=np.float64)
        if intensity.ndim != 1:
            raise ValueError(f"{source_name} 的 Intensity 必须是一维数组")
        if intensity.size != len(reference_axis):
            raise ValueError(
                f"{source_name} 的 Intensity 有 {intensity.size} 个点，但公共 XXX 有 "
                f"{len(reference_axis)} 个点"
            )
        # 强度统一走固定 5 位小数闸门（adaptive=false 契约）
        normalized_intensities.append(
            _normalize_numeric_array(
                intensity,
                f"{source_name} 的 Intensity",
                MAX_OUTPUT_DECIMAL_PLACES,
            )
        )

    metadata = pd.DataFrame(
        {
            "Index": indices,
            "Label": [""] * count,
            "Sample_ID": [""] * count,
            SOURCE_NAME_COLUMN: source_names,
        },
        columns=WIDE_MODELING_METADATA_COLUMNS,
    )
    feature_frame = pd.DataFrame(
        np.asarray(normalized_intensities, dtype=np.float64),
        columns=feature_headers,
    )
    frame = pd.concat([metadata, feature_frame], axis=1)
    total_columns = len(frame.columns)
    return WideModelingFrameResult(
        frame=frame,
        x_axis=reference_axis,
        intensity=normalized_intensities,
        # output_precision 是随结果返回给前端/元数据的精度契约声明：
        # 格式、轴编码、float64 往返、强度位数、不自适应降级、Excel 兼容性
        output_precision={
            "format": WIDE_MODELING_FORMAT,
            "xxx_encoding": WIDE_AXIS_ENCODING,
            "xxx_precision": "float64-roundtrip",
            "intensity_decimal_places": MAX_OUTPUT_DECIMAL_PLACES,
            "adaptive": False,
            "feature_count": len(feature_headers),
            "total_column_count": total_columns,
            "excel_column_limit": EXCEL_WORKSHEET_MAX_COLUMNS,
            "excel_compatible": total_columns <= EXCEL_WORKSHEET_MAX_COLUMNS,
        },
    )


def modeling_metadata_preview(frame: pd.DataFrame, limit: int = 5) -> list[dict[str, object]]:
    """返回紧凑元数据预览，避免把数千个宽表特征塞进 API 响应。"""

    # 只取元数据列的前 limit 行；宽表可能有上万个特征列，绝不能整体进响应
    metadata_columns = list(MODELING_METADATA_COLUMNS)
    if SOURCE_NAME_COLUMN in frame.columns:
        metadata_columns.append(SOURCE_NAME_COLUMN)
    missing = [column for column in metadata_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"宽表缺少元数据列: {', '.join(missing)}")
    return frame.loc[:, metadata_columns].head(limit).to_dict(orient="records")


def _simple_baseline_correct(y: np.ndarray) -> np.ndarray:
    # rampy 不可用时的兜底基线校正：取首尾各 5%（至少 8 个）锚点拟合
    # 低阶多项式作为基线，扣除后整体平移到最小值为 0。
    # 短序列（<8 点）退化为最小值平移，避免过拟合。
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
    # 拉曼基线校正：优先用 rampy 的指定算法（默认 arPLS）；
    # rampy 未安装时退回 _simple_baseline_correct；已安装但算法执行失败则
    # 抛错（不静默降级，避免用户以为自己用了 arPLS 实际却不是）。
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
    # 拉曼/普通色谱的范围选择器（HPLC 走 hplc.py 的严格版本，不走这里）。
    # 返回：(可直接用于 numpy 索引的 slice 或布尔掩码, 范围标签)。
    # 与 HPLC 版本的差别：这里对越界行号做钳制（max(0, start-1)、超界取全长），
    # 属于旧接口的宽松语义，保留以兼容既有行为。
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
    # 拉曼/普通色谱预处理的简洁出口：只返回宽表 DataFrame。
    # 处理顺序固定：先选择数据范围，再执行基线校正（仅拉曼）——顺序不能反，
    # 否则基线会拟合到用户并不关心的区段。
    prepared: list[dict[str, object]] = []
    for index, file_path in enumerate(files, start=1):
        path = Path(file_path)
        display_name = display_names[index - 1] if display_names and index - 1 < len(display_names) else path.name
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
    # 公共轴校验在 build_wide_modeling_frame 内完成；不一致则整批拒绝
    wide = build_wide_modeling_frame(
        indices=[item["index"] for item in prepared],
        x_arrays=[np.asarray(item["x"]) for item in prepared],
        intensity_arrays=[np.asarray(item["processed_y"]) for item in prepared],
        source_names=names,
    )
    return wide.frame


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
    # 与 preprocess_raw_files 同流程，但额外返回逐曲线预览数据：
    # 拉曼给 raw_y（原始）与 corrected_y（基线校正后）两条；其他只给 raw_y。
    # 预览数值与宽表落盘值取自同一份规范化结果，保证"所见即所存"。
    prepared: list[dict[str, object]] = []
    for index, file_path in enumerate(files, start=1):
        path = Path(file_path)
        display_name = display_names[index - 1] if display_names and index - 1 < len(display_names) else path.name
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
    wide = build_wide_modeling_frame(
        indices=[item["index"] for item in prepared],
        x_arrays=[np.asarray(item["x"]) for item in prepared],
        intensity_arrays=[np.asarray(item["processed_y"]) for item in prepared],
        source_names=names,
    )
    curves = []
    for position, item in enumerate(prepared):
        # wide.intensity 是已经过 5 位小数闸门的最终值；raw_y 仅拉曼单独
        # 规范化后给出，用于前端对比"校正前 vs 校正后"
        intensity_values = wide.intensity[position]
        raw_values = (
            _normalize_numeric_array(np.asarray(item["raw_y"]), "raw_y")
            if kind == "raman"
            else intensity_values
        )
        curves.append(
            ({
                "name": item["name"],
                "x": wide.x_axis,
                "raw_y": raw_values,
                "corrected_y": intensity_values,
            }
            if kind == "raman"
            else {
                "name": item["name"],
                "x": wide.x_axis,
                "raw_y": raw_values,
            })
        )
    return {
        "frame": wide.frame,
        "curves": curves,
        "output_precision": wide.output_precision,
    }

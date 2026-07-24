"""HPLC 色谱固定时间轴线性映射管线。"""

# =============================================================================
# 模块级说明（教学注释）
# -----------------------------------------------------------------------------
# 本模块是 AutoAI 预处理子系统中专门负责 HPLC（高效液相色谱）数据的管线。
# 在整体架构中的位置：
#   前端选择文件并提交范围参数 → backend 路由层（preprocess 路由）
#     → 本模块（HPLC 批次检查 + 固定轴映射）→ parsers.build_wide_modeling_frame
#     → 输出 wide-feature-v2 宽表（供建模训练入口读取）。
# 协作模块：
#   - backend.app.parsers：提供 read_raw_spectrum（二列原始文件解析）、
#     build_wide_modeling_frame（共享真实轴宽表构造）与 WIDE_AXIS_ENCODING 契约常量。
#   - 路由层调用 inspect_hplc_files（选择文件后立即展示逐文件状态）与
#     preprocess_hplc_files_with_preview（正式预处理 + 曲线预览）。
# 关键设计约束（与 AGENTS.md 预处理规则一一对应）：
#   1. HPLC 固定轴业务值集中在 HplcGridConfig：时间范围 0–50 分钟，点数由当前
#      批次检测出的公共点数动态构造，算法函数与前端都不得写死数值。
#   2. 行号是 1 基、首尾包含，起止都必须位于 1..point_count；终止行留空才按
#      完整点数处理，禁止静默截断越界值。
#   3. 开启插值（hplc_interpolate=true）时，范围选择作用于固定目标轴：
#      第 n 点真实时间为 start + (n-1) * (stop-start)/(point_count-1)；
#      强度用完整源曲线左右邻点做 float64 线性映射；边界相位差不超过一个
#      采样间隔时允许首尾两点线性延伸（禁止更远处外推）。
#   4. 关闭插值时导出所选原始 X/Y，但多文件所选轴不一致必须拒绝（由
#      build_wide_modeling_frame 的公共轴校验最终兜底）。
#   5. 两种模式都不执行消负或面积归一化。
#   6. 批次点数不一致时，以唯一众数为期望点数并列出全部异常文件；众数并列时
#      列出全部分组并拒绝预处理，绝不自动猜测。
# =============================================================================

from __future__ import annotations

from collections import Counter
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

    # frozen=True 使配置不可变：构造后任何字段都不能再被改写，
    # 保证同一次预处理内所有函数看到的网格参数完全一致。
    # 字段含义：
    #   start_minutes     固定时间轴起点（业务固定 0 分钟）
    #   stop_minutes      固定时间轴终点（业务固定 50 分钟）
    #   point_count       固定轴点数，由批次公共点数动态决定，不写死
    #   tolerance_minutes 端点对齐容差，用于吸收仪器写出端点的微小浮点相位差
    #   unit              单位标签，仅用于错误消息与元数据展示

    start_minutes: float
    stop_minutes: float
    point_count: int
    tolerance_minutes: float
    unit: str = "minute"

    @property
    def step_minutes(self) -> float:
        # 相邻两点的时间间隔；首尾端点都包含，所以间隔数是 point_count - 1
        return (self.stop_minutes - self.start_minutes) / (self.point_count - 1)


@dataclass(frozen=True)
class HplcAxisSelection:
    """固定 HPLC 网格上的连续范围选择结果。"""

    # 一次"范围选择"的全部产物，供后续映射与元数据导出复用：
    #   full_axis          完整固定时间轴（未截取）
    #   indices            选中点在 full_axis 上的整数下标（0 基）
    #   target_x           选中点对应的真实时间坐标（full_axis 的切片）
    #   offset             首个选中点的 0 基下标
    #   length             选中点数量
    #   selected_start_row 选中范围的 1 基起始行号（面向用户/前端展示）
    #   selected_end_row   选中范围的 1 基结束行号（首尾包含）
    #   range_label        "行范围" 或 "保留时间范围"，用于错误消息

    full_axis: np.ndarray
    indices: np.ndarray
    target_x: np.ndarray
    offset: int
    length: int
    selected_start_row: int
    selected_end_row: int
    range_label: str


# 固定网格的业务常量。除 point_count 外全部静态固定；
# 点数必须随批次变化，所以通过 hplc_grid_config() 动态构造，而不是放全局常量。
HPLC_GRID_START_MINUTES = 0.0
HPLC_GRID_STOP_MINUTES = 50.0
HPLC_GRID_TOLERANCE_MINUTES = 1e-8
HPLC_GRID_UNIT = "minute"


def hplc_grid_config(point_count: int) -> HplcGridConfig:
    """按当前批次实际点数构造 HPLC 目标网格配置。"""
    # 参数：point_count —— 批次检查检测出的公共有效色谱点数（整数，>= 2）。
    # 返回：填充好业务固定值（0–50 分钟）的 HplcGridConfig。
    # 异常：point_count 不是整数或小于 2 时抛 ValueError。
    # bool 是 int 的子类，必须显式排除，避免 True 被当成 1 通过校验。
    if isinstance(point_count, bool) or not isinstance(point_count, (int, np.integer)):
        raise ValueError(f"HPLC 批次点数必须是整数；当前为 {point_count!r}")
    if int(point_count) < 2:
        raise ValueError(f"HPLC 批次至少需要 2 个有效色谱点；当前为 {point_count}")
    return HplcGridConfig(
        start_minutes=HPLC_GRID_START_MINUTES,
        stop_minutes=HPLC_GRID_STOP_MINUTES,
        point_count=int(point_count),
        tolerance_minutes=HPLC_GRID_TOLERANCE_MINUTES,
        unit=HPLC_GRID_UNIT,
    )


def build_hplc_target_axis(config: HplcGridConfig) -> np.ndarray:
    """根据注入配置构造包含首尾端点的固定 float64 时间轴。"""
    # 参数：config —— 由 hplc_grid_config 构造的网格配置。
    # 返回：长度 point_count 的 np.linspace 等间距 float64 数组，首点恰为
    #       start_minutes，末点恰为 stop_minutes。
    # 异常：点数不足 2、范围非有限值、起点不小于终点、容差非法时抛 ValueError。
    # 这里对 config 再做一次防御性校验：即使调用方绕过 hplc_grid_config
    # 手工构造配置，也不会生成非法轴。
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
    # 参数：start_row/end_row —— 用户输入的行号（1 基、首尾包含）；
    #       end_row 为 None 表示"留空"，按完整点数 point_count 处理。
    # 返回：(start, end) 均为合法 1 基行号且 start <= end。
    # 异常：非整数、越界（不在 1..point_count）、起点大于终点时抛 ValueError。
    # 设计意图：绝不静默截断越界值——用户填了 9999 而实际只有 4000 点时必须
    # 报错而不是悄悄截到 4000，避免用户误以为处理了不存在的区间。

    def validate(value: object, label: str) -> int:
        # 同样显式排除 bool，避免 True/False 混进行号
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
    # 终止行留空（None）是唯一允许的"省略"形式，此时取整条固定轴
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
    # 参数：x_min/x_max —— 用户输入的保留时间下限/上限（分钟），均可为 None。
    # 返回：(lower, upper)；None 侧规范化为 -inf / +inf，表示"不限制该侧"。
    # 异常：值无法转成有限 float、或下限大于上限时抛 ValueError。
    # 使用 ±inf 而不是 0/50 作为缺省边界，是因为该函数也被"原始轴"路径复用，
    # 原始轴的实际起止未必正好是 0–50，用 inf 不会对原始数据造成误裁。

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
    config: HplcGridConfig,
) -> HplcAxisSelection:
    """在完整固定分钟轴上严格选择连续点，并保留原始整数位置。"""
    # 开启插值时的"范围选择"核心：所有选择都发生在固定目标轴上（而不是源数据
    # 的原始轴），保证选出的 target_x 对所有文件完全一致。
    # 参数：range_mode="row" 按 1 基行号选；"x_value" 按保留时间闭区间选。
    # 返回：HplcAxisSelection，含完整轴、选中下标、目标坐标与 1 基行号。
    # 异常：range_mode 非法、选出 0 个点、选出的点不连续时抛 ValueError。
    # "必须连续"的约束来自线性插值的语义：映射只支持连续切片，跳点选择无意义。

    full_axis = build_hplc_target_axis(config)
    # positions 是 0..point_count-1 的整数下标，选择操作先在下标空间完成，
    # 再用下标回取真实坐标，避免浮点比较参与"第几个点"的定位
    positions = np.arange(config.point_count, dtype=np.int64)
    if range_mode == "row":
        selected_start, selected_end = _validated_hplc_row_range(
            start_row, end_row, config
        )
        # 1 基首尾包含行号 → 0 基半开切片：[start-1, end)
        indices = positions[selected_start - 1 : selected_end]
        range_label = "行范围"
    elif range_mode == "x_value":
        lower, upper = _validated_hplc_time_bounds(x_min, x_max)
        # 闭区间掩码：落在 [lower, upper] 内的固定轴点全部选中
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
        # 对外展示统一回到 1 基行号：offset+1 是首行，offset+length 是末行（包含）
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
    # 关闭插值时不构造固定目标轴，选择直接作用于每个文件自己的原始 X。
    # 参数：source_x —— 当前文件已通过校验的完整原始时间轴（严格递增）。
    # 返回：(选中下标数组, 范围标签)。row 模式返回连续整数下标；
    #       x_value 模式返回满足闭区间条件的原始点下标（np.flatnonzero）。
    # 注意：行号校验仍然用 config.point_count（此时等于批次公共点数），
    # 因为关闭插值前批次已被强制统一为同一点数。

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
    # HPLC 源曲线统一校验入口：所有进入映射/导出的 (x, y) 都必须先过这里。
    # 参数：
    #   source_x/source_y  解析得到的原始时间与强度数组
    #   source_name        文件名，仅用于错误消息定位
    #   require_point_count  是否强制点数恰为 config.point_count
    #                        （批次预检阶段先放宽，逐文件映射时再强制）
    #   require_coverage   是否强制源轴覆盖 0–50 全程（开启插值前的严格模式
    #                      禁止外推；map_hplc_intensity 内部放宽，由它自己的
    #                      "一个采样间隔"规则替代）
    # 返回：(x, y)，x 是 float64 副本且端点已按容差吸附到网格起止。
    # 异常：任一校验失败抛 ValueError，消息中附"未生成结果文件"提示前端本次
    #       请求整体失败（批次级原子性：任何一个文件不合格都不产出）。
    x = np.asarray(source_x, dtype=np.float64).copy()
    y = np.asarray(source_y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError(f"{source_name} 的时间轴和强度必须是一维且长度一致")
    if len(x) < 2:
        raise ValueError(f"{source_name} 解析到 {len(x)} 个有效色谱点，至少需要 2 个点")
    if require_point_count and len(x) != config.point_count:
        raise ValueError(
            f"{source_name} 解析到 {len(x)} 个有效色谱点，HPLC 固定流程要求恰好 "
            f"{config.point_count} 个点；未生成结果文件"
        )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"{source_name} 含有 NaN 或无穷值；未生成结果文件")
    # 源 X 必须严格递增：线性插值与"左右邻点"语义都要求单调。
    # 用 diff <= 0 一次性检查重复与倒序。
    differences = np.diff(x)
    invalid = np.flatnonzero(differences <= 0)
    if invalid.size:
        index = int(invalid[0])
        raise ValueError(
            f"{source_name} 的时间轴在第 {index + 1}、{index + 2} 个点不严格递增"
            f"（{x[index]:.12g}、{x[index + 1]:.12g} {config.unit}）；未生成结果文件"
        )
    # 端点吸附：仪器写出的首尾时间常带微小相位差（如 0.0000001、49.9999999），
    # 在容差内直接对齐到网格起止，避免端点处被迫进入外推分支。
    if abs(x[0] - config.start_minutes) <= config.tolerance_minutes:
        x[0] = config.start_minutes
    if abs(x[-1] - config.stop_minutes) <= config.tolerance_minutes:
        x[-1] = config.stop_minutes
    # 吸附理论上只可能把端点向外推一点点，但仍再验一次单调性，防御极端输入
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


def _hplc_batch_message(
    file_results: list[dict[str, object]],
    point_count_groups: list[dict[str, object]],
    expected_point_count: int | None,
) -> str | None:
    # 汇总批次检查结论为一条面向用户的错误消息；批次完全可处理时返回 None。
    # 三类失败按优先级输出：
    #   1. 存在无法解析的文件 → 逐个列出文件名与原因
    #   2. 存在唯一众数点数但有少数文件不一致 → 列出全部异常文件及实际点数
    #   3. 众数并列（没有唯一多数点数）→ 列出全部分组，拒绝猜测
    invalid = [item for item in file_results if item["status"] == "invalid"]
    if invalid:
        details = "；".join(
            f"{item['name']}（{item.get('message') or '无法解析'}）" for item in invalid
        )
        return f"以下 HPLC 文件无法解析：{details}；未生成结果文件"

    mismatched = [
        item for item in file_results if item["status"] == "point_count_mismatch"
    ]
    if expected_point_count is not None and mismatched:
        details = "、".join(
            f"{item['name']}（{item['point_count']} 点）" for item in mismatched
        )
        return (
            f"批次多数文件为 {expected_point_count} 个有效色谱点；以下文件点数不一致："
            f"{details}；未生成结果文件"
        )

    if len(point_count_groups) > 1 and expected_point_count is None:
        details = "；".join(
            f"{item['point_count']} 点（{', '.join(item['files'])}）"
            for item in point_count_groups
        )
        return f"批次没有唯一多数点数：{details}；未生成结果文件"
    return None


def _inspect_hplc_batch(
    files: Iterable[str | Path],
    display_names: list[str] | None = None,
) -> tuple[dict[str, object], list[tuple[np.ndarray, np.ndarray] | None]]:
    """按正式解析规则检查批次，并保留成功解析的数组供预处理复用。"""
    # 批次预检核心：前端选择文件后立即调用，逐文件展示点数/时间范围/状态；
    # 正式预处理也复用本函数，避免"检查通过、处理时却失败"的双路径漂移。
    # 返回：
    #   inspection   含 files（逐文件状态）、point_count_groups（点数分组）、
    #                expected_point_count（唯一众数或 None）、common_point_count、
    #                processable（整体是否可处理）、message（失败原因或 None）
    #   parsed_pairs 与 files 等长的列表；解析成功为 (x, y) 校验后数组，失败为 None。
    #                保留它们是为了预处理阶段不必二次读取磁盘文件。
    paths = [Path(item) for item in files]
    if not paths:
        raise ValueError("至少需要上传一个文件")

    file_results: list[dict[str, object]] = []
    parsed_pairs: list[tuple[np.ndarray, np.ndarray] | None] = []
    for index, path in enumerate(paths):
        # display_names 保存浏览器端原始文件名（宽表 Name 列契约要求），
        # 缺省时退回磁盘文件名
        name = (
            str(display_names[index])
            if display_names and index < len(display_names)
            else path.name
        )
        try:
            source_x, source_y = read_raw_spectrum(path, kind="hplc")
            # 预检阶段用"本文件自己的点数"构造临时配置，只做结构校验，
            # 不做跨文件点数一致性强制（require_point_count=False），
            # 也不强制覆盖 0–50（require_coverage=False）——这两点留待
            # 批次统一后由正式映射流程把关。
            provisional_config = hplc_grid_config(len(source_x))
            validated_x, validated_y = _validated_hplc_source(
                source_x,
                source_y,
                name,
                provisional_config,
                require_point_count=False,
                require_coverage=False,
            )
            point_count = int(len(validated_x))
            file_results.append(
                {
                    "name": name,
                    "point_count": point_count,
                    "x_start": float(validated_x[0]),
                    "x_stop": float(validated_x[-1]),
                    "status": "ready",
                    "message": None,
                }
            )
            parsed_pairs.append((validated_x, validated_y))
        except Exception as exc:
            # 单文件失败不中断整批：记录为 invalid，由批次消息统一报告
            file_results.append(
                {
                    "name": name,
                    "point_count": None,
                    "x_start": None,
                    "x_stop": None,
                    "status": "invalid",
                    "message": str(exc),
                }
            )
            parsed_pairs.append(None)

    # 点数众数判定：只在解析成功的文件里统计。
    # Counter 取最高频次；若只有一个点数取得最高频次则为唯一众数，
    # 否则 expected_point_count=None（众数并列，拒绝猜测）。
    valid_counts = [
        int(item["point_count"])
        for item in file_results
        if item["status"] != "invalid" and item["point_count"] is not None
    ]
    count_frequency = Counter(valid_counts)
    highest_frequency = max(count_frequency.values(), default=0)
    modes = sorted(
        count for count, frequency in count_frequency.items() if frequency == highest_frequency
    )
    expected_point_count = modes[0] if len(modes) == 1 else None

    # 点数分组明细：无论成败都给前端展示，让用户看清"哪些文件各是多少点"
    point_count_groups = [
        {
            "point_count": int(point_count),
            "file_count": int(count_frequency[point_count]),
            "files": [
                str(item["name"])
                for item in file_results
                if item["point_count"] == point_count
            ],
        }
        for point_count in sorted(count_frequency)
    ]
    # 存在唯一众数且批次内有多种点数时，把少数派文件标记为 point_count_mismatch，
    # 前端可据此高亮异常文件
    if expected_point_count is not None and len(count_frequency) > 1:
        for item in file_results:
            if (
                item["status"] == "ready"
                and item["point_count"] != expected_point_count
            ):
                item["status"] = "point_count_mismatch"
                item["message"] = (
                    f"实际 {item['point_count']} 点，批次多数文件为 "
                    f"{expected_point_count} 点"
                )

    # 可处理的充要条件：有文件、全部 ready、且只有一种点数
    processable = (
        bool(file_results)
        and all(item["status"] == "ready" for item in file_results)
        and len(count_frequency) == 1
    )
    common_point_count = expected_point_count if processable else None
    message = _hplc_batch_message(
        file_results,
        point_count_groups,
        expected_point_count,
    )
    return (
        {
            "files": file_results,
            "point_count_groups": point_count_groups,
            "expected_point_count": expected_point_count,
            "common_point_count": common_point_count,
            "processable": processable,
            "message": message,
        },
        parsed_pairs,
    )


def inspect_hplc_files(
    files: Iterable[str | Path],
    display_names: list[str] | None = None,
) -> dict[str, object]:
    """返回可直接给前端展示的逐文件 HPLC 点数检查结果。"""
    # 对外薄封装：只暴露检查结论，丢弃解析数组（检查接口不需要它们）
    inspection, _pairs = _inspect_hplc_batch(files, display_names=display_names)
    return inspection


def map_hplc_intensity(
    source_x: np.ndarray,
    source_y: np.ndarray,
    source_name: str,
    config: HplcGridConfig,
    target_x: np.ndarray | None = None,
) -> np.ndarray:
    """把源强度映射到固定轴或其范围切片，边界仅允许一个采样间隔内线性延伸。"""
    # 单条曲线的 float64 线性映射核心。
    # 参数：
    #   source_x/source_y  完整源曲线（会先过 _validated_hplc_source；
    #                      require_point_count 默认 True，即强制公共点数）
    #   target_x           目标坐标；None 表示整条固定轴，否则为固定轴的连续切片
    # 返回：与 target_x 等长的 float64 强度数组。
    # 异常：目标轴非法（非一维/空/非有限/不递增）、或源轴与目标范围两端差距
    #       超过一个采样间隔时抛 ValueError。
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

    # 边界容忍度 = max(源采样间隔, 目标采样间隔) + 网格容差。
    # 业务规则：边界相位差不超过一个采样间隔时允许用首尾两点线性延伸；
    # 超过则视为数据缺失，拒绝外推。用中位数而不是首尾差估计采样间隔，
    # 对个别非均匀采样更稳健。
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

    # np.interp 在 [x[0], x[-1]] 内部做线性插值；越界部分它只会取端点常数，
    # 因此下面两个分支把越界的一小段替换为首尾两点的线性延伸（斜率外推）。
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
    # 用于给前端一个直观的"轴是否一致"提示；最终是否可写宽表仍以
    # build_wide_modeling_frame 的严格表头比对为准。
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
    # 生成写入结果/元数据的固定轴描述：起止、单位、步长、映射方式、
    # 网格参数与 1 基选中行号，便于前端与结果页如实展示"这次映射做了什么"。
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
    config: HplcGridConfig | None = None,
) -> dict:
    """保留既有范围/开关交互；开启时映射到配置轴，关闭时导出所选原轴。"""
    # HPLC 预处理对外主入口（对应 /api/preprocess/hplc）。
    # 流程：批次预检 → 构造/核对配置 → 按 interpolate 分支选择目标轴或原轴
    #       → 逐文件映射/截取 → 交给 parsers 构造 wide-feature-v2 宽表
    #       → 组装前端预览曲线与轴元数据。
    # 参数要点：interpolate=True 走固定轴线性映射；False 导出所选原始 X/Y。
    # 返回 dict 关键字段：frame（宽表）、curves（前端预览）、hplc_axis（轴元数据，
    # 关闭插值时为 None）、inspection（批次检查结论）、output_precision（精度契约）。
    files = list(files)
    if not files:
        raise ValueError("至少需要上传一个文件")
    names = [
        str(display_names[index])
        if display_names and index < len(display_names)
        else Path(file_path).name
        for index, file_path in enumerate(files)
    ]
    # 批次原子性：预检不通过则整体拒绝，绝不产出半个结果文件
    inspection, inspected_pairs = _inspect_hplc_batch(files, display_names=names)
    if not bool(inspection["processable"]):
        raise ValueError(str(inspection["message"] or "HPLC 批次检查未通过"))
    detected_point_count = int(inspection["common_point_count"])
    if config is None:
        # 正常路径：点数由批次动态决定，不写死
        config = hplc_grid_config(detected_point_count)
    elif config.point_count != detected_point_count:
        # 显式注入的配置（主要用于测试）也必须与批次实际一致
        raise ValueError(
            f"HPLC 批次统一为 {detected_point_count} 个有效色谱点，但显式配置要求 "
            f"{config.point_count} 点；未生成结果文件"
        )
    full_pairs = [pair for pair in inspected_pairs if pair is not None]

    if interpolate:
        # 开启插值：在固定目标轴上做范围选择，所有文件共享同一 target_x
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
    warnings: list[str] = []
    if interpolate:
        assert selection is not None
        target_x = selection.target_x
        # 逐文件把完整源曲线映射到同一段目标坐标；任一文件失败则整体失败
        mapped_intensities = [
            map_hplc_intensity(full_x, full_y, name, config, target_x)
            for (full_x, full_y), name in zip(full_pairs, names)
        ]
        x_arrays = [target_x] * len(full_pairs)
        intensity_arrays = mapped_intensities
        x_axis_consistent = True
    else:
        # 关闭插值：在每条原始轴上独立套用同一范围请求，导出原始 X/Y
        selected_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for (validated_x, validated_y), name in zip(full_pairs, names):
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
                raise ValueError(f"{name} 在所选{range_label}内没有数据")
            selected_pairs.append((selected_x, selected_y))
        x_arrays = [pair[0] for pair in selected_pairs]
        intensity_arrays = [pair[1] for pair in selected_pairs]
        x_axis_consistent = hplc_x_axes_consistent(x_arrays)

    n_files = len(files)
    try:
        # 宽表契约兜底：build_wide_modeling_frame 会逐点比对各文件表头，
        # 不一致直接抛错——宽表物理上只能保存一条公共真实轴
        wide = build_wide_modeling_frame(
            indices=list(range(1, n_files + 1)),
            x_arrays=x_arrays,
            intensity_arrays=intensity_arrays,
            source_names=names,
        )
    except ValueError as exc:
        if not interpolate and "XXX 与" in str(exc):
            # 关闭插值时轴不一致是典型用户错误，追加可操作的修复提示
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
        "inspection": inspection,
    }

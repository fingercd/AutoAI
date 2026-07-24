"""Document-aligned temporal convolutional classifier."""

# ============================================================================
# 模块说明（教学注释）
#
# 本文件实现一维时序卷积网络（TCN1DDocumentV2），是项目 15 个目标分类
# 模型中 `tcn1d` 的网络定义。它处在"预处理 → 训练 → 可解释性"流水线的
# 模型层：上游由训练模块根据 wide-feature-v2 宽表解析出的 input_length、
# class_count、train_sample_count 构造模型；下游被训练循环调用 forward，
# 并通过 gradcam_target_layer() 向 1D Grad-CAM 可解释性模块暴露目标层
# （最后一个 TCN block 的第二层卷积）。
#
# 关键设计约束：
# - 输入约定为单通道光谱/色谱曲线 (batch, 1, length)。
# - TCN 的核心是"膨胀卷积 + 残差连接"：dilation 按 (1,2,4) 或 (1,2,4,8)
#   指数增长，使感受野随深度指数扩大，适合长序列光谱；v2 版本只允许这两
#   种 dilation 组合，其他组合直接拒绝。
# - 通道数按样本量档位 (small/medium/large) 自动选择，kernel/pool 按特征
#   长度档位 (short/medium/long) 自动选择；也可通过 profile["values"]
#   显式覆盖，但会经过严格校验。
# - 残差块保持序列长度不变（same-length），便于任意堆叠；下采样只发生
#   在 stem 之后的一次 MaxPool。
# - 本文件只定义网络结构，不涉及数据读取、标准化或评估口径。
# ============================================================================

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


# 通道数档位表：按训练样本量分 small/medium/large 三档。
# TCN 所有残差块共用同一个通道宽度，因此每档只有一个整数；
# 样本越多通道越宽，数据充足时才支撑得起更大的参数量。
N_CHANNELS: dict[str, int] = {"small": 32, "medium": 64, "large": 128}

# 长度档位表：按特征点数分 short/medium/long 三档。
# stem_kernel 是入口卷积 kernel，stem_pool 是 stem 后唯一一次下采样倍率，
# dilations 是 TCN 残差块的膨胀系数序列；长序列使用更大的首级 pool 并
# 追加 dilation=8 的一层，以获得更大的指数级感受野。
L_PROFILE: dict[str, dict[str, Any]] = {
    "short": {"stem_kernel": 7, "stem_pool": 2, "dilations": (1, 2, 4)},
    "medium": {"stem_kernel": 9, "stem_pool": 4, "dilations": (1, 2, 4)},
    "long": {"stem_kernel": 9, "stem_pool": 4, "dilations": (1, 2, 4, 8)},
}


def _sample_band(sample_count: int) -> str:
    """按训练样本数划分档位：<=100 为 small，<300 为 medium，否则 large。"""
    return "small" if sample_count <= 100 else ("medium" if sample_count < 300 else "large")


def _feature_band(feature_count: int) -> str:
    """按特征点数（曲线长度）划分档位：<=1000 为 short，<3000 为 medium，否则 long。"""
    return "short" if feature_count <= 1000 else ("medium" if feature_count < 3000 else "long")


def _optional_int(value: Any) -> int | None:
    """把可能为 None/空字符串的配置值解析为 int；空值返回 None 以便走默认逻辑。"""
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    """同上，解析为 float；空值返回 None。"""
    if value in (None, ""):
        return None
    return float(value)


def _profile_values(
    profile: Mapping[str, Any] | Any | None,
) -> tuple[dict[str, Any], int | None, int | None, float | None]:
    """从 profile 中统一抽取超参覆盖值与元信息。

    profile 既可以是 dict（Mapping），也可以是带同名属性的对象（如
    dataclass/pydantic 模型），因此这里做双分支兼容。返回四元组：
    (values 覆盖字典, train_sample_count, feature_count, dropout)，
    任何字段缺失都为 None，由构造函数再套用默认规则。
    """
    if profile is None:
        return {}, None, None, None
    if isinstance(profile, Mapping):
        values = profile.get("values", {})
        return (
            dict(values) if isinstance(values, Mapping) else {},
            _optional_int(profile.get("train_sample_count")),
            _optional_int(profile.get("feature_count")),
            _optional_float(profile.get("dropout")),
        )
    values = getattr(profile, "values", {})
    return (
        dict(values) if isinstance(values, Mapping) else {},
        _optional_int(getattr(profile, "train_sample_count", None)),
        _optional_int(getattr(profile, "feature_count", None)),
        _optional_float(getattr(profile, "dropout", None)),
    )


def _resolve_class_count(class_count: int | None, num_classes: int | None) -> int:
    """解析类别数，兼容新旧两套参数名。

    `num_classes` 是旧版调用方使用的别名；两者同时给出时必须一致，
    否则说明上游传参矛盾，直接抛 ValueError 而不是静默选一个。
    返回值保证是正整数。
    """
    if class_count is None:
        class_count = num_classes
    elif num_classes is not None and int(class_count) != int(num_classes):
        raise ValueError("class_count 与 num_classes 必须一致")
    if class_count is None or int(class_count) <= 0:
        raise ValueError("class_count 必须是正整数")
    return int(class_count)


def _same_length_pair(first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """把主路与残差路两条张量截断到共同的最短长度。

    偶数 kernel 的非对称 padding 可能让主路比残差路多 1 个点，
    残差相加要求形状完全一致，因此统一裁到较短者（至多损失尾部 1 点）。
    """
    length = min(first.shape[-1], second.shape[-1])
    return first[..., :length], second[..., :length]


class TCNBlock1DDocumentV2(nn.Module):
    """Causal-compatible same-length residual block with two dilated convs."""
    # 中文补充（设计意图）：
    # 这是 TCN 的标准残差块：主路为两层同膨胀系数的卷积
    # （Conv→BN→ReLU→Dropout→Conv→BN），残差路用 1x1 卷积匹配通道数。
    # 通过 padding = dilation*(kernel-1)//2 保持序列长度不变，
    # 使块可以按 dilation 序列任意堆叠而不改变时间轴；
    # 残差连接缓解深层膨胀卷积堆叠带来的梯度消失。

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dilation: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        """构造一个膨胀残差块。

        参数：
            in_channels: 输入通道数。
            out_channels: 输出通道数；与 in_channels 不同的时候残差路
                自动启用 1x1 卷积做通道匹配。
            dilation: 膨胀系数，决定感受野间距；按 (1,2,4[,8]) 递增堆叠
                时感受野指数扩大。
            kernel_size: 卷积核大小，默认 3。
            dropout: 主路两层卷积之间的 dropout 概率；为 0 时用 Identity
                替代，避免无意义的算子开销。
        异常：dilation 或 kernel_size 非正时抛 ValueError。
        """
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.dilation = int(dilation)
        self.kernel_size = int(kernel_size)
        if self.dilation <= 0 or self.kernel_size <= 0:
            raise ValueError("dilation 和 kernel_size 必须是正整数")
        # same padding：dilation*(kernel-1) 是膨胀卷积的实际跨度，
        # 除 2 后两侧补齐可使输出长度与输入一致（奇数 kernel 严格相同，
        # 偶数时可能差 1 点，由 forward 中的 _same_length_pair 截齐）。
        padding = self.dilation * (self.kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding=padding,
            dilation=self.dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(self.out_channels)
        self.conv2 = nn.Conv1d(
            self.out_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding=padding,
            dilation=self.dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(self.out_channels)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.net = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True), self.dropout, self.conv2, self.bn2)
        # 残差捷径：通道数一致时直接恒等映射，零额外参数；
        # 不一致时用 1x1 卷积只做通道投影，不改变时间轴。
        self.shortcut = (
            nn.Identity()
            if self.in_channels == self.out_channels
            else nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, bias=False)
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：主路两卷积 + 残差捷径相加，最后过 ReLU。

        主路末端的 BN 之后刻意不接 ReLU，而是等残差相加后再激活——
        这是 ResNet 式 "post-activation" 结构，保证恒等路径畅通。
        """
        main = self.net(x)
        residual = self.shortcut(x)
        main, residual = _same_length_pair(main, residual)
        return self.activation(main + residual)


class TCN1DDocumentV2(nn.Module):
    """Lightweight TCN with only the document dilation schedule."""
    # 中文补充（整体结构）：
    # stem（单 Conv+BN+ReLU）→ 一次 MaxPool 下采样 → 按 dilations 堆叠
    # TCNBlock（长度不变）→ AdaptiveAvgPool → Dropout → Linear 分类。
    # 所有 block 等宽（同一 channels），结构比通用 TCN 更精简。

    def __init__(
        self,
        input_length: int | None = None,
        class_count: int | None = None,
        sample_count: int | float | None = None,
        dropout: float | None = None,
        hidden_size: int | None = None,
        *,
        num_classes: int | None = None,
        train_sample_count: int | None = None,
        profile: Mapping[str, Any] | Any | None = None,
    ) -> None:
        """构造 TCN1D 分类器。

        参数（主要）：
            input_length: 输入曲线点数；缺省时取 profile 的 feature_count。
            class_count / num_classes: 类别数（后者为兼容别名，必须一致）。
            sample_count: 训练样本数，用于选择通道档位；若误传 0~1 的
                float，会被兼容性地重解释为 dropout。
            dropout: dropout 概率，默认 0.25；同时用于 block 内和分类头。
            hidden_size: 若提供，作为通道数上限（宽度裁剪）。
            train_sample_count: 显式训练样本数，优先级高于 sample_count。
            profile: 解析好的 profile，其 values 可覆盖 channels、
                stem_kernel、stem_pool、kernel、dilations 等。
        异常：input_length 非正、sample_count<=1、dilations 不是
            (1,2,4) 或 (1,2,4,8)、channels/stem_kernel/pool_size 非正、
            dropout 越界时抛 ValueError。dilation 白名单是刻意的：
            限制搜索空间，保证 v2 结构可复现、可对照。
        """
        super().__init__()
        values, profile_n, profile_l, profile_dropout = _profile_values(profile)
        if input_length is None:
            input_length = profile_l
        if input_length is None or int(input_length) <= 0:
            raise ValueError("input_length 必须是正整数")
        self.input_length = int(input_length)
        self.class_count = _resolve_class_count(class_count, num_classes)

        # 历史兼容：旧调用方曾把 dropout 误传到 sample_count 位置；
        # 0~1 的 float 不可能是样本数，因此重解释为 dropout 并清空样本数。
        if isinstance(sample_count, float) and 0.0 <= sample_count <= 1.0:
            if dropout is None:
                dropout = float(sample_count)
            sample_count = None
        # 样本数解析优先级：显式 train_sample_count > sample_count >
        # profile 元信息 > 默认 100（取 small 档，保守的小模型）。
        resolved_n = int(train_sample_count or sample_count or profile_n or 100)
        if resolved_n <= 1:
            raise ValueError("sample_count 必须大于 1")
        resolved_l = int(profile_l or self.input_length)
        sample_band = _sample_band(resolved_n)
        feature_band = _feature_band(resolved_l)
        l_values = dict(L_PROFILE[feature_band])
        channels = int(values.get("channels", N_CHANNELS[sample_band]))
        if isinstance(values.get("channels"), (tuple, list)):
            # 兼容 profile 用通道序列的写法：TCN 全块等宽，只取末段值。
            channel_values = tuple(int(item) for item in values["channels"])
            channels = channel_values[-1]
        if hidden_size is not None:
            channels = max(1, min(channels, int(hidden_size)))
        # 各超参允许 profile.values 覆盖，并兼容 K0/P0 等历史键名。
        stem_kernel = int(values.get("stem_kernel", values.get("K0", l_values["stem_kernel"])))
        pool_size = int(values.get("stem_pool", values.get("pool", values.get("P0", l_values["stem_pool"]))))
        kernel_size = int(values.get("kernel", values.get("block_kernel", 3)))
        dilations_value = values.get("dilations", l_values["dilations"])
        dilations = tuple(int(dilation) for dilation in dilations_value)
        if dilations not in {(1, 2, 4), (1, 2, 4, 8)}:
            raise ValueError("TCN v2 只允许 dilation [1,2,4] 或 [1,2,4,8]")
        if channels <= 0 or stem_kernel <= 0 or pool_size <= 0:
            raise ValueError("channels、stem_kernel、pool_size 必须是正整数")
        # dropout 解析：显式参数 > profile > 默认 0.25。
        dropout_value = 0.25 if dropout is None and profile_dropout is None else float(
            dropout if dropout is not None else profile_dropout
        )
        if not 0.0 <= dropout_value <= 1.0:
            raise ValueError("dropout 必须位于 [0, 1]")

        self.sample_count = resolved_n
        self.sample_band = sample_band
        self.feature_band = feature_band
        self.channels = channels
        self.kernels = (stem_kernel, kernel_size)
        self.pool_sizes = (pool_size,)
        self.dilations = dilations
        self.dropout_value = dropout_value
        # ---- 特征提取主干 ----
        # stem 只负责把单通道输入升到目标通道数，不做下采样。
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=stem_kernel, padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        # 全网络唯一一次下采样：集中在这里压缩长度，残差块内部保持
        # 时间轴不变，使感受野增长完全由 dilation 控制、便于分析。
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, ceil_mode=True)
        # 按 dilation 序列堆叠残差块，dilation 指数递增 → 感受野指数扩大。
        self.tcn_blocks = nn.ModuleList(
            [
                TCNBlock1DDocumentV2(
                    channels,
                    channels,
                    dilation=dilation,
                    kernel_size=kernel_size,
                    dropout=dropout_value,
                )
                for dilation in dilations
            ]
        )
        # ---- 分类头 ----
        # AdaptiveAvgPool1d(1) 把任意长度压成单点，使网络对输入长度不敏感。
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout_value),
            nn.Linear(channels, self.class_count),
        )
        # 暴露 dropout 层的引用，方便外部按属性名访问/调整。
        self.dropout = self.head[2]

    # 以下 @property 是兼容视图：历史代码和可解释性模块曾用
    # blocks/features/classifier 等名字访问子模块。
    @property
    def blocks(self) -> nn.ModuleList:
        return self.tcn_blocks

    @property
    def features(self) -> nn.ModuleList:
        return self.tcn_blocks

    @property
    def classifier(self) -> nn.Sequential:
        return self.head

    def gradcam_target_layer(self) -> nn.Conv1d:
        """Return the final TCN block's second convolution."""
        # 选最后一个 block 的第二层卷积作为 Grad-CAM 目标层：它位于
        # 残差相加之前、语义最高且空间分辨率完整的特征图上。

        return self.tcn_blocks[-1].conv2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：stem → pool → 各 TCN block → 分类头，返回 logits。

        允许 (batch, length) 输入（自动升维为单通道），其余非
        (batch, 1, length) 形状直接抛 ValueError，尽早暴露上游管线错误。
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("TCN1DDocumentV2 输入必须是 (batch, 1, length)")
        x = self.pool(self.stem(x))
        for block in self.tcn_blocks:
            x = block(x)
        return self.head(x)


# 兼容别名：历史上该类有过多个名字（v1、DocumentExact 等），全部指向
# 同一实现，保证旧的导入路径和序列化引用不破裂。
TCNBlock1D = TCNBlock1DDocumentV2
TCN1D = TCN1DDocumentV2
TCN1DV2 = TCN1DDocumentV2
DocumentExactTCN1D = TCN1DDocumentV2


__all__ = [
    "N_CHANNELS",
    "L_PROFILE",
    "TCNBlock1DDocumentV2",
    "TCN1DDocumentV2",
    "TCNBlock1D",
    "TCN1D",
    "TCN1DV2",
    "DocumentExactTCN1D",
]

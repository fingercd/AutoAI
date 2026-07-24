"""Document-aligned four-branch 1D Inception classifier."""

# ============================================================================
# 模块说明（教学注释）
#
# 本文件实现一维 Inception 分类网络（Inception1DDocumentV2），是项目 15 个
# 目标分类模型中 `inception1d` 的网络定义。它处在"预处理 → 训练 → 可解释性"
# 流水线的模型层：上游由训练模块根据 wide-feature-v2 宽表解析出的
# input_length（特征点数）、class_count（类别数）、train_sample_count
# （训练样本数）构造模型；下游被训练循环调用 forward，并通过
# gradcam_target_layer() 向 1D Grad-CAM 可解释性模块暴露目标层。
#
# 关键设计约束：
# - 输入约定为单通道光谱/色谱曲线 (batch, 1, length)，length 即宽表特征列数。
# - 结构超参（通道数、kernel、pool）不由调用方硬编码，而是按样本量档位
#   (small/medium/large) 和特征长度档位 (short/medium/long) 自动选择，
#   对应 N_CHANNELS 与 L_PROFILE 两张表；调用方也可通过 profile["values"]
#   显式覆盖，但会经过严格校验。
# - v2 block 与经典 Inception 不同：四个分支全是等宽纯卷积，无 pooling
#   分支、无 bottleneck 压缩，最后按通道维拼接。
# - 本文件只定义网络结构，不做数据读取、标准化或评估；标准化只由训练集
#   拟合（见训练模块），评估口径（holdout/CV）与本文件无关。
# ============================================================================

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


# 通道数档位表：按训练样本量分 small/medium/large 三档。
# 元组含义为 (stem_channels, block1_channels, block2_channels, block3_channels)，
# 样本越多通道越宽，因为数据充足时才支撑得起更大的参数量。
N_CHANNELS: dict[str, tuple[int, int, int, int]] = {
    "small": (8, 16, 32, 32),
    "medium": (16, 32, 64, 64),
    "large": (32, 64, 128, 128),
}

# 长度档位表：按特征点数（光谱/色谱曲线长度）分 short/medium/long 三档。
# stem_kernel 决定入口卷积的感受野，branch_kernels 是 Inception 四个分支
# 各自的 kernel 大小，pools 是 stem 后与 block 间的三次下采样倍率；
# 序列越长，kernel 和首级 pool 越大，以便更快压缩长度、扩大感受野。
L_PROFILE: dict[str, dict[str, Any]] = {
    "short": {"stem_kernel": 7, "branch_kernels": (1, 3, 5, 7), "pools": (2, 2, 2)},
    "medium": {"stem_kernel": 9, "branch_kernels": (1, 3, 7, 11), "pools": (4, 2, 2)},
    "long": {"stem_kernel": 11, "branch_kernels": (1, 5, 9, 15), "pools": (4, 4, 2)},
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


def _same_length(values: list[torch.Tensor]) -> list[torch.Tensor]:
    """把多条分支输出截断到共同的最短长度。

    偶数 kernel 的 padding = kernel // 2 属于"非对称 same padding"，
    输出可能比输入多 1 个点，导致四条分支长度不齐、无法 concat。
    这里统一裁到最短长度，牺牲尾部至多 1 个点来换取拼接可行性。
    """
    length = min(value.shape[-1] for value in values)
    return [value[..., :length] for value in values]


class InceptionBlock1DDocumentV2(nn.Module):
    """Four equal-width convolution-only branches.

    The old implementation includes a pooling branch and a bottleneck.  The
    v2 block intentionally does neither: every branch is a direct Conv1d
    with the same output width, followed by its own normalization/activation.
    """
    # 中文补充（设计意图）：
    # 经典 Inception 模块含 pooling 分支和 1x1 bottleneck 压缩；v2 刻意
    # 全部去掉，四个分支都是"直接 Conv1d + BN + ReLU"，输出通道数相同。
    # 这样做参数量更可控、结构对光谱这种单通道一维信号更简洁；
    # 多尺度特征由四个不同 kernel size 的分支提供，最后按通道维拼接，
    # 因此 block 输出通道数 = branch_channels * 4。

    def __init__(
        self,
        in_channels: int,
        branch_channels: int,
        kernels: tuple[int, int, int, int] = (1, 3, 5, 7),
    ) -> None:
        """构造四分支 Inception block。

        参数：
            in_channels: 输入通道数（上一层输出通道）。
            branch_channels: 每条分支各自的输出通道数（注意不是总输出）。
            kernels: 四个分支的 kernel size，必须恰好四个正整数；
                kernel=1 的分支负责逐点混合，大 kernel 分支负责更宽的
                局部峰形模式。
        异常：branch_channels 非正、kernels 数量不为 4 或含非正值时抛
            ValueError——宁愿构造期失败，也不让畸形结构进入训练。
        """
        super().__init__()
        if int(branch_channels) <= 0:
            raise ValueError("branch_channels 必须是正整数")
        if len(kernels) != 4 or any(int(kernel) <= 0 for kernel in kernels):
            raise ValueError("Inception 必须提供四个正整数 kernel")
        self.in_channels = int(in_channels)
        self.branch_channels = int(branch_channels)
        self.kernels = tuple(int(kernel) for kernel in kernels)
        # 四条分支：bias=False 是因为每条分支后紧跟 BatchNorm，
        # BN 的平移参数已经覆盖了 bias 的作用，保留 bias 只会冗余。
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    self.in_channels,
                    self.branch_channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    bias=False,
                )
                for kernel in self.kernels
            ]
        )
        # 每条分支独立的 BatchNorm：不同尺度的卷积输出分布不同，
        # 共用一个 BN 会互相干扰统计量。
        self.branch_norms = nn.ModuleList([nn.BatchNorm1d(self.branch_channels) for _ in self.branches])
        self.activation = nn.ReLU(inplace=True)
        # 拼接后的总输出通道数，供外层安排下一层的 in_channels。
        self.out_channels = self.branch_channels * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：四分支并行卷积 → 各自 BN/ReLU → 截齐长度 → 通道维拼接。"""
        branches = [self.activation(norm(conv(x))) for conv, norm in zip(self.branches, self.branch_norms)]
        return torch.cat(_same_length(branches), dim=1)


class Inception1DDocumentV2(nn.Module):
    """Single-channel classifier with a document stem and four-branch block."""
    # 中文补充（整体结构）：
    # stem（单 Conv+BN+ReLU+MaxPool）→ 三个 InceptionBlock（block 间各一次
    # MaxPool）→ AdaptiveAvgPool 全局池化 → Dropout → Linear 分类。
    # 结构超参按"样本量档位 × 特征长度档位"自动确定，也可被 profile 覆盖。

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
        inception_blocks: int = 1,
    ) -> None:
        """构造 Inception1D 分类器。

        参数（主要）：
            input_length: 输入曲线点数；缺省时取 profile 的 feature_count。
            class_count / num_classes: 类别数（后者为兼容别名，必须一致）。
            sample_count: 训练样本数，用于选择通道档位；兼容一个历史误用——
                若传入 0~1 的 float，会被当作 dropout 处理（见下）。
            dropout: 分类头 dropout 概率，默认 0.25。
            hidden_size: 若提供，作为 stem 通道数的上限（宽度裁剪）。
            train_sample_count: 显式训练样本数，优先级高于 sample_count。
            profile: 解析好的 profile（dict 或对象），其 values 可覆盖
                stem_channels/block_channels/branch_kernels/pools 等。
            inception_blocks: 保留的兼容参数，当前结构固定为三个 block，
                该参数不参与构图。
        异常：input_length 非正、sample_count<=1、block_channels 不是三个
            可被 4 整除的正整数（每 block 四分支均分通道）、kernels/pools
            数量或取值非法、dropout 越界时抛 ValueError。
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

        # 通道配置：profile.values 覆盖优先，否则按样本档位取默认；
        # block_channels 必须能被 4 整除，因为每个 block 内部四条分支
        # 均分通道后再拼接。
        default_channels = N_CHANNELS[sample_band]
        stem_channels = int(values.get("stem_channels", default_channels[0]))
        block_channels = tuple(int(item) for item in values.get("block_channels", default_channels[1:]))
        if len(block_channels) != 3 or any(channel % 4 != 0 for channel in block_channels):
            raise ValueError("block_channels 必须包含三个可被4整除的正整数")
        stem_kernel = int(values.get("stem_kernel", values.get("K0", l_values["stem_kernel"])))
        kernels_value = values.get("branch_kernels", values.get("kernels", l_values["branch_kernels"]))
        kernels = tuple(int(kernel) for kernel in kernels_value)
        if len(kernels) != 4 or any(kernel <= 0 for kernel in kernels):
            raise ValueError("branch_kernels 必须包含四个正整数")
        pools = tuple(int(item) for item in values.get("pools", l_values["pools"]))
        if len(pools) != 3 or min(pools) <= 0:
            raise ValueError("pools 必须包含三个正整数")
        # dropout 解析：显式参数 > profile > 默认 0.25。
        dropout_value = 0.25 if dropout is None and profile_dropout is None else float(
            dropout if dropout is not None else profile_dropout
        )
        if not 0.0 <= dropout_value <= 1.0:
            raise ValueError("dropout 必须位于 [0, 1]")
        if hidden_size is not None:
            stem_channels = max(1, min(stem_channels, int(hidden_size)))

        self.sample_count = resolved_n
        self.sample_band = sample_band
        self.feature_band = feature_band
        self.branch_channels = tuple(channel // 4 for channel in block_channels)
        self.channels = (stem_channels, *block_channels)
        self.kernels = (stem_kernel, *kernels)
        self.pool_sizes = pools
        self.dropout_value = dropout_value
        # ---- 特征提取主干 ----
        # stem：单通道输入升到 stem_channels，紧跟第一次 MaxPool 压缩长度。
        self.stem = nn.Sequential(
            nn.Conv1d(1, stem_channels, kernel_size=stem_kernel, padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pools[0], stride=pools[0], ceil_mode=True),
        )
        # 三个 Inception block 串联；block 之间插入 MaxPool（pools[1]、
        # pools[2]），最后一个 block 后不再池化（用 Identity 占位，
        # 由分类头的 AdaptiveAvgPool 收尾）。
        blocks: list[InceptionBlock1DDocumentV2] = []
        block_pools: list[nn.Module] = []
        in_channels = stem_channels
        for index, out_channels in enumerate(block_channels):
            block = InceptionBlock1DDocumentV2(in_channels, out_channels // 4, kernels=kernels)
            blocks.append(block)
            in_channels = block.out_channels
            pool_size = pools[index + 1] if index < 2 else 1
            block_pools.append(nn.MaxPool1d(pool_size, pool_size, ceil_mode=True) if pool_size > 1 else nn.Identity())
        self.inception_blocks = nn.ModuleList(blocks)
        self.block_pools = nn.ModuleList(block_pools)
        # ---- 分类头 ----
        # AdaptiveAvgPool1d(1) 把任意长度压成单点，使网络对输入长度不敏感；
        # Dropout 后接线性层输出 logits。
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout_value),
            nn.Linear(in_channels, self.class_count),
        )
        # 暴露 dropout 层的引用，方便外部按属性名访问/调整。
        self.dropout = self.head[2]

    # 以下一组 @property 是兼容视图：历史代码和可解释性模块曾用
    # blocks/features/inception/pools/classifier 等名字访问子模块，
    # 保留这些别名可以避免改动调用方。
    @property
    def blocks(self) -> nn.ModuleList:
        return self.inception_blocks

    @property
    def features(self) -> nn.Sequential:
        return nn.Sequential(self.stem, *self.inception_blocks)

    @property
    def inception(self) -> InceptionBlock1DDocumentV2:
        return self.inception_blocks[-1]

    @property
    def pools(self) -> tuple[nn.Module, ...]:
        return (self.stem[3], *self.block_pools[:2])

    @property
    def classifier(self) -> nn.Sequential:
        return self.head

    def gradcam_target_layer(self) -> InceptionBlock1DDocumentV2:
        """Return the final concatenating block for 1D Grad-CAM."""
        # 选最后一个 Inception block 作为 Grad-CAM 目标层：它是语义最高、
        # 空间分辨率仍保留的拼接特征层，回投到 1D 特征轴的解释性最好。

        return self.inception_blocks[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：stem → (block+pool) x 3 → 分类头，返回 logits。

        允许 (batch, length) 输入（自动升维为单通道），其余非
        (batch, 1, length) 形状直接抛 ValueError，尽早暴露上游管线错误。
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("Inception1DDocumentV2 输入必须是 (batch, 1, length)")
        x = self.stem(x)
        for block, pool in zip(self.inception_blocks, self.block_pools):
            x = block(x)
            x = pool(x)
        return self.head(x)


# 兼容别名：历史上该类有过多个名字（v1、Module、DocumentExact 等），
# 全部指向同一实现，保证旧的导入路径和序列化引用不破裂。
InceptionBlock1D = InceptionBlock1DDocumentV2
InceptionModule1D = InceptionBlock1DDocumentV2
Inception1D = Inception1DDocumentV2
Inception1DV2 = Inception1DDocumentV2
DocumentExactInception1D = Inception1DDocumentV2


__all__ = [
    "N_BRANCH_CHANNELS",
    "L_PROFILE",
    "InceptionBlock1DDocumentV2",
    "Inception1DDocumentV2",
    "InceptionBlock1D",
    "InceptionModule1D",
    "Inception1D",
    "Inception1DV2",
    "DocumentExactInception1D",
]

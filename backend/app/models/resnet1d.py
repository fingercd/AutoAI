"""Document-aligned 1D ResNet classifier.

This module is deliberately versioned next to the legacy implementation.  It
does not import the registry or the shared profile factory, so it can be
promoted by a later integration task without changing the legacy state-dict
contract.
"""

# ======================================================================
# 模块级说明
#
# 本文件实现能力目录中的 `resnet1d`：面向单通道光谱/色谱曲线的文档对齐
# 1D ResNet 分类器（stem 卷积 + 三个残差块 + 全局池化分类头）。
#
# 在系统中的位置：
#   - 位于 backend/app/models/，与 cnn1d.py / cnn_se1d.py 同属分类 v2
#     深度模型族，共享同一套“样本量分档 N / 特征长度分档 L”思路：
#     N 决定通道宽度，L 决定卷积核与池化强度（见 N_CHANNELS / L_PROFILE）。
#   - 刻意与旧版实现并存（“versioned next to the legacy implementation”），
#     不 import registry 和共享 profile 工厂，以便后续集成任务直接提升本版本，
#     而不改变旧版 state-dict 契约（即已保存模型权重的键结构保持可用）。
#
# 协作模块：
#   - 训练编排层实例化 `ResNet1DDocumentV2` 并传入 input_length（L）、
#     sample_count（当前折的 N_train）和 class_count。
#   - 可解释性模块通过 `gradcam_target_layer()` 取得最后一个残差块主路径的
#     conv2，执行 1D Grad-CAM。
#   - 上游数据契约是 wide-feature 宽表：每条曲线一行，输入形状 Batch×1×L。
#
# 关键设计约束：
#   - 残差结构缓解深层网络梯度消失；shortcut 在通道数变化时用 1×1 卷积对齐。
#   - 卷积 padding=kernel//2；偶数核产生的边界长度差由 _same_length_pair
#     裁齐后再做残差相加。
#   - 兼容旧构造签名：位置参数 (dropout, hidden_size) 的历史写法仍被接受。
# ======================================================================

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


# 样本量分档 → 三个阶段的通道宽度。样本越少网络越窄，抑制小样本过拟合。
N_CHANNELS: dict[str, tuple[int, int, int]] = {
    "small": (8, 16, 32),
    "medium": (16, 32, 64),
    "large": (32, 64, 128),
}

# 特征长度分档 → 卷积核与池化配置。
# kernels 的第 1 个值用于 stem 卷积，第 2/3 个值用于残差块；
# pools 的第 1 个值用于 stem 池化，第 2 个值用于第 2 个残差块的块后池化。
# 曲线越长，早期下采样越强，以压缩时序长度、控制计算量。
L_PROFILE: dict[str, dict[str, Any]] = {
    "short": {"kernels": (7, 5, 3), "pools": (2, 2, 2)},
    "medium": {"kernels": (9, 5, 3), "pools": (4, 2, 2)},
    "long": {"kernels": (9, 7, 5), "pools": (4, 4, 2)},
}


# 按训练样本数划分样本档位：<=100 小样本，<300 中样本，其余大样本
def _band_for_sample_count(sample_count: int) -> str:
    return "small" if sample_count <= 100 else ("medium" if sample_count < 300 else "large")


# 按输入特征长度划分特征档位：<=1000 短曲线，<3000 中等，其余长曲线
def _band_for_feature_count(feature_count: int) -> str:
    return "short" if feature_count <= 1000 else ("medium" if feature_count < 3000 else "long")


# 从 profile 对象中提取四元组：(values 子映射, train_sample_count,
# feature_count, dropout)。profile 可以是 Mapping 或带同名属性的对象；
# 缺失的字段统一返回 None，让调用方走默认值分支。
# 这样未来的共享 builder 可以直接把解析好的 profile 传进来。
def _profile_values(
    profile: Mapping[str, Any] | Any | None,
) -> tuple[dict[str, Any], int | None, int | None, float | None]:
    if profile is None:
        return {}, None, None, None
    if isinstance(profile, Mapping):
        values = profile.get("values", {})
        values = dict(values) if isinstance(values, Mapping) else {}
        return (
            values,
            _optional_int(profile.get("train_sample_count")),
            _optional_int(profile.get("feature_count")),
            _optional_float(profile.get("dropout")),
        )
    values = getattr(profile, "values", {})
    values = dict(values) if isinstance(values, Mapping) else {}
    return (
        values,
        _optional_int(getattr(profile, "train_sample_count", None)),
        _optional_int(getattr(profile, "feature_count", None)),
        _optional_float(getattr(profile, "dropout", None)),
    )


# 宽松地把外部值转成 int；None 和空字符串视为“未提供”
def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


# 宽松地把外部值转成 float；None 和空字符串视为“未提供”
def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


# 解析类别数：class_count 与 num_classes 同时给出时必须一致，
# 防止两种命名传入不同值造成静默口径冲突；缺失或非正整数时拒绝。
def _resolve_class_count(class_count: int | None, num_classes: int | None) -> int:
    if class_count is None:
        class_count = num_classes
    elif num_classes is not None and int(class_count) != int(num_classes):
        raise ValueError("class_count 与 num_classes 必须一致")
    if class_count is None or int(class_count) <= 0:
        raise ValueError("class_count 必须是正整数")
    return int(class_count)


# 解析 dropout：未提供时默认 0.25；允许取到 1.0（与 cnn1d 的 [0,1) 不同），
# 越界则拒绝。
def _resolve_dropout(value: float | None) -> float:
    dropout = 0.25 if value is None else float(value)
    if not 0.0 <= dropout <= 1.0:
        raise ValueError("dropout 必须位于 [0, 1]")
    return dropout


# 把两个张量在长度维裁齐到较短者。
# 文档允许偶数 kernel 存在，此时 padding=kernel//2 无法保持主路径与
# shortcut 路径长度完全一致；残差相加前必须裁掉多出的边界点，
# 否则逐元素相加会因形状不匹配而失败。
def _same_length_pair(first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop only the extra boundary produced by an even document kernel."""

    length = min(first.shape[-1], second.shape[-1])
    return first[..., :length], second[..., :length]


# 把外部值强制转换为正整数元组；length 给定时不匹配即拒绝。
# 用于校验 profile 显式传入的 channels/kernels/pools。
def _as_int_tuple(value: Any, *, name: str, length: int | None = None) -> tuple[int, ...]:
    if isinstance(value, int):
        result = (int(value),)
    else:
        try:
            result = tuple(int(item) for item in value)
        except TypeError as exc:
            raise ValueError(f"{name} 必须是整数或整数序列") from exc
    if not result or any(item <= 0 for item in result):
        raise ValueError(f"{name} 必须包含正整数")
    if length is not None and len(result) != length:
        raise ValueError(f"{name} 必须包含 {length} 个整数")
    return result


# 双卷积残差块：Conv-BN-ReLU-Conv-BN 主路径 + shortcut 相加后 ReLU，
# 再做可选的块后池化和 dropout。
#
# 设计要点：
#   - shortcut 在输入/输出通道数一致时是 Identity（零成本恒等映射），
#     不一致时用 1×1 卷积只做通道对齐，不改变时间维分辨率。
#   - 卷积 bias=False：紧跟 BatchNorm 时偏置会被归一化抵消。
#   - 池化用 ceil_mode=True：长度不整除时向上取整，保留末尾信号。
#   - dropout=0 时用 Identity 占位，保持模块结构稳定（state-dict 键不变）。
class ResidualBlock1DDocumentV2(nn.Module):
    """Two-convolution residual block with an optional post-block pool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        pool_size: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        kernel = int(kernel_size)
        if kernel <= 0:
            raise ValueError("kernel_size 必须是正整数")
        padding = kernel // 2
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = kernel
        self.pool_size = int(pool_size)
        if self.pool_size <= 0:
            raise ValueError("pool_size 必须是正整数")

        self.conv1 = nn.Conv1d(self.in_channels, self.out_channels, kernel, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(self.out_channels)
        self.conv2 = nn.Conv1d(self.out_channels, self.out_channels, kernel, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(self.out_channels)
        self.main_path = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True), self.conv2, self.bn2)
        self.shortcut = (
            nn.Identity()
            if self.in_channels == self.out_channels
            else nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, bias=False)
        )
        self.activation = nn.ReLU(inplace=True)
        self.pool = (
            nn.Identity()
            if self.pool_size == 1
            else nn.MaxPool1d(kernel_size=self.pool_size, stride=self.pool_size, ceil_mode=True)
        )
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

    # 前向：主路径与 shortcut 分别计算，裁齐长度后逐元素相加并激活，
    # 最后依次经过 dropout 和块后池化。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        main = self.main_path(x)
        main, residual = _same_length_pair(main, residual)
        result = self.activation(main + residual)
        result = self.dropout(result)
        return self.pool(result)


# 文档版三段 ResNet 分类器：stem（Conv-BN-ReLU + MaxPool）后接三个
# ResidualBlock1DDocumentV2，最后是 AdaptiveAvgPool → Flatten → Dropout →
# Linear 的分类头。
#
# profile 解析要点：
#   - sample_count 取 train_sample_count / sample_count / profile 中第一个
#     可用值，全缺时按 100（small 档）兜底；必须大于 1。
#   - 三个残差块的 kernel 取 kernels[1]、kernels[2]、kernels[2]（第三个块
#     复用第二个核），块后池化取 pools[1]、pools[2]、1（最后一块不池化，
#     由分类头的全局平均池化收尾）。
#   - hidden_size 是历史遗留旋钮：只能抬高首段宽度，不改变三段式拓扑。
class ResNet1DDocumentV2(nn.Module):
    """Three-block document ResNet for a single-channel spectral curve.

    ``sample_count`` is the current fold's N_train and ``input_length`` is L.
    Both are used only to resolve the lightweight document profile.  A
    resolved profile mapping can be passed by the future shared builder.
    """

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
        super().__init__()
        values, profile_n, profile_l, profile_dropout = _profile_values(profile)
        if input_length is None:
            input_length = profile_l
        if input_length is None or int(input_length) <= 0:
            raise ValueError("input_length 必须是正整数")
        self.input_length = int(input_length)
        self.class_count = _resolve_class_count(class_count, num_classes)

        # Accept the old positional (dropout, hidden_size) shape while keeping
        # the v2 sample_count-first constructor used by new builders.
        # 兼容旧签名：若第三个位置参数实为 [0,1] 的浮点数，说明调用方按旧的
        # (input_length, class_count, dropout, hidden_size) 顺序传参，
        # 此时把它当 dropout 用，并把 sample_count 置回未提供状态。
        if isinstance(sample_count, float) and 0.0 <= sample_count <= 1.0:
            if dropout is None:
                dropout = float(sample_count)
            sample_count = None
        resolved_n = int(train_sample_count or sample_count or profile_n or 100)
        if resolved_n <= 1:
            raise ValueError("sample_count 必须大于 1")
        resolved_l = int(profile_l or self.input_length)
        sample_band = _band_for_sample_count(resolved_n)
        feature_band = _band_for_feature_count(resolved_l)
        profile_l_values = dict(L_PROFILE[feature_band])

        # channels/kernels/pools：profile values 显式值优先，否则按档位查默认表
        channels_value = values.get("channels", N_CHANNELS[sample_band])
        channels = _as_int_tuple(channels_value, name="channels", length=3)
        kernels = _as_int_tuple(values.get("kernels", profile_l_values["kernels"]), name="kernels", length=3)
        pools = _as_int_tuple(values.get("pools", profile_l_values["pools"]), name="pools", length=3)
        stem_kernel, stem_pool = kernels[0], pools[0]
        block_kernels = (kernels[1], kernels[2], kernels[2])
        block_pools = (pools[1], pools[2], 1)
        dropout_value = _resolve_dropout(dropout if dropout is not None else profile_dropout)
        if hidden_size is not None:
            # Hidden size is a legacy knob.  It may lower the first width, but
            # never changes the document profile's three-stage topology.
            # 旧调用方传入 hidden_size 时，把它作为首段通道数的下限参考，
            # 但绝不缩小任何一段的文档宽度，也不改变三段式拓扑。
            requested = max(1, int(hidden_size))
            scale = max(1, min(requested, channels[0]))
            channels = tuple(max(scale, channel) for channel in channels)

        # 平铺解析结果为实例属性，供训练器/测试读取架构口径
        self.sample_count = resolved_n
        self.sample_band = sample_band
        self.feature_band = feature_band
        self.channels = tuple(channels)
        self.kernels = kernels
        self.pool_sizes = pools
        self.dropout_value = dropout_value

        # stem：首段卷积提低层特征后立刻池化，快速压缩时序长度
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=stem_kernel, padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.stem_pool = nn.MaxPool1d(kernel_size=stem_pool, stride=stem_pool, ceil_mode=True)
        # 三个残差块：通道数逐段提升，最后一块不做块后池化（pool_size=1 → Identity）
        blocks: list[ResidualBlock1DDocumentV2] = []
        in_channels = channels[0]
        for out_channels, pool_size, block_kernel in zip(channels, block_pools, block_kernels):
            blocks.append(
                ResidualBlock1DDocumentV2(
                    in_channels,
                    out_channels,
                    kernel_size=block_kernel,
                    pool_size=pool_size,
                )
            )
            in_channels = out_channels
        self.residual_blocks = nn.ModuleList(blocks)
        # 分类头：全局平均池化使输入维度只取决于末段通道数，与剩余长度无关
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout_value),
            nn.Linear(in_channels, self.class_count),
        )
        # 兼容旧调用方对 model.dropout 的直接访问
        self.dropout = self.head[2]

    @property
    def blocks(self) -> nn.ModuleList:
        # 兼容命名：blocks 即残差块列表
        return self.residual_blocks

    @property
    def features(self) -> nn.ModuleList:
        # 兼容命名：features 也指向残差块列表
        return self.residual_blocks

    @property
    def pools(self) -> tuple[nn.Module, ...]:
        # 按前向顺序暴露全部池化层（stem 池化 + 各残差块的块后池化），
        # 供测试核对下采样结构
        return (self.stem_pool, *(block.pool for block in self.residual_blocks))

    @property
    def classifier(self) -> nn.Sequential:
        # 兼容命名：classifier 即分类头
        return self.head

    def gradcam_target_layer(self) -> nn.Conv1d:
        """Return the final residual main-path convolution for 1D Grad-CAM."""

        # 取最后一个残差块主路径的 conv2：网络中最深的卷积特征，
        # 语义级别最高，Grad-CAM 热力图最能反映类别判别依据
        return self.residual_blocks[-1].conv2

    # 前向：接受 Batch×L 或 Batch×1×L，统一补成单通道三维；
    # 非单通道输入直接拒绝，避免形状错误在深层才暴露。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("ResNet1DDocumentV2 输入必须是 (batch, 1, length)")
        x = self.stem_pool(self.stem(x))
        for block in self.residual_blocks:
            x = block(x)
        return self.head(x)


# 对外别名：兼容历史调用方与文档中的不同命名
ResidualBlock1D = ResidualBlock1DDocumentV2
ResNet1D = ResNet1DDocumentV2
ResNet1DV2 = ResNet1DDocumentV2
DocumentExactResNet1D = ResNet1DDocumentV2


__all__ = [
    "N_CHANNELS",
    "L_PROFILE",
    "ResidualBlock1DDocumentV2",
    "ResNet1DDocumentV2",
    "ResidualBlock1D",
    "ResNet1D",
    "ResNet1DV2",
    "DocumentExactResNet1D",
]

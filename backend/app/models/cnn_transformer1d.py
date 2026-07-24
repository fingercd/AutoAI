"""CNN-Transformer classifier for long one-dimensional signals.

The module is intentionally independent from the model registry.  Training
profiles are resolved by the caller and passed to the constructor explicitly.
"""

# ============================================================================
# 模块说明（教学注释）
#
# 本文件实现 CNN + Transformer 混合分类网络（CNNTransformer1D），对应项目
# 15 个目标分类模型中的 `cnn_transformer1d`；`transformer1d` 只是它的
# 兼容别名，不是独立网络。它处在"预处理 → 训练 → 可解释性"流水线的
# 模型层。
#
# 与 inception1d/tcn1d 不同，本模块刻意不依赖模型注册表、不自行解析
# profile 策略：所有超参由调用方（共享 builder）解析后显式传入构造函数，
# 保持本模块的纯粹性。因此这里没有 sample_band/feature_band 自动档位表，
# 只有 conv_kernels/pool_sizes 按 input_length 的三档默认值。
#
# 结构分三段：
# 1) 三层 CNN 前端（Conv→BN→ReLU→MaxPool），把长光谱序列压缩成较短的
#    token 序列，通道升到 d_model；
# 2) 可学习位置编码 + batch-first TransformerEncoder，建模 token 间长程
#    依赖；
# 3) 对 token 维做 mean pooling 后接线性分类头。
#
# 关键设计约束：
# - 输入约定为单通道曲线 (batch, 1, length)。
# - 注意力头数必须整除 d_model，_resolve_heads 会自动向下缩减到合法值。
# - 位置编码长度按理论 token_length 预分配；实际长度不符时通过截取或
#   线性插值适配（见 _position_for），保证 forward 不因 ceil_mode 池化
#   的边界差异而崩溃。
# - 该模型的可解释性不走 Grad-CAM，而是走"窗口遮挡 + 真实类别 Log-loss
#   增量"的重要性分析（见训练/解释模块），本文件无需暴露 gradcam 目标层。
# ============================================================================

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


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


def _resolve_heads(d_model: int, requested_heads: int) -> int:
    """把请求的注意力头数调整为能整除 d_model 的合法值。

    PyTorch 的 MultiheadAttention 要求 d_model % nhead == 0；这里从请求值
    向下递减直到满足整除，保证至少为 1。用"缩减"而不是"报错"，是为了让
    profile 给出一个理想头数时模型总能落地。
    """
    heads = max(1, int(requested_heads))
    while heads > 1 and d_model % heads != 0:
        heads -= 1
    return heads


def _ceil_pool_length(length: int, pools: Sequence[int]) -> int:
    """按各层 MaxPool（ceil_mode=True）推算池化后的 token 序列长度。

    模型里所有 MaxPool1d 都开了 ceil_mode，因此每层长度按向上取整除法
    缩减；最后用 max(1, ...) 兜底，防止极端短输入算出 0 导致后续
    位置编码申请到空张量。
    """
    for pool in pools:
        length = math.ceil(length / int(pool))
    return max(1, length)


class CNNTransformer1D(nn.Module):
    """Three-stage CNN front-end followed by a batch-first Transformer.

    Parameters are deliberately explicit so a resolved model profile can be
    passed in by a future shared builder without this module importing or
    recomputing profile policy.
    """
    # 中文补充（设计意图）：
    # 本类只做"已解析参数 → 可训练网络"的映射，不读取数据集、不解析
    # profile 文件。大量关键字别名参数（nhead/heads、num_layers/layers）
    # 是为了兼容历史调用方和 profile 字典的不同命名习惯。

    def __init__(
        self,
        input_length: int,
        class_count: int | None = None,
        dropout: float | None = None,
        hidden_size: int = 64,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        *,
        num_classes: int | None = None,
        d_model: int | None = None,
        nhead: int | None = None,
        heads: int | None = None,
        num_layers: int | None = None,
        layers: int | None = None,
        conv_channels: Sequence[int] | None = None,
        conv_kernels: Sequence[int] | None = None,
        pool_sizes: Sequence[int] | None = None,
        dim_feedforward: int | None = None,
    ) -> None:
        """构造 CNN-Transformer 分类器。

        参数（主要）：
            input_length: 输入曲线点数，即 wide-feature 宽表的特征列数。
            class_count / num_classes: 类别数（后者为兼容别名）。
            dropout: 全局 dropout 概率，默认 0.25。
            hidden_size / d_model: Transformer 嵌入维度（后者优先）。
            transformer_heads / nhead / heads: 注意力头数，自动缩减到
                能整除 d_model 的值。
            transformer_layers / num_layers / layers: encoder 层数。
            conv_channels / conv_kernels / pool_sizes: 三段 CNN 的通道、
                kernel、池化倍率；缺省时按 input_length 档位自动选择，
                且通道末段会被对齐到 d_model（保证 CNN 输出可直接作为
                Transformer 的 token 特征）。
            dim_feedforward: encoder 前馈层宽度，默认 2 * d_model。
        异常：各尺寸参数非正、conv_channels/kernels/pools 数量不为 3、
            dropout 越界时抛 ValueError，构造期即失败。
        """
        super().__init__()
        if int(input_length) <= 0:
            raise ValueError("input_length 必须是正整数")
        self.input_length = int(input_length)
        self.class_count = _resolve_class_count(class_count, num_classes)

        if nhead is not None:
            transformer_heads = nhead
        if heads is not None:
            transformer_heads = heads
        if num_layers is not None:
            transformer_layers = num_layers
        if layers is not None:
            transformer_layers = layers
        if int(transformer_layers) <= 0:
            raise ValueError("transformer_layers 必须是正整数")

        dropout_value = 0.25 if dropout is None else float(dropout)
        if not 0.0 <= dropout_value <= 1.0:
            raise ValueError("dropout 必须位于 [0, 1]")

        requested_d_model = int(hidden_size if d_model is None else d_model)
        if requested_d_model <= 0:
            raise ValueError("d_model/hidden_size 必须是正整数")

        if conv_channels is None:
            # 默认三段通道按 d_model 的 1/4、1/2、1 递增，并设下限 8/16，
            # 防止很小的 hidden_size 导致前两段通道为 0 或过窄。
            channels = (
                max(8, requested_d_model // 4),
                max(16, requested_d_model // 2),
                requested_d_model,
            )
        else:
            channels = tuple(int(value) for value in conv_channels)
            if len(channels) != 3 or any(value <= 0 for value in channels):
                raise ValueError("conv_channels 必须包含三个正整数")
            # 显式给了 conv_channels 但没给 d_model 时，以 CNN 末段通道
            # 作为 d_model——两者必须相等，否则 CNN 输出无法直接喂给
            # Transformer（token 特征维度即 d_model）。
            if d_model is None:
                requested_d_model = channels[-1]

        self.d_model = requested_d_model
        self.transformer_heads = _resolve_heads(self.d_model, transformer_heads)
        self.transformer_layers = int(transformer_layers)
        if channels[-1] != self.d_model:
            # 最终防线：无论参数如何组合，强制末段通道 == d_model。
            channels = (*channels[:2], self.d_model)
        self.conv_channels = channels
        if conv_kernels is None or pool_sizes is None:
            # 按输入长度选默认 kernel/pool：序列越长，首层 kernel 和
            # 前两级 pool 越大，更快压缩时间轴、控制 token 数量，
            # 使 Transformer 的注意力开销保持在可承受范围。
            if self.input_length <= 1000:
                default_kernels, default_pools = (7, 5, 3), (2, 2, 2)
            elif self.input_length < 3000:
                default_kernels, default_pools = (9, 5, 3), (4, 2, 2)
            else:
                default_kernels, default_pools = (9, 7, 5), (4, 4, 2)
            conv_kernels = default_kernels if conv_kernels is None else conv_kernels
            pool_sizes = default_pools if pool_sizes is None else pool_sizes
        kernels = tuple(int(value) for value in conv_kernels)
        pools = tuple(int(value) for value in pool_sizes)
        if len(kernels) != 3 or len(pools) != 3 or min(*kernels, *pools) <= 0:
            raise ValueError("conv_kernels 和 pool_sizes 必须各包含三个正整数")
        self.conv_kernels = kernels
        self.pool_sizes = pools

        cnn_layers: list[nn.Module] = []
        # 输入是单通道光谱/色谱曲线，因此首层 in_channels 固定为 1。
        in_channels = 1
        for out_channels, kernel_size, pool_size in zip(channels, kernels, pools):
            # 每段：Conv(bias=False，因紧跟 BN)→BN→ReLU→MaxPool(ceil_mode)。
            # ceil_mode 保证长度除不尽时仍向上取整保留尾部信息，
            # 与 _ceil_pool_length 的长度推算保持一致。
            cnn_layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, ceil_mode=True),
                ]
            )
            in_channels = out_channels
        self.cnn = nn.Sequential(*cnn_layers)

        # 预推算池化后的 token 数，据此分配可学习位置编码。
        token_length = _ceil_pool_length(self.input_length, pools)
        self.token_length = token_length
        # 可学习位置编码（而非正弦固定编码）：光谱 token 序列较短，
        # 可学习编码在小数据上更灵活；初始化为零，让训练初期等价于
        # 无位置信息，避免随机初始化干扰 CNN 特征的早期学习。
        self.position = nn.Parameter(torch.zeros(1, token_length, self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.transformer_heads,
            dim_feedforward=(self.d_model * 2 if dim_feedforward is None else int(dim_feedforward)),
            dropout=dropout_value,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_layers)
        self.dropout = nn.Dropout(dropout_value)
        self.head = nn.Linear(self.d_model, self.class_count)

    @property
    def conv_blocks(self) -> nn.Sequential:
        """Compatibility view of the three flattened convolutional blocks."""

        return self.cnn

    @property
    def classifier(self) -> nn.Linear:
        """Compatibility alias matching the legacy CNN classifier name."""

        return self.head

    def _position_for(self, token_count: int) -> torch.Tensor:
        """按实际 token 数取出（或重采样出）匹配长度的位置编码。

        构造时按理论 token_length 预分配位置编码；forward 中实际长度可能
        因 ceil_mode 池化的边界效应差 1 个 token：
        - 长度一致：直接用；
        - 实际更短：截取前缀；
        - 实际更长：用线性插值把位置编码拉伸到所需长度，
          保证位置信息平滑过渡而不是补零。
        """
        if token_count == self.position.shape[1]:
            return self.position
        if token_count < self.position.shape[1]:
            return self.position[:, :token_count, :]
        return F.interpolate(
            self.position.transpose(1, 2),
            size=token_count,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：CNN 压缩 → 加位置编码 → Transformer → mean pooling → 分类。

        参数 x 允许 (batch, length) 或 (batch, 1, length) 两种形状，
        内部统一升为单通道三维；其他形状直接抛 ValueError。
        返回 (batch, class_count) 的 logits（未过 softmax，配合
        CrossEntropyLoss 使用）。
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("CNNTransformer1D 输入必须是 (batch, 1, length)")
        # transpose(1, 2)：从 (batch, C=d_model, L) 变成 Transformer
        # 需要的 (batch, token=L, feature=d_model)。
        sequence = self.cnn(x).transpose(1, 2)
        sequence = sequence + self._position_for(sequence.shape[1])
        encoded = self.encoder(sequence)
        # 对 token 维做均值池化，把变长序列聚成单向量再分类；
        # 相比取 CLS token，mean pooling 对短序列小数据集更稳。
        pooled = encoded.mean(dim=1)
        return self.head(self.dropout(pooled))


# 兼容别名：`transformer1d`/`CNNTransformer` 是历史命名，指向同一实现。
CNNTransformer = CNNTransformer1D


def build_cnn_transformer1d(
    input_length: int,
    class_count: int | None = None,
    dropout: float | None = None,
    hidden_size: int = 64,
    transformer_heads: int = 4,
    transformer_layers: int = 2,
    **kwargs: object,
) -> CNNTransformer1D:
    """Build a CNN-Transformer from already-resolved profile parameters."""
    # 中文补充：这是给训练/注册模块用的工厂函数。约定调用方已完成
    # profile 解析，这里只负责透传参数构造模型；**kwargs 允许未来新增
    # 超参时不必同步修改本函数签名。

    return CNNTransformer1D(
        input_length=input_length,
        class_count=class_count,
        dropout=dropout,
        hidden_size=hidden_size,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        **kwargs,
    )


__all__ = ["CNNTransformer1D", "CNNTransformer", "build_cnn_transformer1d"]

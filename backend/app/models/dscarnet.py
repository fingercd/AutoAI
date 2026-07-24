"""
Faithful PyTorch reproduction of the original TensorFlow DSCARNet.
Ref: https://github.com/songlinlu/DSCAR

Original single_dscarnet & dual_dscarnet are 2D convolutional networks designed
for AggMap-transformed 2D spectral activity representations (SAR / CAR).

This file mirrors the TF implementation exactly:
- 3-layer stem (64→96→128) with conditional stride based on input size
- Inception blocks with bottleneck (branch5x5 0.5x compression)
- unit = 48 * (2 ** i) channel formula
- Optional batchnorm + L2 weight decay (matching TF behaviour)
- GlobalMaxPool + configurable dense head
- Dual-input architecture for SAR + CAR fusion

Also includes DSCARNet1D — a faithful 1D adaptation for direct spectral input
that preserves the same architecture topology, channel progression, and
bottleneck ratios as the original 2D network.
"""

# ============================================================================
# 模块说明（教学注释）
#
# 本文件是 DSCARNet 模型族在 AutoAI 平台中的 PyTorch 实现，位于
# backend/app/models/ 模型注册目录下，由训练流程通过模型注册表按名称
# "dscarnet" 实例化（对应能力目录中 15 个公开分类模型之一）。
#
# 在整条链路中的位置：
#   拉曼/HPLC 预处理（wide-feature-v2 宽表）
#       → 建模读取与标准化（仅用训练集拟合）
#       → 本文件的 DSCARNet1D（或 AggMap/PCA 映射后的 2D 版本）训练
#       → 可解释性阶段：DSCARNet 走 AggMap/PCA 的 SAR/CAR 双通路 2D 映射，
#         再做双通路 2D Grad-CAM 回投到 1D 特征（见 AGENTS.md 建模规则，
#         不能当作普通 1D CNN 来解释）。
#
# 设计约束：
#   1. 对原始 TensorFlow 实现（songlinlu/DSCAR）做 1:1 结构复刻——
#      通道数公式（64→96→128 stem、unit = 48×2^i）、5×5 分支 0.5× 瓶颈、
#      "same/valid" padding 语义都不能随意改动，否则与论文/复现口径不一致。
#   2. TF 用 kernel/activity L2 正则；PyTorch 惯例是把 L2 放到优化器的
#      weight_decay，因此本文件的卷积一律 bias=False + 可选 BN，
#      并在 docstring 中提示调用方在 batchnorm=False 时给优化器配
#      较大的 weight_decay 来复刻 TF 行为。
#   3. DSCARNet1D 的构造签名保持与旧注册表调用点兼容：
#      DSCARNet1D(input_length, class_count, dropout, hidden_size,
#      inception_blocks)，其中 dropout / hidden_size 仅为兼容而接收，
#      实际不使用（TF 原版没有 Dropout，通道数是固定公式）。
#   4. 顶层任务只支持分类：输出层是 n_outputs 个 logits/概率，
#      不处理回归目标。
# ============================================================================

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 2D building blocks  —  matching TF conv2d_bn / Conv2D_BN helpers
# ============================================================================
# 本节对应 TF 代码里的两个基础助手：
#   - conv2d_bn   ：Inception 分支内部使用（TF 版本在 batchnorm=False 时带 L2）
#   - Conv2D_BN   ：stem 使用（两种情况都不带 L2）
# PyTorch 侧统一返回 nn.Sequential(Conv → [BN] → ReLU)，
# 结构上保持与 TF 完全相同的"卷积→(归一化)→激活"顺序。

def _conv2d_bn_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] | str = 0,
    batchnorm: bool = False,
) -> nn.Sequential:
    """
    TF conv2d_bn(x, filters, num_row, num_col, ...)

    When *batchnorm* is True  → Conv + BN + ReLU  (no L2).
    When *batchnorm* is False → Conv + ReLU + L2 on kernel/bias/activity.

    Note on L2: the TF code applies ``kernel_regularizer=L2(0.01)`` and
    ``activity_regularizer=L2(0.01)`` on every inception Conv2D when
    batchnorm=False.  In PyTorch weight decay is typically applied
    **per-parameter-group via the optimizer** (e.g. ``Adam(..., weight_decay=1e-2)``).
    Pass a higher ``weight_decay`` to the optimizer when ``batchnorm=False``
    to replicate the TF behaviour.
    """
    # 中文教学注释：
    # 该函数对应 TF 的 conv2d_bn，用于 Inception 各分支内部。
    # bias=False 是关键：启用 BN 时偏置会被 BN 的平移参数抵消，属冗余；
    # 不启用 BN 时也保持 bias=False 是为了与 TF 复刻口径一致（L2 正则
    # 由优化器 weight_decay 承担，而不是由层内部实现）。
    # 参数含义：
    #   in_channels/out_channels —— 输入/输出通道数；
    #   kernel_size/stride/padding —— 卷积超参数，padding 传 "same" 之类
    #     字符串时直接透传给 PyTorch（PyTorch 支持 stride=1 的 "same"）；
    #   batchnorm —— 是否在卷积后插入 BatchNorm2d。
    # 返回：nn.Sequential(Conv2d → [BatchNorm2d] → ReLU)。
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False)
    ]
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _conv2d_bn_stem(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] | str = 0,
    batchnorm: bool = False,
) -> nn.Sequential:
    """TF Conv2D_BN → Conv (+ optional BN) + ReLU.  No L2 in either case."""
    # 中文教学注释：
    # 对应 TF 的 Conv2D_BN，专用于 stem（网络最前面的三层卷积）。
    # 与 _conv2d_bn_block 的区别仅在 TF 原版的正则策略：stem 两种情况
    # 都不加 L2。PyTorch 实现上两者代码形状相同，拆成两个函数是为了
    # 与 TF 源码一一对照、便于核对复刻是否忠实。
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False)
    ]
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# ============================================================================
# 1D building blocks  —  faithful adaptation of the same TF helpers
# ============================================================================
# 本节是上面两个 2D 助手的一维版本：Conv2d→Conv1d、BatchNorm2d→BatchNorm1d，
# 其余（bias=False、可选 BN、ReLU、L2 交由优化器）完全沿用同一套约定。
# DSCARNet1D 直接用 1D 光谱曲线训练时就走这里的积木。

def _conv1d_bn_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    batchnorm: bool = False,
) -> nn.Sequential:
    """1D equivalent of TF conv2d_bn."""
    # 中文教学注释：_conv2d_bn_block 的 1D 对应物，用于 1D Inception 分支。
    layers: list[nn.Module] = [
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False)
    ]
    if batchnorm:
        layers.append(nn.BatchNorm1d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _conv1d_bn_stem(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    batchnorm: bool = False,
) -> nn.Sequential:
    """1D equivalent of TF Conv2D_BN."""
    # 中文教学注释：_conv2d_bn_stem 的 1D 对应物，用于 1D stem 三层卷积。
    layers: list[nn.Module] = [
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False)
    ]
    if batchnorm:
        layers.append(nn.BatchNorm1d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# ============================================================================
# 2D Inception block
# ============================================================================

class InceptionBlock2D(nn.Module):
    """
    One DSCAR inception unit (2D).

    TF reference::

        unit = 48 * (2 ** i)
        branch1x1   = conv2d_bn(x, unit,       1,1)
        branch5x5   = conv2d_bn(x, 0.5*unit,   1,1) → conv2d_bn(unit, 5,5)
        branch3x3dbl= conv2d_bn(x, unit,       1,1) → conv2d_bn(2*unit,3,3)
                                                     → conv2d_bn(2*unit,3,3)
        branch_pool = AvgPool(3,s=1,same) → conv2d_bn(unit, 1,1)
        x = concat([1x1, 5x5, 3x3dbl, pool])
    """

    def __init__(self, in_channels: int, unit: int, batchnorm: bool = False):
        super().__init__()
        # 中文教学注释：
        # 参数：
        #   in_channels —— 输入特征图通道数；
        #   unit        —— 本块的基准通道数，TF 公式为 48×2^i（i 为块序号），
        #                  四个分支的输出通道都围绕 unit 定义；
        #   batchnorm   —— 透传给每个分支的卷积块。
        # half_unit：5×5 分支先用 1×1 卷积把通道压到 0.5×unit（bottleneck），
        # 再做 5×5 卷积——这是 Inception 的经典降本设计，大核卷积的计算量
        # 随通道数平方增长，先压缩通道可显著省算力；max(1, ...) 防止
        # unit 很小时通道数被压成 0。
        half_unit = max(1, int(unit * 0.5))

        # 分支 1：1×1 卷积，直接提取逐点跨通道特征，输出 unit 通道。
        self.branch1x1 = _conv2d_bn_block(in_channels, unit, 1, batchnorm=batchnorm)

        # 分支 2：1×1 降通道（half_unit）→ 5×5 卷积（恢复 unit），
        # padding=2 对应 TF 的 'same'（k=5 时 same padding = 2），
        # 负责捕获较大感受野的空间模式。
        self.branch5x5_reduce = _conv2d_bn_block(in_channels, half_unit, 1, batchnorm=batchnorm)
        self.branch5x5 = _conv2d_bn_block(half_unit, unit, 5, padding=2, batchnorm=batchnorm)

        # 分支 3：1×1 → 3×3 → 3×3 双卷积串联，输出 2×unit 通道。
        # 两个 3×3 串联的感受野等效一个 5×5，但参数更少、非线性更多；
        # padding=1 对应 3×3 的 'same'。
        self.branch3x3dbl_reduce = _conv2d_bn_block(in_channels, unit, 1, batchnorm=batchnorm)
        self.branch3x3dbl_a = _conv2d_bn_block(unit, unit * 2, 3, padding=1, batchnorm=batchnorm)
        self.branch3x3dbl_b = _conv2d_bn_block(unit * 2, unit * 2, 3, padding=1, batchnorm=batchnorm)

        # 分支 4：3×3 平均池化（stride=1、same，不改变空间尺寸）→ 1×1 卷积，
        # 给网络提供一条"平滑/低频"通路，AggMap 热图的局部平滑结构由此保留。
        self.branch_pool_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_conv = _conv2d_bn_block(in_channels, unit, 1, batchnorm=batchnorm)

        # 输出通道数 = unit + unit + 2×unit + unit = 5×unit，
        # 供外层堆叠下一个 block 时推算 in_channels。
        self.out_channels = unit * 5  # unit + unit + 2*unit + unit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 中文教学注释：
        # 四条分支并行计算（注意分支 2、3 是先 reduce 再大核卷积），
        # 最后沿通道维（dim=1，NCHW 中 C 维）拼接。所有分支的空间尺寸
        # 都通过 same padding 保持一致，因此可以直接 cat。
        b1 = self.branch1x1(x)
        b5 = self.branch5x5_reduce(x)
        b5 = self.branch5x5(b5)
        b3 = self.branch3x3dbl_reduce(x)
        b3 = self.branch3x3dbl_a(b3)
        b3 = self.branch3x3dbl_b(b3)
        bp = self.branch_pool_pool(x)
        bp = self.branch_pool_conv(bp)
        return torch.cat([b1, b5, b3, bp], dim=1)


# ============================================================================
# 1D Inception block  —  faithful translation of the 2D version
# ============================================================================

class InceptionBlock1D(nn.Module):
    """
    1D equivalent of the DSCAR inception unit.  Same topology, channel formula,
    and 0.5× bottleneck as the TF 2D original.

    Mapping from 2D → 1D:
      Conv2D(k=1×1)      → Conv1d(k=1)
      Conv2D(k=5×5,same) → Conv1d(k=5,padding=2)
      Conv2D(k=3×3,same) → Conv1d(k=3,padding=1)
      AvgPool2D(3,s=1,same) → AvgPool1d(3,s=1,padding=1)
    """

    def __init__(self, in_channels: int, unit: int, batchnorm: bool = False):
        super().__init__()
        # 中文教学注释：
        # 与 InceptionBlock2D 完全相同的拓扑和通道公式，只是把每个算子
        # 换成 1D 版本，作用于 (B, C, L) 的光谱序列：多尺度分支分别捕获
        # 窄峰（1×1/3×3）与宽峰/包络（5×5、pool）形态。
        half_unit = max(1, int(unit * 0.5))

        self.branch1x1 = _conv1d_bn_block(in_channels, unit, 1, batchnorm=batchnorm)

        self.branch5x5_reduce = _conv1d_bn_block(in_channels, half_unit, 1, batchnorm=batchnorm)
        self.branch5x5 = _conv1d_bn_block(half_unit, unit, 5, padding=2, batchnorm=batchnorm)

        self.branch3x3dbl_reduce = _conv1d_bn_block(in_channels, unit, 1, batchnorm=batchnorm)
        self.branch3x3dbl_a = _conv1d_bn_block(unit, unit * 2, 3, padding=1, batchnorm=batchnorm)
        self.branch3x3dbl_b = _conv1d_bn_block(unit * 2, unit * 2, 3, padding=1, batchnorm=batchnorm)

        self.branch_pool_pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_conv = _conv1d_bn_block(in_channels, unit, 1, batchnorm=batchnorm)

        self.out_channels = unit * 5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 中文教学注释：与 2D 版一致——四分支并行后沿通道维拼接。
        b1 = self.branch1x1(x)
        b5 = self.branch5x5_reduce(x)
        b5 = self.branch5x5(b5)
        b3 = self.branch3x3dbl_reduce(x)
        b3 = self.branch3x3dbl_a(b3)
        b3 = self.branch3x3dbl_b(b3)
        bp = self.branch_pool_pool(x)
        bp = self.branch_pool_conv(bp)
        return torch.cat([b1, b5, b3, bp], dim=1)


# ============================================================================
# 2D Stem
# ============================================================================

class DSCARStem2D(nn.Module):
    """
    TF stem::

        x = Conv2D_BN(inp, 64, 19, 19, 'same', 1)
        if h > 25:
            x = Conv2D_BN(x, 96, 5, 5, 'valid', 2)
            x = Conv2D_BN(x, 128, 5, 5, 'valid', 1)
        else:
            x = Conv2D_BN(x, 96, 5, 5, 'valid', 1)
            x = Conv2D_BN(x, 128, 5, 5, 'same', 1)
        x = MaxPool2D(3, 2, 'same')
    """

    def __init__(
        self,
        input_height: int,
        input_channels: int = 1,
        conv1_kernel_size: int = 19,
        filter_number: int = 64,
        batchnorm: bool = False,
    ):
        super().__init__()
        # 中文教学注释：
        # 参数：
        #   input_height     —— 输入特征图高度（AggMap 图的边长），
        #                      决定 stem 走"大图分支"还是"小图分支"；
        #   input_channels   —— 输入通道（SAR/CAR 单通道图为 1）；
        #   conv1_kernel_size—— 首层大核尺寸，默认 19，大核负责在浅层
        #                      直接捕获全局相关性结构；
        #   filter_number    —— 首层通道数 f1（默认 64），f2/f3 由 f1
        #                      按公式推导，保持 64→96→128 的比例关系；
        #   batchnorm        —— 是否每层后接 BN。
        # padding = k//2 即 TF 的 'same'（奇数核时成立）。
        padding = int(conv1_kernel_size) // 2
        f1 = int(filter_number)
        # f2 ≈ 1.5×f1（且至少 f1+1），f3 = 2×f1（且至少 f2+1）——
        # 用 max 是为了在 filter_number 极小的情况下仍保证通道严格递增。
        f2 = max(f1 + 1, int(round(f1 * 1.5)))
        f3 = max(f2 + 1, f1 * 2)
        self.out_channels = f3
        self.conv1 = _conv2d_bn_stem(
            input_channels,
            f1,
            kernel_size=conv1_kernel_size,
            padding=padding,
            batchnorm=batchnorm,
        )

        # 大图（h>25）：第二层用 stride=2 + 'valid' 主动下采样，
        # 第三层 'valid' 继续压缩边界；小图（h≤25）：两层都 stride=1，
        # 第三层改 'same' 避免空间尺寸被卷没了——这是 TF 原版针对
        # 不同 AggMap 尺寸的条件结构，不能统一。
        if input_height > 25:
            self.conv2 = _conv2d_bn_stem(f1, f2, kernel_size=5, stride=2, padding=0, batchnorm=batchnorm)
            self.conv3 = _conv2d_bn_stem(f2, f3, kernel_size=5, stride=1, padding=0, batchnorm=batchnorm)
        else:
            self.conv2 = _conv2d_bn_stem(f1, f2, kernel_size=5, stride=1, padding=0, batchnorm=batchnorm)
            self.conv3 = _conv2d_bn_stem(f2, f3, kernel_size=5, stride=1, padding=2, batchnorm=batchnorm)

        # 末尾 3×3、stride=2 的 MaxPool（same）再做一次下采样，
        # 输出通道数为 f3（默认 128）。
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 中文教学注释：三层卷积 + 一次池化，顺序与 TF stem 完全一致。
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        return x


# ============================================================================
# 1D Stem  —  faithful translation of the 2D stem
# ============================================================================

class DSCARStem1D(nn.Module):
    """
    1D equivalent of the TF DSCAR stem.

    Mapping from 2D → 1D:
      Conv2D(64, 19×19, same, s=1) → Conv1d(1→64, k=19, same, s=1)
      Conv2D(96, 5×5, valid, s=2) → Conv1d(64→96, k=5, valid, s=2)
      Conv2D(128, 5×5, valid, s=1) → Conv1d(96→128, k=5, valid, s=1)
      Conv2D(128, 5×5, same, s=1) → Conv1d(96→128, k=5, same, s=1)
      MaxPool2D(3, s=2, same)      → MaxPool1d(3, s=2, padding=1)
    """

    def __init__(self, input_length: int, batchnorm: bool = False):
        super().__init__()
        # 中文教学注释：
        # 参数：
        #   input_length —— 1D 光谱曲线长度（wide-feature-v2 宽表的特征列数），
        #                   用与 2D 版相同的 >25 阈值选择 stride/padding 分支，
        #                   防止短序列被 'valid' 卷积耗尽长度；
        #   batchnorm    —— 是否每层后接 BN。
        # 首层 k=19、padding=9 即 'same'，对应 2D 版的 19×19 大核首层。
        self.conv1 = _conv1d_bn_stem(1, 64, kernel_size=19, padding=9, batchnorm=batchnorm)

        if input_length > 25:
            self.conv2 = _conv1d_bn_stem(64, 96, kernel_size=5, stride=2, padding=0, batchnorm=batchnorm)
            self.conv3 = _conv1d_bn_stem(96, 128, kernel_size=5, stride=1, padding=0, batchnorm=batchnorm)
        else:
            self.conv2 = _conv1d_bn_stem(64, 96, kernel_size=5, stride=1, padding=0, batchnorm=batchnorm)
            self.conv3 = _conv1d_bn_stem(96, 128, kernel_size=5, stride=1, padding=2, batchnorm=batchnorm)

        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 中文教学注释：与 2D stem 同序——三卷积 + 一池化。
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        return x


# ============================================================================
# SingleDSCARNet2D  —  1:1 PyTorch mirror of TF single_dscarnet
# ============================================================================

def _parse_2d_input_shape(input_shape: tuple[int, ...]) -> tuple[int, int, int]:
    # 中文教学注释：
    # 兼容两种 input_shape 写法：(H, W) 默认单通道，(H, W, C) 显式给通道。
    # 注意返回顺序是 (h, w, c)，而网络前向期望 PyTorch 的 NCHW 输入——
    # 调用方需自行保证数据布局，本函数只做形状解析不做转置。
    # 其他长度直接 ValueError，避免静默误解形状。
    if len(input_shape) == 2:
        h, w = input_shape
        return int(h), int(w), 1
    if len(input_shape) == 3:
        h, w, c = input_shape
        return int(h), int(w), int(c)
    raise ValueError("input_shape must be (height, width) or (height, width, channels)")


class SingleDSCARNet2D(nn.Module):
    """
    Exact PyTorch equivalent of TF ``single_dscarnet``.

    Parameters
    ----------
    input_shape : tuple[int, int] | tuple[int, int, int]
        (height, width) or (height, width, channels), expects PyTorch NCHW.
    n_outputs : int
        Number of classes.
    conv1_kernel_size : int
        Kernel size of the very first convolution (default 19).
    n_inception : int
        Number of sequential Inception blocks.
    dense_layers : Sequence[int]
        Hidden units before the final classifier (default [128]).
    dense_avf : str
        Activation for hidden dense layers ('relu' or 'gelu').
    batchnorm : bool
        Whether to insert BatchNorm after every convolution.
    last_avf : str or None
        Final activation.  ``'softmax'`` or ``None`` (for CrossEntropyLoss).
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        n_outputs: int = 2,
        conv1_kernel_size: int = 19,
        filter_number: int = 64,
        n_inception: int = 1,
        dense_layers: Sequence[int] = (128,),
        dense_avf: str = "relu",
        batchnorm: bool = False,
        last_avf: str | None = "softmax",
    ):
        super().__init__()
        # 中文教学注释：
        # 整体结构 = stem → n_inception 个 InceptionBlock2D 串联
        #   → AdaptiveMaxPool 到 1×1 → flatten → 可配置 Dense 头。
        # 这是平台可解释性流程中"SAR/CAR 单视图"路径使用的 2D 网络。
        # 宽度 _w 未使用：stem 的分支选择只看高度 h，宽度由卷积自适应。
        h, _w, input_channels = _parse_2d_input_shape(input_shape)

        self.stem = DSCARStem2D(
            input_height=h,
            input_channels=input_channels,
            conv1_kernel_size=conv1_kernel_size,
            filter_number=filter_number,
            batchnorm=batchnorm,
        )

        # 逐块堆叠 Inception：unit 按 TF 公式 48×2^i 增长（这里写成
        # filter_number×0.75 的等价形式，默认 64×0.75=48），
        # max(4, ...) 防止 filter_number 过小时 unit 退化。
        # 每块的 out_channels（=5×unit）作为下一块的 in_channels。
        in_channels = self.stem.out_channels
        blocks: list[InceptionBlock2D] = []
        for i in range(n_inception):
            unit = max(4, int(round(filter_number * 0.75))) * (2 ** i)
            block = InceptionBlock2D(in_channels, unit=unit, batchnorm=batchnorm)
            blocks.append(block)
            in_channels = block.out_channels
        self.inception = nn.Sequential(*blocks)

        # 全局最大池化到 (1,1)：对应 TF 的 GlobalMaxPool2D，
        # 用 Adaptive 版本是为了对任意输入空间尺寸都成立。
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Dense 头：按 dense_layers 依次堆 Linear+激活，最后接
        # Linear(→n_outputs) 分类层；n_outputs 即类别数（仅分类任务）。
        dense_activation = _get_activation(dense_avf)
        dense_modules: list[nn.Module] = []
        dense_input = in_channels
        for units in dense_layers:
            dense_modules.append(nn.Linear(dense_input, units))
            dense_modules.append(dense_activation)
            dense_input = units
        dense_modules.append(nn.Linear(dense_input, n_outputs))

        self.last_avf = last_avf
        self.classifier = nn.Sequential(*dense_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 中文教学注释：
        # 前向顺序：stem → inception → 全局池化 → flatten → dense 头。
        # last_avf='softmax' 时输出概率（与 TF 行为一致）；置 None 时
        # 输出 raw logits，供 PyTorch 的 CrossEntropyLoss 直接使用
        # （CrossEntropyLoss 内部已含 log_softmax，不能再加 softmax）。
        x = self.stem(x)
        x = self.inception(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.last_avf == "softmax":
            x = F.softmax(x, dim=1)
        return x


# ============================================================================
# DSCARNet1D  —  faithful 1D adaptation of the original 2D DSCARNet
# ============================================================================

class DSCARNet1D(nn.Module):
    """
    Faithful 1D adaptation of the original DSCARNet architecture.

    This preserves the exact same architectural decisions as the TF 2D version:
    - 3-layer stem:  1→64 (k=19, same) → 96 (k=5, valid) → 128 (k=5)
      with conditional stride when input_length > 25
    - Inception blocks:  unit = 48 × 2^i,  branch5×5 uses 0.5× unit bottleneck
    - GlobalMaxPool → Dense(128, relu) → Dense(n_classes)

    Constructor signature kept compatible with the old registry call site::

        DSCARNet1D(input_length, class_count, dropout, hidden_size, inception_blocks)

    Parameters
    ----------
    input_length : int
        Length of 1D spectral curves.  Used to select the stem branch (>25 or not).
    class_count : int
        Number of output classes.
    dropout : float | None
        **Accepted for backward compatibility but not used.**

        The original TF DSCARNet does not use Dropout — it uses L2 weight decay
        (when batchnorm=False) or BatchNorm (when batchnorm=True).  Pass
        ``weight_decay`` to the optimizer to replicate the L2 behaviour.
    hidden_size : int
        **Accepted for backward compatibility but not used.**

        The original TF DSCARNet uses a fixed channel progression (64→96→128 in
        stem, 48×2^i in inception, 128 in the hidden dense layer).
    inception_blocks : int
        Number of sequential Inception blocks (default 1).  Maps directly to
        TF ``n_inception``.
    batchnorm : bool
        Whether to insert BatchNorm after every convolution (default False,
        matching TF default).
    """

    def __init__(
        self,
        input_length: int,
        class_count: int,
        dropout: float | None = None,
        hidden_size: int = 32,
        inception_blocks: int = 1,
        batchnorm: bool = False,
    ):
        super().__init__()
        # 中文教学注释：
        # 这是平台模型注册表中 "dscarnet" 直接 1D 输入路径使用的类，
        # 构造签名与旧注册表调用点保持位置参数兼容，因此 dropout 和
        # hidden_size 必须保留在签名里；但 TF 原版没有 Dropout、通道数
        # 也是固定公式，所以这里显式丢弃（赋给 _ 并注释说明），
        # 调用方若想要正则化，应通过优化器 weight_decay 或 batchnorm=True。
        _ = dropout      # not used — TF version has no Dropout
        _ = hidden_size  # not used — TF version uses fixed channel sizes

        self.stem = DSCARStem1D(input_length=input_length, batchnorm=batchnorm)

        # stem 输出固定 128 通道；Inception 块数至少为 1（max(1, ...) 防止
        # 注册表传入 0 或负数导致网络没有特征提取主体），unit 严格按
        # TF 公式 48×2^i 增长。
        in_channels = 128
        blocks: list[InceptionBlock1D] = []
        for i in range(max(1, int(inception_blocks))):
            unit = 48 * (2 ** i)
            block = InceptionBlock1D(in_channels, unit=unit, batchnorm=batchnorm)
            blocks.append(block)
            in_channels = block.out_channels
        self.inception = nn.Sequential(*blocks)

        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # Hidden dense layer: 128 units with ReLU, matching the TF default
        # 中文教学注释：分类头固定为 Flatten → Linear(→128) → ReLU
        # → Linear(→class_count)，输出 raw logits 供 CrossEntropyLoss 使用。
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, L)  tensor.

        Returns:
            (B, class_count) logits.
        """
        # 中文教学注释：输入是建模流程张量化的单通道光谱 (B, 1, L)，
        # 依次过 stem → inception → 全局池化 → 分类头，返回 logits。
        x = self.stem(x)
        x = self.inception(x)
        x = self.global_pool(x)
        return self.classifier(x)


# ============================================================================
# DualDSCARNet2D  —  two-branch version (SAR + CAR)
# ============================================================================

class DualDSCARNet2D(nn.Module):
    """
    Exact PyTorch equivalent of TF ``dual_dscarnet``.

    Two independent stems + inception stacks (one per 2D view), whose
    GlobalMaxPool features are concatenated before a shared dense head.
    """

    def __init__(
        self,
        input_shape1: tuple[int, ...],
        input_shape2: tuple[int, ...],
        n_outputs: int = 2,
        conv1_kernel_size: int = 19,
        filter_number: int = 64,
        n_inception: int = 1,
        dense_layers: Sequence[int] = (128,),
        dense_avf: str = "relu",
        batchnorm: bool = False,
        last_avf: str | None = "softmax",
    ):
        super().__init__()

        # 中文教学注释：
        # 双通路版本：input_shape1 / input_shape2 分别是 SAR 与 CAR 两个
        # AggMap 视图的形状。两条支路各自拥有独立的 stem + inception
        # （参数不共享——SAR/CAR 的统计特性不同，共享会互相干扰），
        # 最后把两路 GlobalMaxPool 特征拼接后进共享 dense 头融合。
        # 这也是 DSCARNet 可解释性"双通路 2D Grad-CAM"的结构基础。

        # ---- 支路 1（SAR 视图）----
        h1, _w1, input_channels1 = _parse_2d_input_shape(input_shape1)
        self.stem1 = DSCARStem2D(
            input_height=h1,
            input_channels=input_channels1,
            conv1_kernel_size=conv1_kernel_size,
            filter_number=filter_number,
            batchnorm=batchnorm,
        )
        ch1 = self.stem1.out_channels
        blocks1: list[InceptionBlock2D] = []
        for i in range(n_inception):
            unit = max(4, int(round(filter_number * 0.75))) * (2 ** i)
            block = InceptionBlock2D(ch1, unit=unit, batchnorm=batchnorm)
            blocks1.append(block)
            ch1 = block.out_channels
        self.inception1 = nn.Sequential(*blocks1)
        self.pool1 = nn.AdaptiveMaxPool2d((1, 1))

        # ---- 支路 2（CAR 视图），结构与支路 1 对称 ----
        h2, _w2, input_channels2 = _parse_2d_input_shape(input_shape2)
        self.stem2 = DSCARStem2D(
            input_height=h2,
            input_channels=input_channels2,
            conv1_kernel_size=conv1_kernel_size,
            filter_number=filter_number,
            batchnorm=batchnorm,
        )
        ch2 = self.stem2.out_channels
        blocks2: list[InceptionBlock2D] = []
        for i in range(n_inception):
            unit = max(4, int(round(filter_number * 0.75))) * (2 ** i)
            block = InceptionBlock2D(ch2, unit=unit, batchnorm=batchnorm)
            blocks2.append(block)
            ch2 = block.out_channels
        self.inception2 = nn.Sequential(*blocks2)
        self.pool2 = nn.AdaptiveMaxPool2d((1, 1))

        # 融合头：输入维度 = 两路特征通道之和 ch1 + ch2。
        dense_activation = _get_activation(dense_avf)
        dense_modules: list[nn.Module] = []
        dense_input = ch1 + ch2
        for units in dense_layers:
            dense_modules.append(nn.Linear(dense_input, units))
            dense_modules.append(dense_activation)
            dense_input = units
        dense_modules.append(nn.Linear(dense_input, n_outputs))

        self.last_avf = last_avf
        self.classifier = nn.Sequential(*dense_modules)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # 中文教学注释：
        # x1、x2 分别是 SAR / CAR 两个 2D 视图（NCHW）。两路独立提取
        # 特征、各自全局池化并 flatten 后拼接，再进共享 dense 头；
        # softmax 语义与 SingleDSCARNet2D 相同（配 CrossEntropyLoss 时
        # 应把 last_avf 置 None）。
        f1 = self.pool1(self.inception1(self.stem1(x1)))
        f1 = f1.view(f1.size(0), -1)
        f2 = self.pool2(self.inception2(self.stem2(x2)))
        f2 = f2.view(f2.size(0), -1)
        x = torch.cat([f1, f2], dim=1)
        x = self.classifier(x)
        if self.last_avf == "softmax":
            x = F.softmax(x, dim=1)
        return x


# ============================================================================
# Helpers
# ============================================================================

def _get_activation(name: str) -> nn.Module:
    # 中文教学注释：
    # 把字符串激活名映射为模块，仅支持 TF 原版用到的 'relu' / 'gelu'；
    # 其他名字直接 ValueError，避免静默退回某个默认激活而改变网络行为。
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


# ============================================================================
# Factory helpers  (matching TF signatures)
# ============================================================================
# 下面两个工厂函数保持与 TF 原版 single_dscarnet / dual_dscarnet 相同的
# 参数签名，方便按 TF 论文/源码的调用方式直接构造 PyTorch 模型。

def single_dscarnet(
    input_shape: tuple[int, ...],
    filter_number: int = 64,
    n_outputs: int = 2,
    conv1_kernel_size: int = 19,
    n_inception: int = 1,
    dense_layers: Sequence[int] = (128,),
    dense_avf: str = "relu",
    batchnorm: bool = False,
    last_avf: str | None = "softmax",
) -> SingleDSCARNet2D:
    """Drop-in constructor matching TF ``single_dscarnet`` signature."""
    # 中文教学注释：纯透传工厂——按 TF 签名收参并构造 SingleDSCARNet2D，
    # 不做任何额外默认值改写，保证"TF 怎么调、这里就怎么调"。
    return SingleDSCARNet2D(
        input_shape=input_shape,
        n_outputs=n_outputs,
        conv1_kernel_size=conv1_kernel_size,
        filter_number=filter_number,
        n_inception=n_inception,
        dense_layers=dense_layers,
        dense_avf=dense_avf,
        batchnorm=batchnorm,
        last_avf=last_avf,
    )


def dual_dscarnet(
    input_shape1: tuple[int, ...],
    input_shape2: tuple[int, ...],
    filter_number: int = 64,
    n_outputs: int = 2,
    conv1_kernel_size: int = 19,
    n_inception: int = 1,
    dense_layers: Sequence[int] = (128,),
    dense_avf: str = "relu",
    batchnorm: bool = False,
    last_avf: str | None = "softmax",
) -> DualDSCARNet2D:
    """Drop-in constructor matching TF ``dual_dscarnet`` signature."""
    # 中文教学注释：与 single_dscarnet 同理，构造双通路 DualDSCARNet2D。
    return DualDSCARNet2D(
        input_shape1=input_shape1,
        input_shape2=input_shape2,
        n_outputs=n_outputs,
        conv1_kernel_size=conv1_kernel_size,
        filter_number=filter_number,
        n_inception=n_inception,
        dense_layers=dense_layers,
        dense_avf=dense_avf,
        batchnorm=batchnorm,
        last_avf=last_avf,
    )

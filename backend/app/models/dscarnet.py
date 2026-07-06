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

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 2D building blocks  —  matching TF conv2d_bn / Conv2D_BN helpers
# ============================================================================

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

def _conv1d_bn_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    batchnorm: bool = False,
) -> nn.Sequential:
    """1D equivalent of TF conv2d_bn."""
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
        half_unit = max(1, int(unit * 0.5))

        self.branch1x1 = _conv2d_bn_block(in_channels, unit, 1, batchnorm=batchnorm)

        self.branch5x5_reduce = _conv2d_bn_block(in_channels, half_unit, 1, batchnorm=batchnorm)
        self.branch5x5 = _conv2d_bn_block(half_unit, unit, 5, padding=2, batchnorm=batchnorm)

        self.branch3x3dbl_reduce = _conv2d_bn_block(in_channels, unit, 1, batchnorm=batchnorm)
        self.branch3x3dbl_a = _conv2d_bn_block(unit, unit * 2, 3, padding=1, batchnorm=batchnorm)
        self.branch3x3dbl_b = _conv2d_bn_block(unit * 2, unit * 2, 3, padding=1, batchnorm=batchnorm)

        self.branch_pool_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_conv = _conv2d_bn_block(in_channels, unit, 1, batchnorm=batchnorm)

        self.out_channels = unit * 5  # unit + unit + 2*unit + unit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        batchnorm: bool = False,
    ):
        super().__init__()
        padding = int(conv1_kernel_size) // 2
        self.conv1 = _conv2d_bn_stem(
            input_channels,
            64,
            kernel_size=conv1_kernel_size,
            padding=padding,
            batchnorm=batchnorm,
        )

        if input_height > 25:
            self.conv2 = _conv2d_bn_stem(64, 96, kernel_size=5, stride=2, padding=0, batchnorm=batchnorm)
            self.conv3 = _conv2d_bn_stem(96, 128, kernel_size=5, stride=1, padding=0, batchnorm=batchnorm)
        else:
            self.conv2 = _conv2d_bn_stem(64, 96, kernel_size=5, stride=1, padding=0, batchnorm=batchnorm)
            self.conv3 = _conv2d_bn_stem(96, 128, kernel_size=5, stride=1, padding=2, batchnorm=batchnorm)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        self.conv1 = _conv1d_bn_stem(1, 64, kernel_size=19, padding=9, batchnorm=batchnorm)

        if input_length > 25:
            self.conv2 = _conv1d_bn_stem(64, 96, kernel_size=5, stride=2, padding=0, batchnorm=batchnorm)
            self.conv3 = _conv1d_bn_stem(96, 128, kernel_size=5, stride=1, padding=0, batchnorm=batchnorm)
        else:
            self.conv2 = _conv1d_bn_stem(64, 96, kernel_size=5, stride=1, padding=0, batchnorm=batchnorm)
            self.conv3 = _conv1d_bn_stem(96, 128, kernel_size=5, stride=1, padding=2, batchnorm=batchnorm)

        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        return x


# ============================================================================
# SingleDSCARNet2D  —  1:1 PyTorch mirror of TF single_dscarnet
# ============================================================================

def _parse_2d_input_shape(input_shape: tuple[int, ...]) -> tuple[int, int, int]:
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
        n_inception: int = 1,
        dense_layers: Sequence[int] = (128,),
        dense_avf: str = "relu",
        batchnorm: bool = False,
        last_avf: str | None = "softmax",
    ):
        super().__init__()
        h, _w, input_channels = _parse_2d_input_shape(input_shape)

        self.stem = DSCARStem2D(
            input_height=h,
            input_channels=input_channels,
            conv1_kernel_size=conv1_kernel_size,
            batchnorm=batchnorm,
        )

        in_channels = 128
        blocks: list[InceptionBlock2D] = []
        for i in range(n_inception):
            unit = 48 * (2 ** i)
            block = InceptionBlock2D(in_channels, unit=unit, batchnorm=batchnorm)
            blocks.append(block)
            in_channels = block.out_channels
        self.inception = nn.Sequential(*blocks)

        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))

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
        _ = dropout      # not used — TF version has no Dropout
        _ = hidden_size  # not used — TF version uses fixed channel sizes

        self.stem = DSCARStem1D(input_length=input_length, batchnorm=batchnorm)

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
        n_inception: int = 1,
        dense_layers: Sequence[int] = (128,),
        dense_avf: str = "relu",
        batchnorm: bool = False,
        last_avf: str | None = "softmax",
    ):
        super().__init__()

        h1, _w1, input_channels1 = _parse_2d_input_shape(input_shape1)
        self.stem1 = DSCARStem2D(
            input_height=h1,
            input_channels=input_channels1,
            conv1_kernel_size=conv1_kernel_size,
            batchnorm=batchnorm,
        )
        ch1 = 128
        blocks1: list[InceptionBlock2D] = []
        for i in range(n_inception):
            unit = 48 * (2 ** i)
            block = InceptionBlock2D(ch1, unit=unit, batchnorm=batchnorm)
            blocks1.append(block)
            ch1 = block.out_channels
        self.inception1 = nn.Sequential(*blocks1)
        self.pool1 = nn.AdaptiveMaxPool2d((1, 1))

        h2, _w2, input_channels2 = _parse_2d_input_shape(input_shape2)
        self.stem2 = DSCARStem2D(
            input_height=h2,
            input_channels=input_channels2,
            conv1_kernel_size=conv1_kernel_size,
            batchnorm=batchnorm,
        )
        ch2 = 128
        blocks2: list[InceptionBlock2D] = []
        for i in range(n_inception):
            unit = 48 * (2 ** i)
            block = InceptionBlock2D(ch2, unit=unit, batchnorm=batchnorm)
            blocks2.append(block)
            ch2 = block.out_channels
        self.inception2 = nn.Sequential(*blocks2)
        self.pool2 = nn.AdaptiveMaxPool2d((1, 1))

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
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


# ============================================================================
# Factory helpers  (matching TF signatures)
# ============================================================================

def single_dscarnet(
    input_shape: tuple[int, ...],
    n_outputs: int = 2,
    conv1_kernel_size: int = 19,
    n_inception: int = 1,
    dense_layers: Sequence[int] = (128,),
    dense_avf: str = "relu",
    batchnorm: bool = False,
    last_avf: str | None = "softmax",
) -> SingleDSCARNet2D:
    """Drop-in constructor matching TF ``single_dscarnet`` signature."""
    return SingleDSCARNet2D(
        input_shape=input_shape,
        n_outputs=n_outputs,
        conv1_kernel_size=conv1_kernel_size,
        n_inception=n_inception,
        dense_layers=dense_layers,
        dense_avf=dense_avf,
        batchnorm=batchnorm,
        last_avf=last_avf,
    )


def dual_dscarnet(
    input_shape1: tuple[int, ...],
    input_shape2: tuple[int, ...],
    n_outputs: int = 2,
    conv1_kernel_size: int = 19,
    n_inception: int = 1,
    dense_layers: Sequence[int] = (128,),
    dense_avf: str = "relu",
    batchnorm: bool = False,
    last_avf: str | None = "softmax",
) -> DualDSCARNet2D:
    """Drop-in constructor matching TF ``dual_dscarnet`` signature."""
    return DualDSCARNet2D(
        input_shape1=input_shape1,
        input_shape2=input_shape2,
        n_outputs=n_outputs,
        conv1_kernel_size=conv1_kernel_size,
        n_inception=n_inception,
        dense_layers=dense_layers,
        dense_avf=dense_avf,
        batchnorm=batchnorm,
        last_avf=last_avf,
    )

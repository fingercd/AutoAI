"""CNN-Transformer classifier for long one-dimensional signals.

The module is intentionally independent from the model registry.  Training
profiles are resolved by the caller and passed to the constructor explicitly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _resolve_class_count(class_count: int | None, num_classes: int | None) -> int:
    if class_count is None:
        class_count = num_classes
    elif num_classes is not None and int(class_count) != int(num_classes):
        raise ValueError("class_count 与 num_classes 必须一致")
    if class_count is None or int(class_count) <= 0:
        raise ValueError("class_count 必须是正整数")
    return int(class_count)


def _resolve_heads(d_model: int, requested_heads: int) -> int:
    heads = max(1, int(requested_heads))
    while heads > 1 and d_model % heads != 0:
        heads -= 1
    return heads


def _ceil_pool_length(length: int, pools: Sequence[int]) -> int:
    for pool in pools:
        length = math.ceil(length / int(pool))
    return max(1, length)


class CNNTransformer1D(nn.Module):
    """Three-stage CNN front-end followed by a batch-first Transformer.

    Parameters are deliberately explicit so a resolved model profile can be
    passed in by a future shared builder without this module importing or
    recomputing profile policy.
    """

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
            channels = (
                max(8, requested_d_model // 4),
                max(16, requested_d_model // 2),
                requested_d_model,
            )
        else:
            channels = tuple(int(value) for value in conv_channels)
            if len(channels) != 3 or any(value <= 0 for value in channels):
                raise ValueError("conv_channels 必须包含三个正整数")
            if d_model is None:
                requested_d_model = channels[-1]

        self.d_model = requested_d_model
        self.transformer_heads = _resolve_heads(self.d_model, transformer_heads)
        self.transformer_layers = int(transformer_layers)
        if channels[-1] != self.d_model:
            channels = (*channels[:2], self.d_model)
        self.conv_channels = channels
        if conv_kernels is None or pool_sizes is None:
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
        in_channels = 1
        for out_channels, kernel_size, pool_size in zip(channels, kernels, pools):
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

        token_length = _ceil_pool_length(self.input_length, pools)
        self.token_length = token_length
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
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("CNNTransformer1D 输入必须是 (batch, 1, length)")
        sequence = self.cnn(x).transpose(1, 2)
        sequence = sequence + self._position_for(sequence.shape[1])
        encoded = self.encoder(sequence)
        pooled = encoded.mean(dim=1)
        return self.head(self.dropout(pooled))


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

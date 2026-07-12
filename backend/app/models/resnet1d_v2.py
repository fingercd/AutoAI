"""Document-aligned 1D ResNet classifier.

This module is deliberately versioned next to the legacy implementation.  It
does not import the registry or the shared profile factory, so it can be
promoted by a later integration task without changing the legacy state-dict
contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


N_CHANNELS: dict[str, tuple[int, int, int]] = {
    "small": (8, 16, 32),
    "medium": (16, 32, 64),
    "large": (32, 64, 128),
}

L_PROFILE: dict[str, dict[str, Any]] = {
    "short": {
        "stem_kernel": 7,
        "stem_pool": 2,
        "block_kernel": 3,
        "block_pools": (2, 2, 2),
    },
    "medium": {
        "stem_kernel": 9,
        "stem_pool": 4,
        "block_kernel": 3,
        "block_pools": (2, 2, 2),
    },
    "long": {
        "stem_kernel": 9,
        "stem_pool": 4,
        "block_kernel": 5,
        "block_pools": (2, 2, 2),
    },
}


def _band_for_sample_count(sample_count: int) -> str:
    return "small" if sample_count <= 100 else ("medium" if sample_count <= 300 else "large")


def _band_for_feature_count(feature_count: int) -> str:
    return "short" if feature_count <= 1000 else ("medium" if feature_count <= 3000 else "long")


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


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _resolve_class_count(class_count: int | None, num_classes: int | None) -> int:
    if class_count is None:
        class_count = num_classes
    elif num_classes is not None and int(class_count) != int(num_classes):
        raise ValueError("class_count 与 num_classes 必须一致")
    if class_count is None or int(class_count) <= 0:
        raise ValueError("class_count 必须是正整数")
    return int(class_count)


def _resolve_dropout(value: float | None) -> float:
    dropout = 0.25 if value is None else float(value)
    if not 0.0 <= dropout <= 1.0:
        raise ValueError("dropout 必须位于 [0, 1]")
    return dropout


def _same_length_pair(first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop only the extra boundary produced by an even document kernel."""

    length = min(first.shape[-1], second.shape[-1])
    return first[..., :length], second[..., :length]


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
            else nn.Sequential(
                nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(self.out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)
        self.pool = (
            nn.Identity()
            if self.pool_size == 1
            else nn.MaxPool1d(kernel_size=self.pool_size, stride=self.pool_size, ceil_mode=True)
        )
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        main = self.main_path(x)
        main, residual = _same_length_pair(main, residual)
        result = self.activation(main + residual)
        result = self.dropout(result)
        return self.pool(result)


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

        channels_value = values.get("channels", N_CHANNELS[sample_band])
        channels = _as_int_tuple(channels_value, name="channels", length=3)
        kernels_value = values.get("kernels")
        stem_kernel = int(values.get("stem_kernel", values.get("K0", profile_l_values["stem_kernel"])))
        block_kernel = int(values.get("block_kernel", values.get("kernel", profile_l_values["block_kernel"])))
        if kernels_value is not None:
            kernels = _as_int_tuple(kernels_value, name="kernels")
            stem_kernel = kernels[0]
            block_kernel = kernels[-1] if len(kernels) > 1 else kernels[0]
        pools_value = values.get("pools")
        stem_pool = int(values.get("stem_pool", values.get("P0", profile_l_values["stem_pool"])))
        block_pools = tuple(int(item) for item in profile_l_values["block_pools"])
        if pools_value is not None:
            pools = _as_int_tuple(pools_value, name="pools")
            if len(pools) == 4:
                stem_pool, block_pools = pools[0], pools[1:]
            elif len(pools) == 3:
                block_pools = pools
                stem_pool = pools[0]
            elif len(pools) == 1:
                stem_pool = block_pools = (pools[0],) * 3
            else:
                raise ValueError("pools 必须包含 1、3 或 4 个整数")
        block_pools = _as_int_tuple(block_pools, name="block_pools", length=3)
        dropout_value = _resolve_dropout(dropout if dropout is not None else profile_dropout)
        if hidden_size is not None:
            # Hidden size is a legacy knob.  It may lower the first width, but
            # never changes the document profile's three-stage topology.
            requested = max(1, int(hidden_size))
            scale = max(1, min(requested, channels[0]))
            channels = tuple(max(scale, channel) for channel in channels)

        self.sample_count = resolved_n
        self.sample_band = sample_band
        self.feature_band = feature_band
        self.channels = tuple(channels)
        self.kernels = (stem_kernel, block_kernel)
        self.pool_sizes = (stem_pool, *block_pools)
        self.dropout_value = dropout_value

        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=stem_kernel, padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.stem_pool = nn.MaxPool1d(kernel_size=stem_pool, stride=stem_pool, ceil_mode=True)
        blocks: list[ResidualBlock1DDocumentV2] = []
        in_channels = channels[0]
        for out_channels, pool_size in zip(channels, block_pools):
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
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout_value),
            nn.Linear(in_channels, self.class_count),
        )
        self.dropout = self.head[2]

    @property
    def blocks(self) -> nn.ModuleList:
        return self.residual_blocks

    @property
    def features(self) -> nn.ModuleList:
        return self.residual_blocks

    @property
    def pools(self) -> tuple[nn.Module, ...]:
        return (self.stem_pool, *(block.pool for block in self.residual_blocks))

    @property
    def classifier(self) -> nn.Sequential:
        return self.head

    def gradcam_target_layer(self) -> nn.Conv1d:
        """Return the final residual main-path convolution for 1D Grad-CAM."""

        return self.residual_blocks[-1].conv2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("ResNet1DDocumentV2 输入必须是 (batch, 1, length)")
        x = self.stem_pool(self.stem(x))
        for block in self.residual_blocks:
            x = block(x)
        return self.head(x)


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

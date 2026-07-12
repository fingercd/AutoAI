"""Document-aligned temporal convolutional classifier."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


N_CHANNELS: dict[str, int] = {"small": 16, "medium": 32, "large": 64}

L_PROFILE: dict[str, dict[str, Any]] = {
    "short": {"stem_kernel": 7, "stem_pool": 2, "dilations": (1, 2, 4)},
    "medium": {"stem_kernel": 9, "stem_pool": 4, "dilations": (1, 2, 4, 8)},
    "long": {"stem_kernel": 9, "stem_pool": 4, "dilations": (1, 2, 4, 8)},
}


def _sample_band(sample_count: int) -> str:
    return "small" if sample_count <= 100 else ("medium" if sample_count <= 300 else "large")


def _feature_band(feature_count: int) -> str:
    return "short" if feature_count <= 1000 else ("medium" if feature_count <= 3000 else "long")


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _profile_values(
    profile: Mapping[str, Any] | Any | None,
) -> tuple[dict[str, Any], int | None, int | None, float | None]:
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
    if class_count is None:
        class_count = num_classes
    elif num_classes is not None and int(class_count) != int(num_classes):
        raise ValueError("class_count 与 num_classes 必须一致")
    if class_count is None or int(class_count) <= 0:
        raise ValueError("class_count 必须是正整数")
    return int(class_count)


def _same_length_pair(first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length = min(first.shape[-1], second.shape[-1])
    return first[..., :length], second[..., :length]


class TCNBlock1DDocumentV2(nn.Module):
    """Causal-compatible same-length residual block with two dilated convs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dilation: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.dilation = int(dilation)
        self.kernel_size = int(kernel_size)
        if self.dilation <= 0 or self.kernel_size <= 0:
            raise ValueError("dilation 和 kernel_size 必须是正整数")
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
        self.net = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True), self.conv2, self.bn2)
        self.shortcut = (
            nn.Identity()
            if self.in_channels == self.out_channels
            else nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, bias=False)
        )
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.net(x)
        residual = self.shortcut(x)
        main, residual = _same_length_pair(main, residual)
        return self.activation(self.dropout(main + residual))


class TCN1DDocumentV2(nn.Module):
    """Lightweight TCN with only the document dilation schedule."""

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

        if isinstance(sample_count, float) and 0.0 <= sample_count <= 1.0:
            if dropout is None:
                dropout = float(sample_count)
            sample_count = None
        resolved_n = int(train_sample_count or sample_count or profile_n or 100)
        if resolved_n <= 1:
            raise ValueError("sample_count 必须大于 1")
        resolved_l = int(profile_l or self.input_length)
        sample_band = _sample_band(resolved_n)
        feature_band = _feature_band(resolved_l)
        l_values = dict(L_PROFILE[feature_band])
        channels = int(values.get("channels", N_CHANNELS[sample_band]))
        if isinstance(values.get("channels"), (tuple, list)):
            channel_values = tuple(int(item) for item in values["channels"])
            channels = channel_values[-1]
        if hidden_size is not None:
            channels = max(1, min(channels, int(hidden_size)))
        stem_kernel = int(values.get("stem_kernel", values.get("K0", l_values["stem_kernel"])))
        pool_size = int(values.get("stem_pool", values.get("pool", values.get("P0", l_values["stem_pool"]))))
        kernel_size = int(values.get("kernel", values.get("block_kernel", 3)))
        dilations_value = values.get("dilations", l_values["dilations"])
        dilations = tuple(int(dilation) for dilation in dilations_value)
        if dilations not in {(1, 2, 4), (1, 2, 4, 8)}:
            raise ValueError("TCN v2 只允许 dilation [1,2,4] 或 [1,2,4,8]")
        if channels <= 0 or stem_kernel <= 0 or pool_size <= 0:
            raise ValueError("channels、stem_kernel、pool_size 必须是正整数")
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
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=stem_kernel, padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, ceil_mode=True)
        self.tcn_blocks = nn.ModuleList(
            [
                TCNBlock1DDocumentV2(
                    channels,
                    channels,
                    dilation=dilation,
                    kernel_size=kernel_size,
                )
                for dilation in dilations
            ]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout_value),
            nn.Linear(channels, self.class_count),
        )
        self.dropout = self.head[2]

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

        return self.tcn_blocks[-1].conv2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("TCN1DDocumentV2 输入必须是 (batch, 1, length)")
        x = self.pool(self.stem(x))
        for block in self.tcn_blocks:
            x = block(x)
        return self.head(x)


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

"""Document-aligned four-branch 1D Inception classifier."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


N_BRANCH_CHANNELS: dict[str, int] = {"small": 8, "medium": 16, "large": 32}

L_PROFILE: dict[str, dict[str, Any]] = {
    "short": {"stem_kernel": 7, "stem_pool": 2, "branch_kernels": (1, 3, 5, 7)},
    "medium": {"stem_kernel": 9, "stem_pool": 4, "branch_kernels": (3, 5, 7, 9)},
    "long": {"stem_kernel": 9, "stem_pool": 4, "branch_kernels": (5, 7, 9, 11)},
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


def _same_length(values: list[torch.Tensor]) -> list[torch.Tensor]:
    length = min(value.shape[-1] for value in values)
    return [value[..., :length] for value in values]


class InceptionBlock1DDocumentV2(nn.Module):
    """Four equal-width convolution-only branches.

    The old implementation includes a pooling branch and a bottleneck.  The
    v2 block intentionally does neither: every branch is a direct Conv1d
    with the same output width, followed by its own normalization/activation.
    """

    def __init__(
        self,
        in_channels: int,
        branch_channels: int,
        kernels: tuple[int, int, int, int] = (1, 3, 5, 7),
    ) -> None:
        super().__init__()
        if int(branch_channels) <= 0:
            raise ValueError("branch_channels 必须是正整数")
        if len(kernels) != 4 or any(int(kernel) <= 0 for kernel in kernels):
            raise ValueError("Inception 必须提供四个正整数 kernel")
        self.in_channels = int(in_channels)
        self.branch_channels = int(branch_channels)
        self.kernels = tuple(int(kernel) for kernel in kernels)
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
        self.branch_norms = nn.ModuleList([nn.BatchNorm1d(self.branch_channels) for _ in self.branches])
        self.activation = nn.ReLU(inplace=True)
        self.out_channels = self.branch_channels * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [self.activation(norm(conv(x))) for conv, norm in zip(self.branches, self.branch_norms)]
        return torch.cat(_same_length(branches), dim=1)


class Inception1DDocumentV2(nn.Module):
    """Single-channel classifier with a document stem and four-branch block."""

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

        branch_channels_value = values.get("branch_channels", values.get("channels", N_BRANCH_CHANNELS[sample_band]))
        if isinstance(branch_channels_value, (tuple, list)):
            branch_channels_value = branch_channels_value[-1]
        branch_channels = int(branch_channels_value)
        stem_kernel = int(values.get("stem_kernel", values.get("K0", l_values["stem_kernel"])))
        kernels_value = values.get("branch_kernels", values.get("kernels", l_values["branch_kernels"]))
        kernels = tuple(int(kernel) for kernel in kernels_value)
        if len(kernels) != 4 or any(kernel <= 0 for kernel in kernels):
            raise ValueError("branch_kernels 必须包含四个正整数")
        pool_size = int(values.get("stem_pool", values.get("pool", values.get("P0", l_values["stem_pool"]))))
        if pool_size <= 0:
            raise ValueError("pool_size 必须是正整数")
        dropout_value = 0.25 if dropout is None and profile_dropout is None else float(
            dropout if dropout is not None else profile_dropout
        )
        if not 0.0 <= dropout_value <= 1.0:
            raise ValueError("dropout 必须位于 [0, 1]")
        if hidden_size is not None:
            branch_channels = max(1, min(branch_channels, int(hidden_size)))
        block_count = int(values.get("inception_blocks", inception_blocks))
        if block_count <= 0:
            raise ValueError("inception_blocks 必须是正整数")

        self.sample_count = resolved_n
        self.sample_band = sample_band
        self.feature_band = feature_band
        self.branch_channels = branch_channels
        self.channels = (branch_channels,) * 4
        self.kernels = (stem_kernel, *kernels)
        self.pool_sizes = (pool_size,)
        self.dropout_value = dropout_value
        self.stem = nn.Sequential(
            nn.Conv1d(1, branch_channels, kernel_size=stem_kernel, padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, ceil_mode=True),
        )
        blocks: list[InceptionBlock1DDocumentV2] = []
        in_channels = branch_channels
        for _ in range(block_count):
            block = InceptionBlock1DDocumentV2(in_channels, branch_channels, kernels=kernels)
            blocks.append(block)
            in_channels = block.out_channels
        self.inception_blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout_value),
            nn.Linear(in_channels, self.class_count),
        )
        self.dropout = self.head[2]

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
        return (self.stem[3],)

    @property
    def classifier(self) -> nn.Sequential:
        return self.head

    def gradcam_target_layer(self) -> InceptionBlock1DDocumentV2:
        """Return the final concatenating block for 1D Grad-CAM."""

        return self.inception_blocks[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("Inception1DDocumentV2 输入必须是 (batch, 1, length)")
        x = self.stem(x)
        for block in self.inception_blocks:
            x = block(x)
        return self.head(x)


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

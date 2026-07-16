"""在文档版 1D CNN 上加入 squeeze-and-excitation 通道门控。"""

from __future__ import annotations

import torch
from torch import nn

from .cnn1d import (
    CNN1DProfile,
    CNNProfile,
    _DocumentCNN1D,
    build_cnn_profile,
    resolve_cnn_profile,
)


class SEBlock1D(nn.Module):
    """Squeeze-and-excitation gate for one-dimensional convolution features."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        channels = int(channels)
        reduction = int(reduction)
        if channels <= 0 or reduction <= 0:
            raise ValueError("SEBlock1D 的 channels/reduction 必须为正整数")
        hidden = max(channels // reduction, 4)
        self.channels = channels
        self.reduction = reduction
        self.hidden_channels = hidden
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(f"SEBlock1D 需要 Batch×{self.channels}×Length 输入")
        weights = self.pool(x).flatten(1)
        weights = self.gate(self.fc(weights)).unsqueeze(-1)
        return x * weights


class CNNSE1DDocumentV2(_DocumentCNN1D):
    """Document-exact three-block CNN with one SE gate per convolution block."""

    def __init__(
        self,
        input_length: int | None = None,
        class_count: int | None = None,
        sample_count: int | None = None,
        dropout: float | None = None,
        hidden_size: int = 64,
        *,
        profile: object | None = None,
        model_profile: object | None = None,
        N: int | None = None,
        L: int | None = None,
        classes: int | None = None,
        num_classes: int | None = None,
        n_classes: int | None = None,
        n_samples: int | None = None,
    ) -> None:
        super().__init__(
            input_length,
            class_count,
            sample_count,
            dropout,
            hidden_size,
            profile=profile,
            model_profile=model_profile,
            N=N,
            L=L,
            classes=classes,
            num_classes=num_classes,
            n_classes=n_classes,
            n_samples=n_samples,
            use_se=True,
        )

    def _make_se_block(self, channels: int) -> nn.Module:
        return SEBlock1D(channels)


CNNSE1D = CNNSE1DDocumentV2
CNNSE1DV2 = CNNSE1DDocumentV2
CNNSE1D_V2 = CNNSE1DDocumentV2


__all__ = [
    "CNN1DProfile",
    "CNNProfile",
    "SEBlock1D",
    "CNNSE1DDocumentV2",
    "CNNSE1D",
    "CNNSE1DV2",
    "CNNSE1D_V2",
    "resolve_cnn_profile",
    "build_cnn_profile",
]

from __future__ import annotations

import torch
from torch import nn


class TCNBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dilation: int, kernel_size: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x) + self.skip(x))


class TCN1D(nn.Module):
    def __init__(self, input_length: int, class_count: int, dropout: float | None = None, hidden_size: int = 32) -> None:
        super().__init__()
        dropout = 0.25 if dropout is None else float(dropout)
        if input_length <= 3000:
            dilations = [1, 2, 4, 8]
            channels = 32
            stem_stride = 1
        elif input_length <= 6000:
            dilations = [1, 2, 4, 8, 16]
            channels = 48
            stem_stride = 2
        else:
            dilations = [1, 2, 4, 8, 16, 32]
            channels = 48
            stem_stride = 4
        channels = max(16, min(max(int(hidden_size), channels), 64))
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=5, stride=stem_stride, padding=2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        blocks = [TCNBlock1D(channels, channels, dilation=dilation, dropout=dropout) for dilation in dilations]
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(channels, class_count))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(self.stem(x)))

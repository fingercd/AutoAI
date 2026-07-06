from __future__ import annotations

import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, dilation: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm1d(out_channels))
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x) + self.skip(x))


class ResNet1D(nn.Module):
    def __init__(self, input_length: int, class_count: int, dropout: float | None = None, hidden_size: int = 32) -> None:
        super().__init__()
        dropout = 0.25 if dropout is None else float(dropout)
        if input_length <= 3000:
            channels = [32, 64]
            strides = [1, 2]
            dilations = [1, 1]
            stem_stride = 1
        elif input_length <= 6000:
            channels = [32, 64, 96]
            strides = [1, 2, 2]
            dilations = [1, 1, 1]
            stem_stride = 2
        else:
            channels = [32, 64, 96]
            strides = [1, 2, 1]
            dilations = [1, 2, 4]
            stem_stride = 4
        base = max(16, min(int(hidden_size), channels[0]))
        channels = [max(base, ch) for ch in channels]
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=7, stride=stem_stride, padding=3, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )
        blocks: list[nn.Module] = []
        in_channels = channels[0]
        for out_channels, stride, dilation in zip(channels, strides, dilations):
            blocks.append(ResidualBlock1D(in_channels, out_channels, stride=stride, dilation=dilation, dropout=dropout))
            blocks.append(ResidualBlock1D(out_channels, out_channels, stride=1, dilation=dilation, dropout=dropout))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(in_channels, class_count))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        return self.head(x)

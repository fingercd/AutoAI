from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    def __init__(self, input_length: int, class_count: int, sample_count: int, dropout: float | None = None, hidden_size: int = 64) -> None:
        super().__init__()
        if input_length <= 3000:
            channels = [1, 16, 32, 64]
            strides = [1, 1, 1]
            kernels = [7, 5, 3]
        elif input_length <= 6000:
            channels = [1, 24, 48, 64, 96]
            strides = [2, 1, 1, 1]
            kernels = [7, 5, 5, 3]
        else:
            channels = [1, 32, 64, 96]
            strides = [4, 1, 1]
            kernels = [9, 5, 3]
        dropout = dropout if dropout is not None else (0.45 if sample_count < 100 else 0.25)
        layers: list[nn.Module] = []
        for idx, (in_channels, out_channels) in enumerate(zip(channels, channels[1:])):
            kernel_size = kernels[idx]
            stride = strides[idx]
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(2, ceil_mode=True),
                    nn.Dropout(dropout),
                ]
            )
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Linear(channels[-1], class_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).squeeze(-1)
        return self.classifier(x)

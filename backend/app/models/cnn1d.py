from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    def __init__(self, input_length: int, class_count: int, sample_count: int, dropout: float | None = None, hidden_size: int = 64) -> None:
        super().__init__()
        if input_length < 500:
            channels = [1, 16, 32]
        elif input_length <= 3000:
            channels = [1, 24, 48, 64]
        else:
            channels = [1, 32, 64, 96, 128]
        dropout = dropout if dropout is not None else (0.45 if sample_count < 100 else 0.25)
        layers: list[nn.Module] = []
        for in_channels, out_channels in zip(channels, channels[1:]):
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout),
                ]
            )
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Linear(channels[-1], class_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).squeeze(-1)
        return self.classifier(x)

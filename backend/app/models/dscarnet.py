from __future__ import annotations

import torch
from torch import nn


class Inception1DBlock(nn.Module):
    def __init__(self, channels: int, branch_channels: int, dropout: float) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv1d(channels, branch_channels, kernel_size=1), nn.BatchNorm1d(branch_channels), nn.ReLU())
        self.branch5 = nn.Sequential(
            nn.Conv1d(channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(),
        )
        self.branch3dbl = nn.Sequential(
            nn.Conv1d(channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(),
            nn.Conv1d(branch_channels, branch_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_channels * 2),
            nn.ReLU(),
            nn.Conv1d(branch_channels * 2, branch_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_channels * 2),
            nn.ReLU(),
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.out_channels = branch_channels * 5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.branch1(x), self.branch5(x), self.branch3dbl(x), self.branch_pool(x)], dim=1)
        return self.dropout(x)


class DSCARNet1D(nn.Module):
    """A PyTorch 1D adaptation of the reference DSCARNet inception-style classifier."""

    def __init__(
        self,
        input_length: int,
        class_count: int,
        dropout: float | None = None,
        hidden_size: int = 32,
        inception_blocks: int = 1,
    ) -> None:
        super().__init__()
        dropout = 0.15 if dropout is None else dropout
        base_channels = max(16, int(hidden_size))
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=19, padding=9),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )

        blocks = []
        channels = base_channels * 2
        for idx in range(max(1, int(inception_blocks))):
            branch_channels = max(8, base_channels // 2) * (2**idx)
            block = Inception1DBlock(channels, branch_channels, dropout)
            blocks.append(block)
            channels = block.out_channels
        self.inception = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, max(base_channels * 2, 32)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(base_channels * 2, 32), class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.inception(self.stem(x)))

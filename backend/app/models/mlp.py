from __future__ import annotations

import torch
from torch import nn


class MLPBaseline(nn.Module):
    def __init__(self, input_length: int, class_count: int, dropout: float | None = None, hidden_size: int = 128) -> None:
        super().__init__()
        dropout = 0.35 if dropout is None else dropout
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_length, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max(hidden_size // 2, 16)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_size // 2, 16), class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

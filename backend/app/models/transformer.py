from __future__ import annotations

import torch
import numpy as np
from torch import nn


class Transformer1D(nn.Module):
    def __init__(
        self,
        input_length: int,
        class_count: int,
        dropout: float | None = None,
        hidden_size: int = 64,
        heads: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        dropout = 0.25 if dropout is None else dropout
        if input_length <= 3000:
            patch_size, stride, model_dim, layers = 32, 16, 48, min(int(layers), 2)
        elif input_length <= 6000:
            patch_size, stride, model_dim, layers = 64, 32, 64, 2
        else:
            patch_size, stride, model_dim, layers = 96, 48, max(64, min(int(hidden_size), 96)), 2
        hidden_size = max(model_dim, int(hidden_size))
        heads = max(1, heads)
        while hidden_size % heads != 0 and heads > 1:
            heads -= 1
        self.patch = nn.Conv1d(1, hidden_size, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        max_tokens = int(np.ceil((input_length + patch_size) / max(1, stride))) + 1
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(hidden_size, class_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.patch(x).transpose(1, 2)
        seq = seq + self.pos[:, : seq.shape[1], :]
        seq = self.encoder(seq)
        return self.head(seq.mean(dim=1))

from __future__ import annotations

import torch
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
        heads = max(1, heads)
        while hidden_size % heads != 0 and heads > 1:
            heads -= 1
        self.proj = nn.Linear(1, hidden_size)
        self.pos = nn.Parameter(torch.zeros(1, input_length, hidden_size))
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
        seq = x.transpose(1, 2)
        seq = self.proj(seq) + self.pos[:, : seq.shape[1], :]
        seq = self.encoder(seq)
        return self.head(seq.mean(dim=1))

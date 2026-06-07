from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet1D(nn.Module):
    def __init__(self, input_length: int, class_count: int, dropout: float | None = None, hidden_size: int = 16, depth: int = 3) -> None:
        super().__init__()
        dropout = 0.15 if dropout is None else dropout
        depth = max(2, min(int(depth), 5))
        base_channels = max(8, int(hidden_size))
        channels = [base_channels * (2**idx) for idx in range(depth)]

        self.downs = nn.ModuleList()
        prev_channels = 1
        for channels_out in channels:
            self.downs.append(ConvBlock1D(prev_channels, channels_out, dropout))
            prev_channels = channels_out
        self.pools = nn.ModuleList([nn.MaxPool1d(2) for _ in range(depth - 1)])

        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for idx in range(depth - 2, -1, -1):
            self.ups.append(nn.ConvTranspose1d(channels[idx + 1], channels[idx], kernel_size=2, stride=2))
            self.up_blocks.append(ConvBlock1D(channels[idx] * 2, channels[idx], dropout))

        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(channels[0], class_count))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        out = x
        for idx, down in enumerate(self.downs):
            out = down(out)
            skips.append(out)
            if idx < len(self.pools):
                out = self.pools[idx](out)

        for up, block, skip in zip(self.ups, self.up_blocks, reversed(skips[:-1])):
            out = up(out)
            length_delta = skip.size(-1) - out.size(-1)
            if length_delta != 0:
                out = F.pad(out, (0, max(0, length_delta)))[..., : skip.size(-1)]
            out = block(torch.cat([skip, out], dim=1))
        return self.head(out)

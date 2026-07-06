from __future__ import annotations

import torch
from torch import nn


class InceptionModule1D(nn.Module):
    def __init__(self, in_channels: int, filters: int, kernels: tuple[int, int, int], bottleneck: int = 16, stride: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.reduce = nn.Conv1d(in_channels, bottleneck, kernel_size=1, bias=False) if in_channels > 1 else nn.Identity()
        reduced_channels = bottleneck if in_channels > 1 else in_channels
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(reduced_channels, filters, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
                for kernel in kernels
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=stride, padding=1),
            nn.Conv1d(in_channels, filters, kernel_size=1, bias=False),
        )
        self.bn = nn.BatchNorm1d(filters * 4)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.out_channels = filters * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        branches = [branch(reduced) for branch in self.branches]
        branches.append(self.pool_branch(x))
        min_len = min(item.shape[-1] for item in branches)
        out = torch.cat([item[..., :min_len] for item in branches], dim=1)
        return self.dropout(self.activation(self.bn(out)))


class Inception1D(nn.Module):
    def __init__(self, input_length: int, class_count: int, dropout: float | None = None, hidden_size: int = 32) -> None:
        super().__init__()
        dropout = 0.25 if dropout is None else float(dropout)
        filters = max(16, min(int(hidden_size), 32))
        if input_length <= 3000:
            module_count = 3
            kernels = (10, 20, 40)
            first_stride = 1
        elif input_length <= 6000:
            module_count = 4
            kernels = (16, 32, 64)
            first_stride = 2
        else:
            module_count = 4
            kernels = (20, 40, 80)
            first_stride = 4
        modules: list[nn.Module] = []
        in_channels = 1
        for idx in range(module_count):
            module = InceptionModule1D(
                in_channels,
                filters=filters,
                kernels=kernels,
                bottleneck=16,
                stride=first_stride if idx == 0 else 1,
                dropout=dropout,
            )
            modules.append(module)
            in_channels = module.out_channels
        self.features = nn.Sequential(*modules)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(in_channels, class_count))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

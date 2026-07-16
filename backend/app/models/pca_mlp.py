"""把训练折拟合的 PCA 固化为首层投影的 MLP 分类器。

PCA 均值与成分注册为 buffer，因此随 model.pt 保存但不参与梯度更新。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class PCAMLPClassifier(nn.Module):
    """先执行固定训练折 PCA 投影，再通过两层 MLP 输出 logits。"""
    def __init__(
        self,
        pca_mean,
        pca_components,
        hidden_sizes: Sequence[int],
        dropout: float,
        class_count: int,
    ) -> None:
        super().__init__()
        mean = torch.as_tensor(pca_mean, dtype=torch.float32).reshape(-1)
        components = torch.as_tensor(pca_components, dtype=torch.float32)
        if components.ndim != 2 or components.shape[1] != mean.shape[0]:
            raise ValueError("PCA 均值与成分矩阵的原始特征轴不一致")
        hidden = [max(1, int(value)) for value in hidden_sizes]
        if len(hidden) != 2:
            raise ValueError("PCA-MLP 必须提供两个 hidden size")
        self.register_buffer("pca_mean", mean)
        self.register_buffer("pca_components", components)
        self.classifier = nn.Sequential(
            nn.Linear(components.shape[0], hidden[0]),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden[1], int(class_count)),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        x = values.squeeze(1)
        x = (x - self.pca_mean) @ self.pca_components.T
        return self.classifier(x)

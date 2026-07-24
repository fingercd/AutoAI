"""把训练折拟合的 PCA 固化为首层投影的 MLP 分类器。

PCA 均值与成分注册为 buffer，因此随 model.pt 保存但不参与梯度更新。

模块职责
--------
本模块定义 ``PCAMLPClassifier``，对应能力目录中的 ``pca_mlp`` 深度模型：
先用"训练折上拟合好的 PCA"把原始高维光谱/色谱特征投影到低维主成分空间，
再过一个两层 MLP 输出分类 logits。

在系统中的位置
--------------
- 上游：``backend/app/services/training.py`` 在每个训练折内拟合 PCA
  （严格只用当前训练集，防止信息泄漏到验证/测试折），把 ``pca_mean`` 与
  ``pca_components`` 交给本类构造模型。
- 下游：模型实例由通用深度训练循环训练，按 ``model.pt`` 保存；可解释性
  分析走窗口遮挡 Log-loss 增量口径（pca_mlp 无卷积结构，不用 Grad-CAM）。

关键设计约束
------------
- PCA 参数不是可学习权重：标准化/降维必须只由训练折拟合，若把它们做成
  ``nn.Parameter`` 参与梯度更新，就破坏了"预处理只由训练集拟合"的评估纪律，
  因此注册为 buffer —— 既随 checkpoint 保存/迁移设备，又不接收梯度。
- 前向输入沿用深度模型统一的 ``(batch, 1, n_features)`` 形状约定，
  forward 内部先 ``squeeze(1)`` 还原为二维矩阵再做线性投影。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class PCAMLPClassifier(nn.Module):
    """先执行固定训练折 PCA 投影，再通过两层 MLP 输出 logits。

    网络结构：PCA 线性投影（不可学习 buffer）→ Linear → ReLU → Dropout
    → Linear → ReLU → Dropout → Linear(类别数)。输出为未归一化 logits，
    交由训练循环的 CrossEntropyLoss 处理（不要在模型内加 softmax）。
    """
    def __init__(
        self,
        pca_mean,
        pca_components,
        hidden_sizes: Sequence[int],
        dropout: float,
        class_count: int,
    ) -> None:
        super().__init__()
        # 参数说明：
        #   pca_mean / pca_components —— 训练折上拟合出的 PCA 均值向量与
        #     成分矩阵（形状分别为 (n_features,) 与 (n_components, n_features)，
        #     即 sklearn PCA.components_ 的布局），构造时被固化进 buffer。
        #   hidden_sizes —— 两个隐藏层的宽度。
        #   dropout —— 每个隐藏层后的 Dropout 概率。
        #   class_count —— 类别数，决定输出层宽度。
        # 统一转成 float32 张量，避免 numpy float64 与模型权重精度不一致；
        # reshape(-1) 容许调用方传入 (1, n) 或 (n,) 形状的均值。
        mean = torch.as_tensor(pca_mean, dtype=torch.float32).reshape(-1)
        components = torch.as_tensor(pca_components, dtype=torch.float32)
        # 构造期就做形状校验：成分矩阵必须是二维，且其列数（原始特征轴）
        # 与均值长度一致，否则投影矩阵乘法会在训练中途才报错，难以定位。
        if components.ndim != 2 or components.shape[1] != mean.shape[0]:
            raise ValueError("PCA 均值与成分矩阵的原始特征轴不一致")
        # 隐藏层宽度强制为正整数（防御 0/负数/浮点配置），并固定要求两层：
        # 结构变了会让旧 checkpoint 与训练服务的搜索空间约定失效。
        hidden = [max(1, int(value)) for value in hidden_sizes]
        if len(hidden) != 2:
            raise ValueError("PCA-MLP 必须提供两个 hidden size")
        # 注册为 buffer 而非 Parameter：随 state_dict 保存进 model.pt、
        # 随 .to(device) 迁移，但 requires_grad=False，训练时保持不变，
        # 保证"PCA 只由训练折拟合"这一评估纪律不被梯度更新破坏。
        self.register_buffer("pca_mean", mean)
        self.register_buffer("pca_components", components)
        # 分类头输入维度 = 主成分个数（成分矩阵的行数）。
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
        """前向：去掉通道维 → 固定 PCA 投影 → MLP 分类头。

        参数 ``values`` 形状为 ``(batch, 1, n_features)``（深度模型统一
        输入约定，中间的 1 是留给卷积类模型的通道维）；返回形状为
        ``(batch, class_count)`` 的 logits。
        """
        x = values.squeeze(1)
        # 手写 PCA 变换：(x - mean) @ components.T，与 sklearn 的
        # PCA.transform 等价，但全程在 torch 内完成，可随模型一起搬上 GPU。
        x = (x - self.pca_mean) @ self.pca_components.T
        return self.classifier(x)

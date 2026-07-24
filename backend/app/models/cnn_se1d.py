"""在文档版 1D CNN 上加入 squeeze-and-excitation 通道门控。"""

from __future__ import annotations

# ======================================================================
# 模块级说明
#
# 本文件实现能力目录中的 `cnn1d_se`：在 cnn1d.py 的文档版三段式 1D CNN
# 每个卷积块里插入一个 SEBlock1D 通道门控（Squeeze-and-Excitation）。
#
# 在系统中的位置：
#   - 位于 backend/app/models/，与 `cnn1d` 共用同一套 N/L profile 解析逻辑
#     （resolve_cnn_profile / CNN1DProfile 均从 .cnn1d 导入），因此两者的
#     通道数、卷积核、池化和 dropout 口径完全一致，唯一差别是 SE 门控。
#   - 训练编排层实例化 `CNNSE1DDocumentV2`；可解释性走与 cnn1d 相同的
#     1D Grad-CAM 路线（gradcam_target_layer 继承自基类，仍指向末段卷积）。
#
# 设计意图：
#   - SE 门控让网络按通道自适应地放大/抑制特征图：对光谱任务而言，
#     相当于让模型学习“哪些卷积通道（响应哪些谱峰模式）更值得关注”。
#   - 通过继承 `_DocumentCNN1D` 并仅覆盖 `_make_se_block`，保证与文档版
#     结构严格对齐，不引入第二份架构代码。
# ======================================================================

import torch
from torch import nn

from .cnn1d import (
    CNN1DProfile,
    CNNProfile,
    _DocumentCNN1D,
    build_cnn_profile,
    resolve_cnn_profile,
)


# 一维版 squeeze-and-excitation 通道门控。
#
# 原理（SE-Net 的 1D 移植）：
#   1) Squeeze：对长度维做全局平均池化，把每个通道压成一个标量，
#      得到“该通道在整个曲线上的平均响应强度”；
#   2) Excitation：两层全连接（先降维再升维）+ Sigmoid，学出每个通道
#      在 (0, 1) 区间的重要性权重；
#   3) Scale：把权重逐通道乘回特征图，实现通道级重标定。
#
# 瓶颈层宽度 hidden = max(channels // reduction, 4)：reduction=8 控制
# 门控的参数量开销，同时设下限 4 防止小通道数时瓶颈过窄学不动。
class SEBlock1D(nn.Module):
    """Squeeze-and-excitation gate for one-dimensional convolution features."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        channels = int(channels)
        reduction = int(reduction)
        # 非正整数属于调用方配置错误，直接拒绝而不是默默造出非法层
        if channels <= 0 or reduction <= 0:
            raise ValueError("SEBlock1D 的 channels/reduction 必须为正整数")
        hidden = max(channels // reduction, 4)
        self.channels = channels
        self.reduction = reduction
        self.hidden_channels = hidden
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        self.gate = nn.Sigmoid()

    # 前向：输入必须是 Batch×channels×Length，且通道数与构造时一致，
    # 否则说明被接错了位置，立即报错。
    # 权重形状为 Batch×channels×1，靠广播逐通道乘回特征图。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(f"SEBlock1D 需要 Batch×{self.channels}×Length 输入")
        weights = self.pool(x).flatten(1)
        weights = self.gate(self.fc(weights)).unsqueeze(-1)
        return x * weights


# 带 SE 门控的文档版三段 CNN：能力目录中的 `cnn1d_se`。
# 唯一差异是 use_se=True 并实现 `_make_se_block`，让基类在每个卷积块的
# ReLU 之后、MaxPool 之前插入 SEBlock1D；其余（profile 解析、Grad-CAM
# 目标层、分类头）全部继承，保证与 `cnn1d` 的口径一致。
class CNNSE1DDocumentV2(_DocumentCNN1D):
    """Document-exact three-block CNN with one SE gate per convolution block."""

    def __init__(
        self,
        input_length: int | None = None,
        class_count: int | None = None,
        sample_count: int | None = None,
        dropout: float | None = None,
        hidden_size: int = 64,
        *,
        profile: object | None = None,
        model_profile: object | None = None,
        N: int | None = None,
        L: int | None = None,
        classes: int | None = None,
        num_classes: int | None = None,
        n_classes: int | None = None,
        n_samples: int | None = None,
    ) -> None:
        super().__init__(
            input_length,
            class_count,
            sample_count,
            dropout,
            hidden_size,
            profile=profile,
            model_profile=model_profile,
            N=N,
            L=L,
            classes=classes,
            num_classes=num_classes,
            n_classes=n_classes,
            n_samples=n_samples,
            use_se=True,
        )

    # 基类在 use_se=True 时通过该钩子创建每个块的 SE 门控
    def _make_se_block(self, channels: int) -> nn.Module:
        return SEBlock1D(channels)


# 对外别名：兼容历史调用方的不同命名
CNNSE1D = CNNSE1DDocumentV2
CNNSE1DV2 = CNNSE1DDocumentV2
CNNSE1D_V2 = CNNSE1DDocumentV2


__all__ = [
    "CNN1DProfile",
    "CNNProfile",
    "SEBlock1D",
    "CNNSE1DDocumentV2",
    "CNNSE1D",
    "CNNSE1DV2",
    "CNNSE1D_V2",
    "resolve_cnn_profile",
    "build_cnn_profile",
]

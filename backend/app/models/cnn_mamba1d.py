"""Optional CNN-Mamba classifier with an explicit dependency boundary."""

# ============================================================================
# 模块说明（教学注释）
#
# 本文件实现模型注册表中的 "cnn_mamba1d"：三段式 1D CNN 前端 + 真实
# mamba_ssm.Mamba 序列建模层 + 均值池化分类头。
#
# 在系统中的位置与关键约束：
#   - 能力目录固定公开 15 个目标模型，cnn_mamba1d 是其中之一；但当前
#     环境 mamba-ssm 依赖不可用，因此目录中该模型返回 available=false。
#   - 业务硬约束（AGENTS.md）：依赖缺失时"不得用近似网络静默替代"。
#     本文件就是这条约束的代码化——模块顶部用 try/except 探测导入，
#     构造函数在构建任何层之前先 _require_mamba()，缺依赖就直接抛
#     ModelDependencyError，把安装指引带给调用方。
#   - 导入边界设计：模块级 import 失败不会让整个 models 包挂掉，
#     其他 14 个模型照常可用；只有真正实例化 CNNMamba1D 时才报错。
# ============================================================================

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ModelDependencyError(RuntimeError):
    """Raised when an optional model dependency is unavailable or unusable."""
    # 中文教学注释：可选模型依赖缺失/不可用的专用异常类型。
    # 单独定义一个 RuntimeError 子类，是为了让上层（训练入口 / 能力目录）
    # 能用 except ModelDependencyError 精确识别"这是依赖问题"，
    # 而不是把 ImportError、ValueError 等混在一起处理。


# 模块级依赖探测：尝试导入真实的 mamba_ssm.Mamba。
# 成功则 _MAMBA_CLASS 指向真实类、_MAMBA_IMPORT_ERROR 为 None；
# 失败则记下原始异常（except Exception 而非 except ImportError，
# 因为 mamba-ssm 在 CUDA/编译算子不匹配时也可能抛 OSError 等），
# 但不让模块导入本身失败——延迟到真正构造模型时才报错。
try:
    from mamba_ssm import Mamba as _MAMBA_CLASS
    _MAMBA_IMPORT_ERROR: BaseException | None = None
except Exception as _mamba_import_error:
    _MAMBA_CLASS = None
    _MAMBA_IMPORT_ERROR = _mamba_import_error


# 面向用户的安装指引文案，作为 ModelDependencyError 的消息透出，
# 前端/日志可直接展示，不需要调用方再拼一次提示。
MAMBA_INSTALL_MESSAGE = (
    "CNN-Mamba 需要可选依赖 mamba-ssm（Python 导入名为 mamba_ssm）。"
    "请在当前 Python 环境安装兼容版本："
    "python -m pip install mamba-ssm；"
    "若使用 CUDA，请同时按 PyTorch/CUDA 版本选择官方兼容 wheel。"
)


def mamba_available() -> bool:
    """Return whether the real mamba_ssm implementation can be imported."""

    # 中文教学注释：能力目录用它决定 cnn_mamba1d 的 available 标志。
    # 只反映"模块能否导入"，不保证 CUDA kernel 一定可运行——
    # 后者在构造 Mamba 层时还有第二道检查（见 CNNMamba1D.__init__）。
    return _MAMBA_CLASS is not None


def _require_mamba() -> type:
    # 中文教学注释：拿真实的 Mamba 类；拿不到就抛 ModelDependencyError，
    # 并用 `from _MAMBA_IMPORT_ERROR` 保留原始导入异常作为 __cause__，
    # 方便排查到底是缺包还是 CUDA/版本不匹配。
    if _MAMBA_CLASS is None:
        raise ModelDependencyError(MAMBA_INSTALL_MESSAGE) from _MAMBA_IMPORT_ERROR
    return _MAMBA_CLASS


def _resolve_class_count(class_count: int | None, num_classes: int | None) -> int:
    # 中文教学注释：
    # 统一 class_count / num_classes 两个历史参数名（注册表不同调用点
    # 用的名字不同）。两个都给时必须一致，否则视为配置错误直接拒绝；
    # 都没给或给的是非正整数也拒绝——类别数是分类头的关键维度，
    # 宁可早失败也不让模型带着错误输出维度进入训练。
    if class_count is None:
        class_count = num_classes
    elif num_classes is not None and int(class_count) != int(num_classes):
        raise ValueError("class_count 与 num_classes 必须一致")
    if class_count is None or int(class_count) <= 0:
        raise ValueError("class_count 必须是正整数")
    return int(class_count)


def _resolve_heads(d_model: int, requested_heads: int) -> int:
    # 中文教学注释：把请求的头数收敛到能整除 d_model 的最大值（至少 1）。
    # 当前 CNNMamba1D 的 Mamba 路径并未实际使用多头概念，这个助手是
    # 为兼容旧注册表参数预留的；收敛而非报错，是为了让旧配置仍能跑通。
    heads = max(1, int(requested_heads))
    while heads > 1 and d_model % heads != 0:
        heads -= 1
    return heads


def _build_cnn_stem(channels: Sequence[int]) -> nn.Sequential:
    # 中文教学注释：
    # 按给定的通道序列（本模型固定为三段）搭建 CNN 前端。每段是
    #   Conv1d(k=3, padding=1, bias=False) → BatchNorm1d → ReLU
    #   → MaxPool1d(k=2, s=2, ceil_mode=True)
    # k=3+padding=1 保持长度不变，池化负责把序列长度减半；
    # ceil_mode=True 保证奇数长度时向上取整，避免最后一帧被丢弃
    # （光谱序列长度来自 wide-feature-v2 的特征列数，不保证是 2 的幂）。
    # bias=False 与 BN 搭配是常规做法（偏置被 BN 的平移参数覆盖）。
    layers: list[nn.Module] = []
    in_channels = 1
    for out_channels in channels:
        layers.extend(
            [
                nn.Conv1d(in_channels, int(out_channels), kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(int(out_channels)),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
            ]
        )
        in_channels = int(out_channels)
    return nn.Sequential(*layers)


class CNNMamba1D(nn.Module):
    """Three-stage CNN followed by real ``mamba_ssm`` sequence layers.

    The constructor never substitutes a local approximation for Mamba.  If
    the optional dependency is absent, it raises :class:`ModelDependencyError`
    before trying to build any Mamba layer.
    """

    # 中文教学注释：
    # 网络结构：CNN 前端（局部峰形特征 + 8 倍长度压缩）
    #   → N 层 mamba_ssm.Mamba（长程依赖建模，序列格式为 (B, L, C)）
    #   → LayerNorm → 时间维均值池化 → Dropout → Linear 分类头。
    # 设计意图：CNN 先把高维光谱压成短序列，降低 Mamba 的计算量；
    # Mamba（选择性状态空间模型）负责捕获跨整段谱的长程相关，
    # 这是它比纯 CNN 更适合长光谱序列的原因。

    def __init__(
        self,
        input_length: int,
        class_count: int | None = None,
        dropout: float | None = None,
        hidden_size: int = 64,
        mamba_layers: int = 2,
        *,
        num_classes: int | None = None,
        d_model: int | None = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int | None = None,
        conv_channels: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        # 中文教学注释：
        # 参数含义：
        #   input_length  —— 输入光谱长度（宽表特征列数），仅做合法性校验
        #                    并记录，网络本身靠池化自适应长度；
        #   class_count / num_classes —— 类别数的两个兼容参数名；
        #   dropout       —— 分类头前的 Dropout 率，None 时默认 0.25；
        #   hidden_size / d_model —— Mamba 的模型维度，两个兼容名字，
        #                    d_model 优先；
        #   mamba_layers / num_layers —— Mamba 层数的两个兼容名字，
        #                    num_layers 优先；
        #   d_state/d_conv/expand —— 直接透传给 mamba_ssm.Mamba 的
        #                    状态维度、局部卷积宽度、扩展因子；
        #   conv_channels —— 可选地显式指定三段 CNN 的通道数。
        # 第一道依赖闸门：缺依赖时在这里就抛 ModelDependencyError，
        # 不会建出"半个模型"——符合"不得静默替代"的约束。
        mamba_class = _require_mamba()
        if int(input_length) <= 0:
            raise ValueError("input_length 必须是正整数")
        if num_layers is not None:
            mamba_layers = num_layers
        if int(mamba_layers) <= 0:
            raise ValueError("mamba_layers 必须是正整数")
        if int(d_state) <= 0 or int(d_conv) <= 0 or int(expand) <= 0:
            raise ValueError("d_state、d_conv、expand 必须是正整数")

        # dropout=None → 默认 0.25；范围限制在 [0,1]，越界直接拒绝。
        dropout_value = 0.25 if dropout is None else float(dropout)
        if not 0.0 <= dropout_value <= 1.0:
            raise ValueError("dropout 必须位于 [0, 1]")
        requested_d_model = int(hidden_size if d_model is None else d_model)
        if requested_d_model <= 0:
            raise ValueError("d_model/hidden_size 必须是正整数")
        # CNN 三段通道的默认推导：d_model/4 → d_model/2 → d_model
        # （带下限 8/16），形成逐段加宽的金字塔；显式给了 conv_channels
        # 则必须恰好三个正整数，且未显式给 d_model 时以最后一段通道
        # 作为 d_model，保证 CNN 输出维度 == Mamba 输入维度。
        if conv_channels is None:
            channels = (
                max(8, requested_d_model // 4),
                max(16, requested_d_model // 2),
                requested_d_model,
            )
        else:
            channels = tuple(int(value) for value in conv_channels)
            if len(channels) != 3 or any(value <= 0 for value in channels):
                raise ValueError("conv_channels 必须包含三个正整数")
            if d_model is None:
                requested_d_model = channels[-1]
        self.input_length = int(input_length)
        self.class_count = _resolve_class_count(class_count, num_classes)
        self.d_model = requested_d_model
        # 无论 conv_channels 怎么配，最后一段强制对齐到 d_model——
        # 这是 CNN 与 Mamba 之间的维度契约，不能断。
        self.conv_channels = (*channels[:2], self.d_model)
        self.cnn = _build_cnn_stem(self.conv_channels)
        self.mamba_layers = nn.ModuleList()
        # 第二道依赖闸门：类能 import 不代表能实例化（例如 CUDA kernel
        # 未编译/版本不匹配会在构造时抛错）。把任何构造异常统一包装成
        # ModelDependencyError，让上层按"依赖不可用"一种口径处理。
        try:
            for _ in range(int(mamba_layers)):
                self.mamba_layers.append(
                    mamba_class(
                        d_model=self.d_model,
                        d_state=int(d_state),
                        d_conv=int(d_conv),
                        expand=int(expand),
                    )
                )
        except Exception as exc:
            raise ModelDependencyError(
                "mamba_ssm 已找到，但 Mamba 层初始化失败；请安装与当前 PyTorch/CUDA 匹配的 mamba-ssm 版本。"
            ) from exc
        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout_value)
        self.head = nn.Linear(self.d_model, self.class_count)

    @property
    def classifier(self) -> nn.Linear:
        # 中文教学注释：以 property 暴露 head 的别名，兼容其他模型类
        # "分类层叫 classifier" 的访问习惯（训练/可解释性代码据此取
        # 最后一层做权重检查或 Grad-CAM 目标定位）。
        return self.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 中文教学注释：
        # 输入允许 (B, L) 或 (B, 1, L)：二维时自动补通道维；
        # 其余形状（多通道、四维等）直接拒绝，避免静默算出错误结果。
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("CNNMamba1D 输入必须是 (batch, 1, length)")
        # CNN 输出 (B, C=d_model, L/8)；transpose 成 Mamba 需要的
        # (B, L, C) 序列格式，逐层过 Mamba。
        sequence = self.cnn(x).transpose(1, 2)
        for layer in self.mamba_layers:
            sequence = layer(sequence)
        sequence = self.norm(sequence)
        # 时间维均值池化聚合成一个向量，再 Dropout + Linear 得 logits
        # （raw logits，供 CrossEntropyLoss 直接使用）。
        return self.head(self.dropout(sequence.mean(dim=1)))


def build_cnn_mamba1d(
    input_length: int,
    class_count: int | None = None,
    dropout: float | None = None,
    hidden_size: int = 64,
    mamba_layers: int = 2,
    **kwargs: object,
) -> CNNMamba1D:
    """Build CNN-Mamba or raise an actionable dependency error."""

    # 中文教学注释：注册表使用的工厂函数。签名与类构造函数对齐，
    # 其余超参经 **kwargs 透传（d_model、d_state 等关键字参数）；
    # 依赖缺失的 ModelDependencyError 会原样向上传播给训练入口。
    return CNNMamba1D(
        input_length=input_length,
        class_count=class_count,
        dropout=dropout,
        hidden_size=hidden_size,
        mamba_layers=mamba_layers,
        **kwargs,
    )


# 旧名字别名：历史调用点用 build_cnn_mamba，保持可用。
build_cnn_mamba = build_cnn_mamba1d


# 显式导出清单：能力目录/注册表只应依赖这些公共符号。
__all__ = [
    "CNNMamba1D",
    "ModelDependencyError",
    "MAMBA_INSTALL_MESSAGE",
    "mamba_available",
    "build_cnn_mamba1d",
    "build_cnn_mamba",
]

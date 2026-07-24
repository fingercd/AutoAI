"""分类 v2 的文档对齐 1D CNN。

训练样本数决定通道宽度，输入特征数决定卷积核和池化强度；最终卷积层暴露为
Grad-CAM 目标层。二分类输出维度由调用方按单 logit 契约传入。
"""

from __future__ import annotations

# ======================================================================
# 模块级说明
#
# 本文件实现“分类 v2”训练路径使用的文档对齐（document-exact）三段式 1D CNN：
# 输入是单通道光谱/色谱曲线（Batch×1×L），输出各类别的 logit。
#
# 在系统中的位置：
#   - 位于 backend/app/models/，是能力目录 15 个分类模型中的 `cnn1d`。
#   - 训练编排层先调用 `resolve_cnn_profile`，把当前折的训练样本数 N 和
#     输入特征长度 L（即 wide-feature-v2 宽表的特征列数）解析成网络超参数，
#     再实例化 `CNN1DDocumentV2` 完成训练。
#   - 可解释性模块通过 `gradcam_target_layer()` 取得最后一个 Conv1d，
#     在其上注册 hook 执行 1D Grad-CAM，把重要性热力图回投到输入特征轴。
#
# 协作模块：
#   - `cnn_se1d.py` 直接继承本文件的 `_DocumentCNN1D`，仅打开 SE 门控开关。
#   - 上游数据契约是 wide-feature-v2/v1 宽表：每条曲线一行，特征列共享同一坐标轴。
#
# 关键设计约束：
#   - 结构随数据规模自适应：样本量决定通道宽度与 dropout 强度，特征长度决定
#     卷积核与池化强度，全部映射集中在模块顶部的查表常量里，便于审阅和对齐文档。
#   - 卷积核必须为奇数并配合 padding=kernel//2，保证卷积不改变序列长度，
#     长度压缩只由池化层完成。
#   - 输出维度 class_count 完全由调用方传入（二分类的单 logit 契约也由调用方
#     决定），本模块不自行推断类别数。
#   - 本文件刻意不依赖共享 registry / profile 工厂，可独立解析、独立测试。
# ======================================================================

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


# 样本量分档 → 三段卷积的输出通道数。
# 样本越少通道越窄：用小容量网络压低方差，缓解小样本过拟合。
_CHANNELS_BY_SAMPLE_BAND: dict[str, tuple[int, int, int]] = {
    "small": (8, 16, 32),
    "medium": (16, 32, 64),
    "large": (32, 64, 128),
}
# 特征长度分档 → 三段卷积的卷积核大小（全部为奇数）。
# 曲线越长首层感受野越大：先用大核抓宽峰/整体包络，再用小核细化局部特征。
_KERNELS_BY_FEATURE_BAND: dict[str, tuple[int, int, int]] = {
    "short": (7, 5, 3),
    "medium": (9, 5, 3),
    "long": (9, 7, 5),
}
# 特征长度分档 → 三段池化窗口。长曲线需要更强的早期下采样来压缩时序长度，
# 控制后续层的计算量与特征图尺寸。
_POOLS_BY_FEATURE_BAND: dict[str, tuple[int, int, int]] = {
    "short": (2, 2, 2),
    "medium": (4, 2, 2),
    "long": (4, 4, 2),
}
# 样本量分档 → dropout 强度：样本越少正则化越强。
_DROPOUT_BY_SAMPLE_BAND = {"small": 0.5, "medium": 0.4, "large": 0.3}


# 解析后的架构档案（frozen dataclass）：一次解析、多处共享且不可变，
# 网络构建、训练日志和可解释性读取的都是同一组数值，避免口径漂移。
@dataclass(frozen=True)
class CNN1DProfile:
    """Resolved N/L architecture values shared by the document CNN variants."""

    sample_count: int
    input_length: int
    sample_band: str
    feature_band: str
    channels: tuple[int, int, int]
    kernels: tuple[int, int, int]
    pools: tuple[int, int, int]
    dropout: float

    @property
    def N(self) -> int:
        # 文档/旧调用方习惯用 N 表示训练样本数，这里提供等价别名
        return self.sample_count

    @property
    def L(self) -> int:
        # 同理，L 即输入特征长度（每条曲线的点数）
        return self.input_length


# 旧名称别名：保持历史调用方/import 不破裂
CNNProfile = CNN1DProfile


# 按训练样本数划分样本档位：<=100 为小样本，<300 为中样本，其余为大样本
def _sample_band(sample_count: int) -> str:
    if sample_count <= 100:
        return "small"
    if sample_count < 300:
        return "medium"
    return "large"


# 按输入特征长度划分特征档位：<=1000 短曲线，<3000 中等，其余长曲线
def _feature_band(input_length: int) -> str:
    if input_length <= 1000:
        return "short"
    if input_length < 3000:
        return "medium"
    return "long"


# 把外部传入的样本档位描述统一规整为 small/medium/large。
# 容忍大小写、下划线和 short/long 等同义写法；无法识别时抛 ValueError，
# 由训练入口转成对用户可读的拒绝原因。
def _normalise_sample_band(value: Any) -> str:
    band = str(value or "").strip().lower().replace("_", "-")
    aliases = {"small": "small", "short": "small", "medium": "medium", "large": "large", "long": "large"}
    try:
        return aliases[band]
    except KeyError as exc:
        raise ValueError(f"未知 CNN N profile: {value}") from exc


# 同理，把特征档位描述统一规整为 short/medium/long
def _normalise_feature_band(value: Any) -> str:
    band = str(value or "").strip().lower().replace("_", "-")
    aliases = {"short": "short", "small": "short", "medium": "medium", "long": "long", "large": "long"}
    try:
        return aliases[band]
    except KeyError as exc:
        raise ValueError(f"未知 CNN L profile: {value}") from exc


# 从 profile 对象中取出 values 子映射。
# profile 既可以是 Mapping（如 dict），也可以是带 values 属性的对象；
# 取不到合法 values 时返回空映射，让后续逻辑走默认值分支。
def _profile_values(profile: object | None) -> Mapping[str, Any]:
    if profile is None:
        return {}
    if isinstance(profile, Mapping):
        values = profile.get("values")
    else:
        values = getattr(profile, "values", None)
    return values if isinstance(values, Mapping) else {}


# 依次在 values 子映射和 profile 本体上按多个候选键查找配置值，
# 返回第一个非 None 的结果。多候选键是为了兼容历史调用方与未来共享
# builder 可能使用的不同字段命名（如 L / input_length / feature_count）。
def _profile_lookup(profile: object | None, values: Mapping[str, Any], *keys: str) -> Any:
    for source in (values, profile):
        if source is None:
            continue
        for key in keys:
            if isinstance(source, Mapping):
                value = source.get(key)
            else:
                value = getattr(source, key, None)
            if value is not None:
                return value
    return None


# 把外部值强制转换为“三个正整数”的元组（channels/kernels/pools 专用）。
# 字符串、字节串或非序列一律拒绝；长度不为 3 或含非正数也拒绝，
# 防止非法 profile 静默造出畸形网络。
def _coerce_triplet(value: Any, *, name: str) -> tuple[int, int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"CNN profile 的 {name} 必须是三个整数")
    values = tuple(int(item) for item in value)
    if len(values) != 3 or any(item <= 0 for item in values):
        raise ValueError(f"CNN profile 的 {name} 必须是三个正整数")
    return values  # type: ignore[return-value]


# 解析文档版 N/L profile，返回冻结的 CNN1DProfile。
#
# 取值优先级：显式入参（input_length/sample_count/N/L/dropout）>
# profile 对象中的字段 > 按 N/L 自动分档后的默认查表值。
# input_length 与 L 同时给出时必须一致，否则视为调用方口径错误并拒绝。
# 只有 sample_band 没有具体 sample_count 时，用档位代表值（100/200/301）
# 兜底，保证仍能确定 dropout 等依赖档位的超参数。
#
# 异常：N/L 缺失或不一致、非正数、三元组非法、kernel 为偶数、dropout
# 不在 [0, 1) 时均抛 ValueError。
def resolve_cnn_profile(
    input_length: int | None = None,
    sample_count: int | None = None,
    *,
    profile: object | None = None,
    N: int | None = None,
    L: int | None = None,
    dropout: float | None = None,
) -> CNN1DProfile:
    """Resolve the document's N/L profile without depending on the shared registry."""

    values = _profile_values(profile)
    profile_length = _profile_lookup(profile, values, "input_length", "feature_count", "L", "length")
    profile_samples = _profile_lookup(
        profile,
        values,
        "sample_count",
        "train_sample_count",
        "N",
        "n",
        "n_train",
    )

    # 解析输入特征长度 L：显式 input_length 优先，其次 L，最后 profile
    if input_length is None:
        input_length = L if L is not None else profile_length
    elif L is not None and int(L) != int(input_length):
        raise ValueError("CNN 的 input_length 与 L profile 不一致")
    if input_length is None:
        raise ValueError("CNN 需要 input_length/L 才能解析特征 profile")
    input_length = int(input_length)
    if input_length <= 0:
        raise ValueError("CNN 的 input_length/L 必须大于 0")

    # 解析训练样本数 N：显式 sample_count 优先，其次 N，最后 profile；
    # profile 只给档位时映射到该档位的代表样本数
    if sample_count is None:
        sample_count = N if N is not None else profile_samples
    profile_sample_band = _profile_lookup(profile, values, "sample_band", "n_band")
    if sample_count is None:
        if profile_sample_band is None:
            raise ValueError("CNN 需要 sample_count/N 才能解析样本 profile")
        sample_band = _normalise_sample_band(profile_sample_band)
        sample_count = {"small": 100, "medium": 200, "large": 301}[sample_band]
    sample_count = int(sample_count)
    if sample_count <= 0:
        raise ValueError("CNN 的 sample_count/N 必须大于 0")
    if profile_sample_band is None:
        sample_band = _sample_band(sample_count)
    else:
        sample_band = _normalise_sample_band(profile_sample_band)

    # 特征档位：profile 显式指定优先，否则按 L 自动分档
    profile_feature_band = _profile_lookup(profile, values, "feature_band", "l_band")
    if profile_feature_band is None:
        feature_band = _feature_band(input_length)
    else:
        feature_band = _normalise_feature_band(profile_feature_band)

    # channels/kernels/pools：profile 显式三元组优先，否则按档位查默认表
    channels_value = _profile_lookup(profile, values, "channels", "cnn_channels", "channel_widths")
    kernels_value = _profile_lookup(profile, values, "kernels", "cnn_kernels")
    pools_value = _profile_lookup(profile, values, "pools", "cnn_pools", "pool_sizes")
    channels = (
        _coerce_triplet(channels_value, name="channels")
        if channels_value is not None
        else _CHANNELS_BY_SAMPLE_BAND[sample_band]
    )
    kernels = (
        _coerce_triplet(kernels_value, name="kernels")
        if kernels_value is not None
        else _KERNELS_BY_FEATURE_BAND[feature_band]
    )
    pools = (
        _coerce_triplet(pools_value, name="pools")
        if pools_value is not None
        else _POOLS_BY_FEATURE_BAND[feature_band]
    )
    # 偶数 kernel 会让 padding=kernel//2 无法保持长度对齐，直接拒绝
    if any(kernel % 2 == 0 for kernel in kernels):
        raise ValueError("CNN profile 的 kernels 必须使用奇数以保持长度对齐")

    # dropout：显式入参 > profile 字段 > 按样本档位的默认值；必须在 [0, 1)
    resolved_dropout = dropout
    if resolved_dropout is None:
        resolved_dropout = _profile_lookup(profile, values, "dropout", "head_dropout")
    if resolved_dropout is None:
        resolved_dropout = _DROPOUT_BY_SAMPLE_BAND[sample_band]
    resolved_dropout = float(resolved_dropout)
    if not 0.0 <= resolved_dropout < 1.0:
        raise ValueError("CNN dropout 必须位于 [0, 1) 区间")

    return CNN1DProfile(
        sample_count=sample_count,
        input_length=input_length,
        sample_band=sample_band,
        feature_band=feature_band,
        channels=channels,
        kernels=kernels,
        pools=pools,
        dropout=resolved_dropout,
    )


# 旧名称别名：与 resolve_cnn_profile 完全等价
build_cnn_profile = resolve_cnn_profile


# 汇总类别数的各种候选命名（class_count/classes/num_classes/n_classes），
# 取第一个非 None 值；缺失或非正整数时拒绝。
# 注意：本模块不推断类别数，二分类单 logit 契约也由调用方通过该值表达。
def _resolve_class_count(
    class_count: int | None,
    *,
    classes: int | None,
    num_classes: int | None,
    n_classes: int | None,
) -> int:
    resolved = class_count
    if resolved is None:
        resolved = classes if classes is not None else num_classes
    if resolved is None:
        resolved = n_classes
    if resolved is None or int(resolved) <= 0:
        raise ValueError("CNN 需要正整数 class_count/classes")
    return int(resolved)


# 文档版 1D CNN 的共享基类：三段
# “Conv1d → BatchNorm1d → ReLU（→ 可选 SE 门控）→ MaxPool1d”
# 堆叠后接 AdaptiveAvgPool1d(1) 全局平均池化、Dropout 和线性分类头。
#
# 设计要点：
#   - padding=kernel//2 且 kernel 为奇数，卷积前后序列长度不变，
#     长度压缩全部交给池化层，便于按 profile 精确控制形状。
#   - MaxPool1d 用 ceil_mode=True：长度不能被窗口整除时向上取整，
#     保留末尾不足一个窗口的信号，避免边缘谱峰被丢弃。
#   - 末尾 AdaptiveAvgPool1d(1) 使分类头输入维度只取决于末段通道数，
#     与池化后剩余长度无关，因此任意 L 都能工作。
#   - 卷积层 bias=False：紧跟 BatchNorm 时偏置会被归一化抵消，属于冗余参数。
#   - 最后一个卷积层保存为 _gradcam_layer，供 1D Grad-CAM 注册 hook。
#   - 同时暴露 conv_layers/block_layers/blocks/conv_blocks/se_blocks 等多种
#     属性命名，兼容不同调用方与测试对内部结构的访问习惯。
class _DocumentCNN1D(nn.Module):
    def __init__(
        self,
        input_length: int | None,
        class_count: int | None,
        sample_count: int | None,
        dropout: float | None,
        hidden_size: int,
        *,
        profile: object | None,
        model_profile: object | None,
        N: int | None,
        L: int | None,
        classes: int | None,
        num_classes: int | None,
        n_classes: int | None,
        n_samples: int | None,
        use_se: bool,
    ) -> None:
        super().__init__()
        # hidden_size 是历史遗留参数，文档版 profile 不使用它；显式丢弃以保持签名兼容
        _ = hidden_size
        # profile 与 model_profile 是同一信息的两种命名，取先出现者
        profile_object = profile if profile is not None else model_profile
        if sample_count is None:
            sample_count = n_samples
        resolved_classes = _resolve_class_count(
            class_count,
            classes=classes,
            num_classes=num_classes,
            n_classes=n_classes,
        )
        resolved = resolve_cnn_profile(
            input_length,
            sample_count,
            profile=profile_object,
            N=N,
            L=L,
            dropout=dropout,
        )

        # 把解析结果平铺为实例属性，便于训练器/测试直接读取架构口径
        self.profile = resolved
        self.model_profile = resolved
        self.input_length = resolved.input_length
        self.sample_count = resolved.sample_count
        self.class_count = resolved_classes
        self.channels = resolved.channels
        self.kernels = resolved.kernels
        self.pools = resolved.pools
        self.dropout_rate = resolved.dropout

        layers: list[nn.Module] = []
        block_layers: list[tuple[nn.Module, ...]] = []
        convolution_layers: list[nn.Conv1d] = []
        se_layers: list[nn.Module] = []
        in_channels = 1
        # 按 profile 依次堆叠三个卷积块；每块的输出通道成为下一块的输入通道
        for out_channels, kernel_size, pool_size in zip(resolved.channels, resolved.kernels, resolved.pools):
            convolution = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            )
            batch_norm = nn.BatchNorm1d(out_channels)
            activation = nn.ReLU(inplace=True)
            block: list[nn.Module] = [convolution, batch_norm, activation]
            if use_se:
                # SE 门控插在激活之后、池化之前：对已激活的特征图做通道重标定
                se = self._make_se_block(out_channels)
                block.append(se)
                se_layers.append(se)
            block.append(nn.MaxPool1d(pool_size, ceil_mode=True))
            layers.extend(block)
            block_layers.append(tuple(block))
            convolution_layers.append(convolution)
            in_channels = out_channels

        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1))
        self.dropout = nn.Dropout(resolved.dropout)
        self.classifier = nn.Linear(resolved.channels[-1], resolved_classes)
        self.conv_layers = tuple(convolution_layers)
        self.block_layers = tuple(block_layers)
        self.blocks = self.block_layers
        self.conv_blocks = self.block_layers
        self.se_blocks = tuple(se_layers)
        self._gradcam_layer = convolution_layers[-1]

    # 基类不提供 SE 实现；只有 use_se=True 的子类（cnn_se1d）会覆盖它。
    # 若基类被误用为 SE 网络，这里立即报错而不是静默退化。
    def _make_se_block(self, channels: int) -> nn.Module:
        raise RuntimeError(f"SE block is not available for {self.__class__.__name__}: {channels}")

    # 前向传播：接受 Batch×L 或 Batch×1×L 两种输入形状，
    # 统一补成单通道三维后过特征提取器，再展平进分类头。
    # 其他维度直接拒绝，避免错误形状在卷积深处才爆出难以定位的异常。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError("CNN 输入必须是 Batch×1×L 或 Batch×L")
        features = self.features(x).flatten(1)
        return self.classifier(self.dropout(features))

    def gradcam_target_layer(self) -> nn.Conv1d:
        """Return the explicit last Conv1d used by the 1D Grad-CAM route."""

        # 显式返回最后一段卷积：越靠后的特征图语义级别越高，
        # Grad-CAM 热力图与原始谱峰位置的对应关系也更直观
        return self._gradcam_layer


# 不带 SE 门控的具体文档版 CNN：能力目录中的 `cnn1d`。
# 与基类的唯一区别是 use_se=False；所有结构超参数都继承 N/L profile 解析逻辑。
class CNN1DDocumentV2(_DocumentCNN1D):
    """Document-exact three-block CNN classifier for the v2 training path."""

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
            use_se=False,
        )


# 对外别名：保持不同历史调用方的 import 名称都可用
CNN1DV2 = CNN1DDocumentV2
CNN1D_V2 = CNN1DDocumentV2


__all__ = [
    "CNN1DProfile",
    "CNNProfile",
    "CNN1DDocumentV2",
    "CNN1DV2",
    "CNN1D_V2",
    "resolve_cnn_profile",
    "build_cnn_profile",
]

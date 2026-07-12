from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


_CHANNELS_BY_SAMPLE_BAND: dict[str, tuple[int, int, int]] = {
    "small": (8, 16, 32),
    "medium": (16, 32, 64),
    "large": (32, 64, 128),
}
_KERNELS_BY_FEATURE_BAND: dict[str, tuple[int, int, int]] = {
    "short": (7, 5, 3),
    "medium": (9, 5, 3),
    "long": (9, 7, 5),
}
_POOLS_BY_FEATURE_BAND: dict[str, tuple[int, int, int]] = {
    "short": (2, 2, 2),
    "medium": (4, 2, 2),
    "long": (4, 2, 2),
}
_DROPOUT_BY_SAMPLE_BAND = {"small": 0.5, "medium": 0.4, "large": 0.3}


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
        return self.sample_count

    @property
    def L(self) -> int:
        return self.input_length


CNNProfile = CNN1DProfile


def _sample_band(sample_count: int) -> str:
    if sample_count <= 100:
        return "small"
    if sample_count <= 300:
        return "medium"
    return "large"


def _feature_band(input_length: int) -> str:
    if input_length <= 1000:
        return "short"
    if input_length <= 3000:
        return "medium"
    return "long"


def _normalise_sample_band(value: Any) -> str:
    band = str(value or "").strip().lower().replace("_", "-")
    aliases = {"small": "small", "short": "small", "medium": "medium", "large": "large", "long": "large"}
    try:
        return aliases[band]
    except KeyError as exc:
        raise ValueError(f"未知 CNN N profile: {value}") from exc


def _normalise_feature_band(value: Any) -> str:
    band = str(value or "").strip().lower().replace("_", "-")
    aliases = {"short": "short", "small": "short", "medium": "medium", "long": "long", "large": "long"}
    try:
        return aliases[band]
    except KeyError as exc:
        raise ValueError(f"未知 CNN L profile: {value}") from exc


def _profile_values(profile: object | None) -> Mapping[str, Any]:
    if profile is None:
        return {}
    if isinstance(profile, Mapping):
        values = profile.get("values")
    else:
        values = getattr(profile, "values", None)
    return values if isinstance(values, Mapping) else {}


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


def _coerce_triplet(value: Any, *, name: str) -> tuple[int, int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"CNN profile 的 {name} 必须是三个整数")
    values = tuple(int(item) for item in value)
    if len(values) != 3 or any(item <= 0 for item in values):
        raise ValueError(f"CNN profile 的 {name} 必须是三个正整数")
    return values  # type: ignore[return-value]


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

    if input_length is None:
        input_length = L if L is not None else profile_length
    elif L is not None and int(L) != int(input_length):
        raise ValueError("CNN 的 input_length 与 L profile 不一致")
    if input_length is None:
        raise ValueError("CNN 需要 input_length/L 才能解析特征 profile")
    input_length = int(input_length)
    if input_length <= 0:
        raise ValueError("CNN 的 input_length/L 必须大于 0")

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

    profile_feature_band = _profile_lookup(profile, values, "feature_band", "l_band")
    if profile_feature_band is None:
        feature_band = _feature_band(input_length)
    else:
        feature_band = _normalise_feature_band(profile_feature_band)

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
    if any(kernel % 2 == 0 for kernel in kernels):
        raise ValueError("CNN profile 的 kernels 必须使用奇数以保持长度对齐")

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


build_cnn_profile = resolve_cnn_profile


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
        _ = hidden_size
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

    def _make_se_block(self, channels: int) -> nn.Module:
        raise RuntimeError(f"SE block is not available for {self.__class__.__name__}: {channels}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError("CNN 输入必须是 Batch×1×L 或 Batch×L")
        features = self.features(x).flatten(1)
        return self.classifier(self.dropout(features))

    def gradcam_target_layer(self) -> nn.Conv1d:
        """Return the explicit last Conv1d used by the 1D Grad-CAM route."""

        return self._gradcam_layer


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

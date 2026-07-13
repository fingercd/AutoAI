"""Optional CNN-Mamba classifier with an explicit dependency boundary."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ModelDependencyError(RuntimeError):
    """Raised when an optional model dependency is unavailable or unusable."""


try:
    from mamba_ssm import Mamba as _MAMBA_CLASS
    _MAMBA_IMPORT_ERROR: BaseException | None = None
except Exception as _mamba_import_error:
    _MAMBA_CLASS = None
    _MAMBA_IMPORT_ERROR = _mamba_import_error


MAMBA_INSTALL_MESSAGE = (
    "CNN-Mamba 需要可选依赖 mamba-ssm（Python 导入名为 mamba_ssm）。"
    "请在当前 Python 环境安装兼容版本："
    "python -m pip install mamba-ssm；"
    "若使用 CUDA，请同时按 PyTorch/CUDA 版本选择官方兼容 wheel。"
)


def mamba_available() -> bool:
    """Return whether the real mamba_ssm implementation can be imported."""

    return _MAMBA_CLASS is not None


def _require_mamba() -> type:
    if _MAMBA_CLASS is None:
        raise ModelDependencyError(MAMBA_INSTALL_MESSAGE) from _MAMBA_IMPORT_ERROR
    return _MAMBA_CLASS


def _resolve_class_count(class_count: int | None, num_classes: int | None) -> int:
    if class_count is None:
        class_count = num_classes
    elif num_classes is not None and int(class_count) != int(num_classes):
        raise ValueError("class_count 与 num_classes 必须一致")
    if class_count is None or int(class_count) <= 0:
        raise ValueError("class_count 必须是正整数")
    return int(class_count)


def _resolve_heads(d_model: int, requested_heads: int) -> int:
    heads = max(1, int(requested_heads))
    while heads > 1 and d_model % heads != 0:
        heads -= 1
    return heads


def _build_cnn_stem(channels: Sequence[int]) -> nn.Sequential:
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
        mamba_class = _require_mamba()
        if int(input_length) <= 0:
            raise ValueError("input_length 必须是正整数")
        if num_layers is not None:
            mamba_layers = num_layers
        if int(mamba_layers) <= 0:
            raise ValueError("mamba_layers 必须是正整数")
        if int(d_state) <= 0 or int(d_conv) <= 0 or int(expand) <= 0:
            raise ValueError("d_state、d_conv、expand 必须是正整数")

        dropout_value = 0.25 if dropout is None else float(dropout)
        if not 0.0 <= dropout_value <= 1.0:
            raise ValueError("dropout 必须位于 [0, 1]")
        requested_d_model = int(hidden_size if d_model is None else d_model)
        if requested_d_model <= 0:
            raise ValueError("d_model/hidden_size 必须是正整数")
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
        self.conv_channels = (*channels[:2], self.d_model)
        self.cnn = _build_cnn_stem(self.conv_channels)
        self.mamba_layers = nn.ModuleList()
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
        return self.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("CNNMamba1D 输入必须是 (batch, 1, length)")
        sequence = self.cnn(x).transpose(1, 2)
        for layer in self.mamba_layers:
            sequence = layer(sequence)
        sequence = self.norm(sequence)
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

    return CNNMamba1D(
        input_length=input_length,
        class_count=class_count,
        dropout=dropout,
        hidden_size=hidden_size,
        mamba_layers=mamba_layers,
        **kwargs,
    )


build_cnn_mamba = build_cnn_mamba1d


__all__ = [
    "CNNMamba1D",
    "ModelDependencyError",
    "MAMBA_INSTALL_MESSAGE",
    "mamba_available",
    "build_cnn_mamba1d",
    "build_cnn_mamba",
]

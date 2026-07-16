"""按训练样本数 N 与特征长度 L 解析模型 profile。

profile 决定网络宽度、卷积核、池化、dropout 或传统模型候选范围，但不改变模型 ID。
分档只使用当前训练折的规模，最终解析值会写进 Run 元数据以便复核。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class ModelProfile:
    """一次训练折实际采用的样本/特征分档与参数集合。"""
    model_type: str
    train_sample_count: int
    feature_count: int
    sample_band: str
    feature_band: str
    dropout: float
    values: dict[str, Any]


PROFILE_MODEL_TYPES = {
    "pls_da",
    "pca_lda",
    "logistic_regression",
    "svm",
    "random_forest",
    "xgboost",
    "pca_mlp",
    "cnn1d",
    "cnn1d_se",
    "transformer1d",
    "resnet1d",
    "inception1d",
    "tcn1d",
    "cnn_transformer1d",
    "cnn_mamba1d",
    "dscarnet",
}

_CNN_CHANNELS_BY_SAMPLE_BAND = {
    "small": [8, 16, 32],
    "medium": [16, 32, 64],
    "large": [32, 64, 128],
}
_CNN_KERNELS_BY_FEATURE_BAND = {
    "short": [7, 5, 3],
    "medium": [9, 5, 3],
    "long": [9, 7, 5],
}
_CNN_POOLS_BY_FEATURE_BAND = {
    "short": [2, 2, 2],
    "medium": [4, 2, 2],
    "long": [4, 4, 2],
}
_RESNET_L_VALUES = {
    "short": {"kernels": [7, 5, 3], "pools": [2, 2, 2]},
    "medium": {"kernels": [9, 5, 3], "pools": [4, 2, 2]},
    "long": {"kernels": [9, 7, 5], "pools": [4, 4, 2]},
}
_INCEPTION_CHANNELS_BY_SAMPLE_BAND = {
    "small": {"stem_channels": 8, "block_channels": [16, 32, 32]},
    "medium": {"stem_channels": 16, "block_channels": [32, 64, 64]},
    "large": {"stem_channels": 32, "block_channels": [64, 128, 128]},
}
_INCEPTION_L_VALUES = {
    "short": {"stem_kernel": 7, "branch_kernels": [1, 3, 5, 7], "pools": [2, 2, 2]},
    "medium": {"stem_kernel": 9, "branch_kernels": [1, 3, 7, 11], "pools": [4, 2, 2]},
    "long": {"stem_kernel": 11, "branch_kernels": [1, 5, 9, 15], "pools": [4, 4, 2]},
}
_TCN_CHANNELS_BY_SAMPLE_BAND = {"small": 32, "medium": 64, "large": 128}
_TCN_L_VALUES = {
    "short": {"stem_kernel": 7, "stem_pool": 2, "dilations": [1, 2, 4]},
    "medium": {"stem_kernel": 9, "stem_pool": 4, "dilations": [1, 2, 4]},
    "long": {"stem_kernel": 9, "stem_pool": 4, "dilations": [1, 2, 4, 8]},
}
_LONG_RANGE_VALUES = {
    "small": {"d_model": 32, "heads": 2, "layers": 1, "ffn": 64, "conv_channels": [8, 16, 32], "d_state": 16},
    "medium": {"d_model": 64, "heads": 4, "layers": 1, "ffn": 128, "conv_channels": [16, 32, 64], "d_state": 16},
    "large": {"d_model": 128, "heads": 4, "layers": 2, "ffn": 256, "conv_channels": [32, 64, 128], "d_state": 16},
}


def sample_band(n: int) -> str:
    """把当前训练折样本数 N 分成 small/medium/large。"""
    return "small" if int(n) <= 100 else ("medium" if int(n) < 300 else "large")


def feature_band(length: int) -> str:
    """把特征长度 L 分成 short/medium/long。"""
    return "short" if int(length) <= 1000 else ("medium" if int(length) < 3000 else "long")


def default_dropout(n: int) -> float:
    """小样本使用更强 dropout，降低深度模型过拟合风险。"""
    return {"small": 0.5, "medium": 0.4, "large": 0.3}[sample_band(n)]


def build_model_profile(
    model_type: str,
    *,
    train_sample_count: int,
    feature_count: int,
) -> ModelProfile:
    """解析非 DSCARNet 模型在当前训练折实际使用的 profile。"""
    model_key = str(model_type or "").strip().lower()
    model_key = {"transformer": "cnn_transformer1d", "transformer1d": "cnn_transformer1d"}.get(model_key, model_key)
    if model_key not in PROFILE_MODEL_TYPES:
        raise ValueError(f"未知模型，无法构建 profile: {model_type}")
    if int(train_sample_count) <= 1:
        raise ValueError("PCA-MLP 至少需要 2 个训练样本才能拟合 profile")
    if int(feature_count) <= 0:
        raise ValueError("特征数 L 必须大于 0")
    resolved_sample_band = sample_band(train_sample_count)
    resolved_feature_band = feature_band(feature_count)
    dropout = default_dropout(train_sample_count)
    values: dict[str, Any] = {
        "dropout": dropout,
        "sample_band": resolved_sample_band,
        "feature_band": resolved_feature_band,
    }
    if model_key == "pca_mlp":
        if int(train_sample_count) <= 100:
            requested_components, hidden_sizes = 32, [32, 16]
        elif int(train_sample_count) <= 300:
            requested_components, hidden_sizes = 64, [64, 32]
        else:
            requested_components, hidden_sizes = 128, [128, 64]
        values.update(
            {
                "pca_components": max(1, min(requested_components, int(train_sample_count) - 1, int(feature_count))),
                "hidden_sizes": hidden_sizes,
            }
        )
    elif model_key in {"cnn1d", "cnn1d_se"}:
        values.update(
            {
                "channels": list(_CNN_CHANNELS_BY_SAMPLE_BAND[resolved_sample_band]),
                "kernels": list(_CNN_KERNELS_BY_FEATURE_BAND[resolved_feature_band]),
                "pools": list(_CNN_POOLS_BY_FEATURE_BAND[resolved_feature_band]),
            }
        )
    elif model_key == "resnet1d":
        values.update(
            {
                "channels": list(_CNN_CHANNELS_BY_SAMPLE_BAND[resolved_sample_band]),
                **_RESNET_L_VALUES[resolved_feature_band],
            }
        )
    elif model_key == "inception1d":
        values.update(
            {
                **_INCEPTION_CHANNELS_BY_SAMPLE_BAND[resolved_sample_band],
                **_INCEPTION_L_VALUES[resolved_feature_band],
            }
        )
    elif model_key == "tcn1d":
        values.update(
            {
                "channels": _TCN_CHANNELS_BY_SAMPLE_BAND[resolved_sample_band],
                "kernel": 3,
                **_TCN_L_VALUES[resolved_feature_band],
            }
        )
    elif model_key in {"cnn_transformer1d", "cnn_mamba1d"}:
        long_range = _LONG_RANGE_VALUES[resolved_sample_band]
        values.update(
            {
                "d_model": long_range["d_model"],
                "heads": long_range["heads"],
                "layers": long_range["layers"],
                "ffn": long_range["ffn"],
                "conv_channels": list(long_range["conv_channels"]),
                "kernels": list(_CNN_KERNELS_BY_FEATURE_BAND[resolved_feature_band]),
                "pools": list(_CNN_POOLS_BY_FEATURE_BAND[resolved_feature_band]),
                "d_state": long_range["d_state"],
                "d_conv": 4,
                "expand": 2,
            }
        )
    return ModelProfile(
        model_type=model_key,
        train_sample_count=int(train_sample_count),
        feature_count=int(feature_count),
        sample_band=resolved_sample_band,
        feature_band=resolved_feature_band,
        dropout=dropout,
        values=values,
    )


def build_dscarnet_profile(*, train_sample_count: int, feature_count: int) -> dict[str, Any]:
    """解析 DSCARNet 的 PCA 数量、聚类通道和网络容量。"""
    n = int(train_sample_count)
    length = int(feature_count)
    n_band = sample_band(n)
    l_band = feature_band(length)
    cluster_grid = {
        "small": {"short": 3, "medium": 5, "long": 7},
        "medium": {"short": 5, "medium": 7, "long": 9},
        "large": {"short": 7, "medium": 9, "long": 11},
    }
    capacity = {
        "small": {"filter_number": 16, "n_inception": 1, "dense_layers": [32]},
        "medium": {"filter_number": 32, "n_inception": 1, "dense_layers": [64]},
        "large": {"filter_number": 64, "n_inception": 2, "dense_layers": [128]},
    }[n_band]
    n_target = math.ceil((math.sqrt(8 * (0.8**2) * length + 1) - 1) / 2)
    return {
        "sample_band": n_band,
        "feature_band": l_band,
        "pca_components": min(n_target, n - 1, length),
        "cluster_channels": cluster_grid[n_band][l_band],
        "conv1_kernel_size": {"short": 7, "medium": 11, "long": 19}[l_band],
        **capacity,
    }


def model_range_warnings(*, train_sample_count: int, feature_count: int) -> list[str]:
    """报告超出文档验证范围的 N/L，但不擅自拒绝可运行输入。"""
    warnings: list[str] = []
    if not 50 <= int(train_sample_count) <= 1000:
        warnings.append(f"训练样本数 N={train_sample_count} 超出文档适用范围 50-1000")
    if not 500 <= int(feature_count) <= 10000:
        warnings.append(f"特征数 L={feature_count} 超出文档适用范围 500-10000")
    return warnings

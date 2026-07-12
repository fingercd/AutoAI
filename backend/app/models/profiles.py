from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelProfile:
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


def sample_band(n: int) -> str:
    return "small" if int(n) <= 100 else ("medium" if int(n) <= 300 else "large")


def feature_band(length: int) -> str:
    return "short" if int(length) <= 1000 else ("medium" if int(length) <= 3000 else "long")


def default_dropout(n: int) -> float:
    return {"small": 0.5, "medium": 0.4, "large": 0.3}[sample_band(n)]


def build_model_profile(
    model_type: str,
    *,
    train_sample_count: int,
    feature_count: int,
) -> ModelProfile:
    model_key = str(model_type or "").strip().lower()
    if model_key not in PROFILE_MODEL_TYPES:
        raise ValueError(f"未知模型，无法构建 profile: {model_type}")
    resolved_sample_band = sample_band(train_sample_count)
    resolved_feature_band = feature_band(feature_count)
    dropout = default_dropout(train_sample_count)
    return ModelProfile(
        model_type=model_key,
        train_sample_count=int(train_sample_count),
        feature_count=int(feature_count),
        sample_band=resolved_sample_band,
        feature_band=resolved_feature_band,
        dropout=dropout,
        values={
            "dropout": dropout,
            "sample_band": resolved_sample_band,
            "feature_band": resolved_feature_band,
        },
    )


def model_range_warnings(*, train_sample_count: int, feature_count: int) -> list[str]:
    warnings: list[str] = []
    if not 50 <= int(train_sample_count) <= 1000:
        warnings.append(f"训练样本数 N={train_sample_count} 超出文档适用范围 50-1000")
    if not 500 <= int(feature_count) <= 10000:
        warnings.append(f"特征数 L={feature_count} 超出文档适用范围 500-10000")
    return warnings

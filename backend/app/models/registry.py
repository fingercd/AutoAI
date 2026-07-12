from __future__ import annotations

from typing import Any

import numpy as np
from torch import nn

from .cnn1d import CNN1D
from .dscarnet import dual_dscarnet
from .inception1d import Inception1D
from .pls_da import build_pls_da
from .logistic_regression import build_logistic_regression
from .pca_lda import build_pca_lda
from .random_forest import build_random_forest
from .resnet1d import ResNet1D
from .svm import build_svm
from .tcn1d import TCN1D
from .transformer import Transformer1D
from .xgboost import build_xgboost


MODEL_ALIASES = {
    "pls": "pls_da",
    "pls-da": "pls_da",
    "pls_da": "pls_da",
    "pca_lda": "pca_lda",
    "logistic_regression": "logistic_regression",
    "logistic-regression": "logistic_regression",
    "logreg": "logistic_regression",
    "1d-cnn": "cnn1d",
    "1dcnn": "cnn1d",
    "cnn1d": "cnn1d",
    "cnn1d_se": "cnn1d_se",
    "cnn-se": "cnn1d_se",
    "transformer": "transformer1d",
    "transformer1d": "transformer1d",
    "1d-transformer": "transformer1d",
    "resnet1d": "resnet1d",
    "1d-resnet": "resnet1d",
    "inception1d": "inception1d",
    "1d-inception": "inception1d",
    "tcn1d": "tcn1d",
    "1d-tcn": "tcn1d",
    "pca_mlp": "pca_mlp",
    "cnn_transformer1d": "cnn_transformer1d",
    "cnn_mamba1d": "cnn_mamba1d",
    "dscarnet": "dscarnet",
    "dscar_net": "dscarnet",
    "random_forest": "random_forest",
    "random-forest": "random_forest",
    "rf": "random_forest",
    "svm": "svm",
    "support_vector_machine": "svm",
    "xgboost": "xgboost",
    "xgb": "xgboost",
}

RETIRED_OR_REGRESSION_MODEL_TYPES = {
    "knn",
    "k-nearest-neighbors",
    "mlp",
    "mlp_baseline",
    "unet",
    "unet1d",
    "plsr",
    "svr",
}

TARGET_DEEP_MODEL_TYPES = {
    "pca_mlp",
    "cnn1d",
    "cnn1d_se",
    "resnet1d",
    "inception1d",
    "tcn1d",
    "cnn_transformer1d",
    "cnn_mamba1d",
    "dscarnet",
}
TARGET_TRADITIONAL_MODEL_TYPES = {
    "pls_da",
    "pca_lda",
    "logistic_regression",
    "svm",
    "random_forest",
    "xgboost",
}
TARGET_MODEL_TYPES = TARGET_DEEP_MODEL_TYPES | TARGET_TRADITIONAL_MODEL_TYPES
DEEP_MODEL_TYPES = {"cnn1d", "transformer1d", "resnet1d", "inception1d", "tcn1d", "dscarnet"}
TRADITIONAL_MODEL_TYPES = {
    "pls_da",
    "pca_lda",
    "logistic_regression",
    "svm",
    "random_forest",
    "xgboost",
}
SUPPORTED_MODEL_TYPES = DEEP_MODEL_TYPES | TRADITIONAL_MODEL_TYPES


ARCHITECTURE_VERSION = "docx-classification-v2"


class ModelNotImplementedForVersion(ValueError):
    pass


def canonical_model_type(model_type: str) -> str:
    key = str(model_type or "cnn1d").strip().lower()
    if key in RETIRED_OR_REGRESSION_MODEL_TYPES:
        raise ValueError("当前仅支持分类任务的 10 类模型；KNN/MLP/UNet 已移除，PLSR/SVR 是回归变体暂不启用")
    canonical = MODEL_ALIASES.get(key, key)
    if canonical in TARGET_MODEL_TYPES and canonical not in SUPPORTED_MODEL_TYPES:
        raise ModelNotImplementedForVersion(f"模型 {canonical} 尚未在 docx-classification-v2 实现")
    if canonical not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"当前仅支持分类任务的 10 类模型，不支持: {model_type}")
    return canonical


def model_family(model_type: str) -> str:
    return "traditional_ml" if canonical_model_type(model_type) in TRADITIONAL_MODEL_TYPES else "deep_learning"


def build_deep_model(config: Any, input_length: int, class_count: int, sample_count: int) -> nn.Module:
    model_type = canonical_model_type(config.model_type)
    if model_type == "cnn1d":
        return CNN1D(input_length, class_count, sample_count, config.dropout, config.hidden_size)
    if model_type == "transformer1d":
        return Transformer1D(input_length, class_count, config.dropout, max(config.hidden_size, 16), config.transformer_heads)
    if model_type == "resnet1d":
        return ResNet1D(input_length, class_count, config.dropout, max(config.hidden_size, 32))
    if model_type == "inception1d":
        return Inception1D(input_length, class_count, config.dropout, max(config.hidden_size, 32))
    if model_type == "tcn1d":
        return TCN1D(input_length, class_count, config.dropout, max(config.hidden_size, 32))
    if model_type == "dscarnet":
        raise ValueError("DSCARNet requires 2D AggMap SAR/CAR inputs; use build_dscarnet_model instead")
    raise ValueError(f"Unsupported model_type: {config.model_type}")


def build_dscarnet_model(
    config: Any,
    input_shape1: tuple[int, ...],
    input_shape2: tuple[int, ...],
    class_count: int,
) -> nn.Module:
    return dual_dscarnet(
        input_shape1,
        input_shape2,
        n_outputs=class_count,
        n_inception=max(1, int(config.dscarnet_inception_blocks)),
        last_avf=None,
    )


def parse_optional_int(value: Any) -> int | None:
    if value in {None, "", "none", "None"}:
        return None
    return int(value)


def build_traditional_model(config: Any, y: np.ndarray, class_count: int) -> Any:
    model_type = canonical_model_type(config.model_type)
    class_weight = "balanced" if config.class_balance == "class_weight" else None
    if model_type == "pls_da":
        return build_pls_da(getattr(config, "pls_components", 2) or 2)
    if model_type == "pca_lda":
        return build_pca_lda(getattr(config, "pca_components", 2) or 2)
    if model_type == "logistic_regression":
        return build_logistic_regression(getattr(config, "logistic_c", 1.0), config.seed, class_weight)
    if model_type == "random_forest":
        return build_random_forest(
            config.random_forest_n_estimators,
            parse_optional_int(config.random_forest_max_depth),
            config.random_forest_min_samples_leaf,
            getattr(config, "random_forest_max_features", "sqrt"),
            class_weight,
            config.seed,
        )
    if model_type == "svm":
        return build_svm(config.svm_c, config.svm_gamma, class_weight, config.seed, getattr(config, "svm_kernel", "rbf"))
    if model_type == "xgboost":
        return build_xgboost(
            y,
            class_count,
            config.class_balance,
            config.seed,
            config.xgboost_n_estimators,
            config.xgboost_max_depth,
            config.xgboost_learning_rate,
            config.xgboost_subsample,
            config.xgboost_colsample_bytree,
            config.xgboost_reg_lambda,
            getattr(config, "xgboost_min_child_weight", 1.0),
            getattr(config, "xgboost_gamma", 0.0),
        )
    raise ValueError(f"Unsupported model_type: {config.model_type}")

from __future__ import annotations

from typing import Any

import numpy as np
from torch import nn

from .cnn1d import CNN1D
from .dscarnet import DSCARNet1D
from .knn import build_knn
from .mlp import MLPBaseline
from .random_forest import build_random_forest
from .svm import build_svm
from .transformer import Transformer1D
from .unet1d import UNet1D
from .xgboost import build_xgboost


MODEL_ALIASES = {
    "1d-cnn": "cnn1d",
    "1dcnn": "cnn1d",
    "cnn1d": "cnn1d",
    "mlp": "mlp",
    "mlp_baseline": "mlp",
    "transformer": "transformer",
    "transformer_encoder": "transformer",
    "unet": "unet1d",
    "unet1d": "unet1d",
    "dscarnet": "dscarnet",
    "dscar_net": "dscarnet",
    "knn": "knn",
    "k-nearest-neighbors": "knn",
    "random_forest": "random_forest",
    "random-forest": "random_forest",
    "rf": "random_forest",
    "svm": "svm",
    "support_vector_machine": "svm",
    "xgboost": "xgboost",
    "xgb": "xgboost",
}

DEEP_MODEL_TYPES = {"cnn1d", "mlp", "transformer", "unet1d", "dscarnet"}
TRADITIONAL_MODEL_TYPES = {"knn", "random_forest", "svm", "xgboost"}


def canonical_model_type(model_type: str) -> str:
    key = str(model_type or "cnn1d").strip().lower()
    return MODEL_ALIASES.get(key, key)


def model_family(model_type: str) -> str:
    return "traditional_ml" if canonical_model_type(model_type) in TRADITIONAL_MODEL_TYPES else "deep_learning"


def build_deep_model(config: Any, input_length: int, class_count: int, sample_count: int) -> nn.Module:
    model_type = canonical_model_type(config.model_type)
    if model_type == "cnn1d":
        return CNN1D(input_length, class_count, sample_count, config.dropout, config.hidden_size)
    if model_type == "mlp":
        return MLPBaseline(input_length, class_count, config.dropout, max(config.hidden_size, 32))
    if model_type == "transformer":
        return Transformer1D(input_length, class_count, config.dropout, max(config.hidden_size, 16), config.transformer_heads)
    if model_type == "unet1d":
        return UNet1D(input_length, class_count, config.dropout, max(config.hidden_size, 8), config.unet_depth)
    if model_type == "dscarnet":
        return DSCARNet1D(input_length, class_count, config.dropout, max(config.hidden_size, 16), config.dscarnet_inception_blocks)
    raise ValueError(f"Unsupported model_type: {config.model_type}")


def parse_optional_int(value: Any) -> int | None:
    if value in {None, "", "none", "None"}:
        return None
    return int(value)


def build_traditional_model(config: Any, y: np.ndarray, class_count: int) -> Any:
    model_type = canonical_model_type(config.model_type)
    class_weight = "balanced" if config.class_balance == "class_weight" else None
    if model_type == "knn":
        n_neighbors = min(max(1, int(config.knn_n_neighbors)), max(1, len(y)))
        return build_knn(n_neighbors, config.knn_weights, config.knn_metric, config.knn_p)
    if model_type == "random_forest":
        return build_random_forest(
            config.random_forest_n_estimators,
            parse_optional_int(config.random_forest_max_depth),
            config.random_forest_min_samples_leaf,
            class_weight,
            config.seed,
        )
    if model_type == "svm":
        return build_svm(config.svm_c, config.svm_gamma, class_weight, config.seed)
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
        )
    raise ValueError(f"Unsupported model_type: {config.model_type}")

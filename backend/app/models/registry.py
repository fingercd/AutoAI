"""分类模型 ID、别名、能力集合和构造器的权威注册表。

`TARGET_MODEL_TYPES` 是 15 项能力目录，`TRADITIONAL_MODEL_TYPES` 与
`DEEP_MODEL_TYPES` 是当前可训练集合。可选 Mamba 依赖不可用时必须抛出明确错误，
不能静默换成近似模型；旧模型类仅用于兼容读取，不进入 v2 新 Run。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from torch import nn

from .cnn1d import CNN1DDocumentV2
from .cnn_se1d import CNNSE1DDocumentV2
from .cnn_transformer1d import CNNTransformer1D
from .dscarnet import dual_dscarnet, single_dscarnet
from .inception1d import Inception1DDocumentV2
from .pls_da import build_pls_da
from .logistic_regression import build_logistic_regression
from .pca_lda import build_pca_lda
from .pca_mlp import PCAMLPClassifier
from .profiles import build_dscarnet_profile, build_model_profile
from .random_forest import build_random_forest
from .resnet1d import ResNet1DDocumentV2
from .svm import build_svm
from .tcn1d import TCN1DDocumentV2
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
    "transformer": "cnn_transformer1d",
    "transformer1d": "cnn_transformer1d",
    "1d-transformer": "cnn_transformer1d",
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
DEEP_MODEL_TYPES = {
    "pca_mlp",
    "cnn1d",
    "cnn1d_se",
    "resnet1d",
    "inception1d",
    "tcn1d",
    "cnn_transformer1d",
    "dscarnet",
}
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
    """目标目录中存在、但当前依赖或版本尚不能构造的模型。"""
    pass


def canonical_model_type(model_type: str) -> str:
    """解析别名、拒绝回归/退役模型，并返回 v2 规范模型 ID。"""
    key = str(model_type or "cnn1d").strip().lower()
    if key in RETIRED_OR_REGRESSION_MODEL_TYPES:
        # “10 类模型”是旧客户端依赖的错误文本，契约测试暂时保持原样；实际
        # 能力集合必须读取 TARGET_MODEL_TYPES/SUPPORTED_MODEL_TYPES。
        raise ValueError("当前仅支持分类任务的 10 类模型；KNN/MLP/UNet 已移除，PLSR/SVR 是回归变体暂不启用")
    canonical = MODEL_ALIASES.get(key, key)
    if canonical in TARGET_MODEL_TYPES and canonical not in SUPPORTED_MODEL_TYPES:
        raise ModelNotImplementedForVersion(f"模型 {canonical} 尚未在 docx-classification-v2 实现")
    if canonical not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"当前仅支持分类任务的 10 类模型，不支持: {model_type}")
    return canonical


def model_family(model_type: str) -> str:
    """返回状态与前端使用的 traditional_ml/deep_learning 家族名。"""
    return "traditional_ml" if canonical_model_type(model_type) in TRADITIONAL_MODEL_TYPES else "deep_learning"


def build_deep_model(
    config: Any,
    input_length: int,
    class_count: int,
    sample_count: int,
    *,
    x_train: np.ndarray | None = None,
) -> nn.Module:
    """按规范 ID、当前折 N/L profile 和二分类输出契约构造深度模型。"""
    model_type = canonical_model_type(config.model_type)
    output_dim = 1 if int(class_count) == 2 else int(class_count)
    if model_type == "pca_mlp":
        if x_train is None:
            raise ValueError("PCA-MLP 必须提供当前折训练集用于拟合 PCA")
        train_values = np.asarray(x_train, dtype=np.float32)
        profile = build_model_profile(
            "pca_mlp",
            train_sample_count=len(train_values),
            feature_count=train_values.shape[1],
        )
        pca = PCA(n_components=profile.values["pca_components"], random_state=int(config.seed))
        pca.fit(train_values)
        model = PCAMLPClassifier(
            pca.mean_,
            pca.components_,
            profile.values["hidden_sizes"],
            profile.dropout,
            output_dim,
        )
        model.pca_model = pca
        model.pca_metadata = {
            "components": int(profile.values["pca_components"]),
            "fit_scope": "train",
            "original_feature_count": int(train_values.shape[1]),
            "train_sample_count": int(len(train_values)),
            "hidden_sizes": list(profile.values["hidden_sizes"]),
            "dropout": float(profile.dropout),
        }
        return model
    profile = build_model_profile(
        model_type,
        train_sample_count=sample_count,
        feature_count=input_length,
    )
    values = profile.values
    if model_type == "cnn1d":
        return CNN1DDocumentV2(
            input_length=input_length,
            class_count=output_dim,
            sample_count=sample_count,
            profile=profile,
        )
    if model_type == "cnn1d_se":
        return CNNSE1DDocumentV2(
            input_length=input_length,
            class_count=output_dim,
            sample_count=sample_count,
            profile=profile,
        )
    if model_type == "resnet1d":
        return ResNet1DDocumentV2(
            input_length=input_length,
            class_count=output_dim,
            sample_count=sample_count,
            profile=profile,
        )
    if model_type == "inception1d":
        return Inception1DDocumentV2(
            input_length=input_length,
            class_count=output_dim,
            sample_count=sample_count,
            profile=profile,
        )
    if model_type == "tcn1d":
        return TCN1DDocumentV2(
            input_length=input_length,
            class_count=output_dim,
            sample_count=sample_count,
            profile=profile,
        )
    if model_type == "cnn_transformer1d":
        return CNNTransformer1D(
            input_length=input_length,
            class_count=output_dim,
            dropout=profile.dropout,
            hidden_size=int(values["d_model"]),
            transformer_heads=int(values["heads"]),
            transformer_layers=int(values["layers"]),
            conv_channels=tuple(int(item) for item in values["conv_channels"]),
            conv_kernels=tuple(int(item) for item in values["kernels"]),
            pool_sizes=tuple(int(item) for item in values["pools"]),
            dim_feedforward=int(values["ffn"]),
        )
    if model_type == "dscarnet":
        raise ValueError("DSCARNet requires 2D AggMap SAR/CAR inputs; use build_dscarnet_model instead")
    raise ValueError(f"Unsupported model_type: {config.model_type}")


def build_dscarnet_model(
    config: Any,
    input_shape1: tuple[int, ...] | None,
    input_shape2: tuple[int, ...] | None,
    class_count: int,
) -> nn.Module:
    """按 sar/car/dual 模式和映射张量形状构造二维 DSCARNet。"""
    output_dim = 1 if int(class_count) == 2 else int(class_count)
    mode = str(getattr(config, "dscarnet_input_mode", "dual") or "dual").lower()
    profile = build_dscarnet_profile(
        train_sample_count=int(getattr(config, "resolved_train_sample_count", 100)),
        feature_count=int(getattr(config, "resolved_feature_count", 1000)),
    )
    common = {
        "filter_number": int(profile["filter_number"]),
        "n_outputs": output_dim,
        "conv1_kernel_size": int(profile["conv1_kernel_size"]),
        "n_inception": int(profile["n_inception"]),
        "dense_layers": tuple(int(item) for item in profile["dense_layers"]),
        "last_avf": None,
    }
    if mode == "sar":
        if input_shape1 is None:
            raise ValueError("SAR 模式缺少 SAR 输入形状")
        return single_dscarnet(input_shape1, **common)
    if mode == "car":
        if input_shape2 is None:
            raise ValueError("CAR 模式缺少 CAR 输入形状")
        return single_dscarnet(input_shape2, **common)
    if mode != "dual" or input_shape1 is None or input_shape2 is None:
        raise ValueError("DSCARNet dual 模式需要 SAR 和 CAR 输入形状")
    return dual_dscarnet(
        input_shape1,
        input_shape2,
        **common,
    )


def parse_optional_int(value: Any) -> int | None:
    """把表单/JSON 中的空值或 none 文本归一化为 None。"""
    if value in {None, "", "none", "None"}:
        return None
    return int(value)


def build_traditional_model(config: Any, y: np.ndarray, class_count: int) -> Any:
    """根据已锁定 TrainConfig 构造一个传统分类模型实例。"""
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
            getattr(config, "random_forest_oob_score", False),
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

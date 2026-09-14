"""分类模型 ID、别名、能力集合和构造器的权威注册表。

【模块定位】
本文件是 backend.app.models 包的“总目录 + 装配车间”：
- 上层训练服务在收到用户选择的 model_type 后，先调用 canonical_model_type
  把各种历史别名（如 "rf"、"1d-cnn"、"transformer"）归一化为 v2 规范模型 ID；
- 再按模型家族分别调用 build_traditional_model（sklearn 系）或
  build_deep_model（PyTorch 系）真正实例化模型。

【关键常量】
`TARGET_MODEL_TYPES` 是 15 项能力目录（对外宣传“支持哪些模型”），
`TRADITIONAL_MODEL_TYPES` 与 `DEEP_MODEL_TYPES` 是当前实际可训练集合（14 项：
`cnn_mamba1d` 因 mamba-ssm 依赖不可用，只出现在能力目录中并标记 available=false）。
可选 Mamba 依赖不可用时必须抛出明确错误，不能静默换成近似模型；
旧模型类（KNN/MLP/UNet 等）仅用于兼容读取历史 Run，不进入 v2 新 Run。

【协作模块】
- .profiles：按当前训练折的样本数 N / 特征长度 L 分档，给出网络宽度、卷积核、
  dropout 等超参数，本模块的构造器据此装配模型；
- 各模型实现文件（cnn1d.py、resnet1d.py、pls_da.py 等）：
  真正的 nn.Module / sklearn 封装，本模块只做“选型和参数转发”，不写训练逻辑。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from torch import nn

from .cnn1d import CNN1DDocumentV2
from .cnn_se1d import CNNSE1DDocumentV2
from .cnn_transformer1d import CNNTransformer1D
from .inception1d import Inception1DDocumentV2
from .pls_da import build_pls_da
from .logistic_regression import build_logistic_regression
from .pca_lda import build_pca_lda
from .pca_mlp import PCAMLPClassifier
from .profiles import build_model_profile
from .random_forest import build_random_forest
from .resnet1d import ResNet1DDocumentV2
from .svm import build_svm
from .tcn1d import TCN1DDocumentV2
from .xgboost import build_xgboost


# Public exports retained for callers of the historical registry module.
from ..model_catalog import (
    MODEL_ALIASES, RETIRED_OR_REGRESSION_MODEL_TYPES, TARGET_MODEL_TYPES,
    TARGET_DEEP_MODEL_TYPES, TARGET_TRADITIONAL_MODEL_TYPES,
    DEEP_MODEL_TYPES, TRADITIONAL_MODEL_TYPES, SUPPORTED_MODEL_TYPES,
    ARCHITECTURE_VERSION, ModelNotImplementedForVersion, canonical_model_type,
)


def model_family(model_type: str) -> str:
    """返回状态与前端使用的 traditional_ml/deep_learning 家族名。

    先经 canonical_model_type 归一化（因此别名和非法 ID 的行为与训练入口一致），
    再按 TRADITIONAL_MODEL_TYPES 划分；不在传统集合里的一律视为 deep_learning。
    """
    return "traditional_ml" if canonical_model_type(model_type) in TRADITIONAL_MODEL_TYPES else "deep_learning"


def build_deep_model(
    config: Any,
    input_length: int,
    class_count: int,
    sample_count: int,
    *,
    x_train: np.ndarray | None = None,
) -> nn.Module:
    """按规范 ID、当前折 N/L profile 和二分类输出契约构造深度模型。

    参数：
        config: 已锁定的 TrainConfig（只需能读到 model_type、seed 等属性）。
        input_length: 特征长度 L（宽表特征列数）。
        class_count: 类别数。
        sample_count: 当前训练折样本数 N。
        x_train: 仅 pca_mlp 需要——PCA 必须在当前折训练集上就地拟合，
            避免验证/测试信息泄漏进标准化与降维参数。

    返回：nn.Module。二分类时 output_dim=1（配合 BCEWithLogits），多分类为 class_count。
    """
    model_type = canonical_model_type(config.model_type)
    output_dim = 1 if int(class_count) == 2 else int(class_count)
    # 二分类输出契约：2 类时 output_dim=1（单个 logit + BCE），否则为 class_count（CE）。
    if model_type == "pca_mlp":
        # PCA-MLP 特例：必须先拿到本折训练特征拟合 PCA，再把 mean_/components_
        # 固化进网络第一层（冻结的线性降维），元数据记 fit_scope="train" 以便审计。
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
    raise ValueError(f"Unsupported model_type: {config.model_type}")




def parse_optional_int(value: Any) -> int | None:
    """把表单/JSON 中的空值或 none 文本归一化为 None。

    前端表单的可选整数（如 random_forest_max_depth）可能提交空字符串或 "none"，
    这些在 sklearn 里都等价于“不限制”，因此先归一化为 None 再 int() 转换。
    """
    if value in {None, "", "none", "None"}:
        return None
    return int(value)


def build_traditional_model(config: Any, y: np.ndarray, class_count: int) -> Any:
    """根据已锁定 TrainConfig 构造一个传统分类模型实例。

    class_balance == "class_weight" 时向支持的模型传 class_weight="balanced"
    （类别不均衡补偿）；xgboost 例外，它接收完整 y 与 class_count 自行计算
    样本权重。getattr 带默认值的字段是为了兼容缺少新字段的旧配置。
    """
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

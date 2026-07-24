"""分类模型 ID、别名、能力集合和构造器的权威注册表。

【模块定位】
本文件是 backend.app.models 包的“总目录 + 装配车间”：
- 上层训练服务在收到用户选择的 model_type 后，先调用 canonical_model_type
  把各种历史别名（如 "rf"、"1d-cnn"、"transformer"）归一化为 v2 规范模型 ID；
- 再按模型家族分别调用 build_traditional_model（sklearn 系）或
  build_deep_model / build_dscarnet_model（PyTorch 系）真正实例化模型。

【关键常量】
`TARGET_MODEL_TYPES` 是 15 项能力目录（对外宣传“支持哪些模型”），
`TRADITIONAL_MODEL_TYPES` 与 `DEEP_MODEL_TYPES` 是当前实际可训练集合（14 项：
`cnn_mamba1d` 因 mamba-ssm 依赖不可用，只出现在能力目录中并标记 available=false）。
可选 Mamba 依赖不可用时必须抛出明确错误，不能静默换成近似模型；
旧模型类（KNN/MLP/UNet 等）仅用于兼容读取历史 Run，不进入 v2 新 Run。

【协作模块】
- .profiles：按当前训练折的样本数 N / 特征长度 L 分档，给出网络宽度、卷积核、
  dropout 等超参数，本模块的构造器据此装配模型；
- 各模型实现文件（cnn1d.py、resnet1d.py、dscarnet.py、pls_da.py 等）：
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


# 别名表：key 是用户/历史数据可能出现的各种写法（统一小写后查表），
# value 是 v2 规范模型 ID。保留大量历史别名是为了兼容旧客户端与旧 Run 配置，
# 例如 "transformer1d" 是 "cnn_transformer1d" 的兼容别名。
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

# 已退役或属于回归任务的模型 ID：当前版本只支持分类，这些 ID 一旦被请求，
# canonical_model_type 会抛出带固定文案的 ValueError（见下），而不是静默训练。
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

# 能力目录（15 项）：对外宣称“平台支持哪些模型”的全集，含暂不可训练的 cnn_mamba1d。
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
# 下面的 DEEP/TRADITIONAL/SUPPORTED 是当前实际可训练集合（14 项）：
# 与能力目录的差别就在 cnn_mamba1d——目录里有它但这里刻意排除，
# 由 canonical_model_type 对其抛 ModelNotImplementedForVersion。
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


# 架构版本号：写入 Run 元数据，标记模型结构/输出契约属于 docx-classification-v2，
# 供结果读取端区分新旧格式的 Run。
ARCHITECTURE_VERSION = "docx-classification-v2"


# 专门的异常类型：模型在能力目录（TARGET_MODEL_TYPES）里、但当前环境/版本无法构造
# （目前只有 cnn_mamba1d）。与“完全不支持的模型”用的普通 ValueError 区分开，
# 上层可据此返回 available=false 而不是 400 错误。
class ModelNotImplementedForVersion(ValueError):
    """目标目录中存在、但当前依赖或版本尚不能构造的模型。"""
    pass


def canonical_model_type(model_type: str) -> str:
    """解析别名、拒绝回归/退役模型，并返回 v2 规范模型 ID。

    处理顺序（先拦退役模型，再解析别名，再区分“未实现”与“不支持”）：
    1. 空值默认按 "cnn1d" 处理，统一 strip+lower；
    2. 命中 RETIRED_OR_REGRESSION_MODEL_TYPES 直接抛 ValueError（固定文案，
       契约测试依赖原文，不能改）；
    3. 经查表得到规范 ID 后：在能力目录但不在可训练集合 → ModelNotImplementedForVersion；
       连能力目录都不在 → 普通 ValueError。
    """
    # 空 model_type 回落到 "cnn1d"：历史默认模型，保证旧调用不传参也能工作。
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
    注意 dscarnet 不走这里：它需要 2D AggMap 输入，必须用 build_dscarnet_model。
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
    if model_type == "dscarnet":
        raise ValueError("DSCARNet requires 2D AggMap SAR/CAR inputs; use build_dscarnet_model instead")
    raise ValueError(f"Unsupported model_type: {config.model_type}")


def build_dscarnet_model(
    config: Any,
    input_shape1: tuple[int, ...] | None,
    input_shape2: tuple[int, ...] | None,
    class_count: int,
) -> nn.Module:
    """按 sar/car/dual 模式和映射张量形状构造二维 DSCARNet。

    与 build_deep_model 分开的原因：DSCARNet 的输入不是 1D 谱，而是 AggMap/PCA
    SAR、CAR 双通路 2D 映射张量，形状只有在映射完成后才知道，因此需要独立的构造入口。
    mode 取自 config.dscarnet_input_mode，默认 "dual"；缺对应输入形状时抛 ValueError。
    """
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

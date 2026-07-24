"""分类模型 registry、profile 和当前模型类的公共导出。

【模块定位】
这是 backend.app.models 包的“门面”：包外代码（训练服务、能力目录接口等）
只应通过 `from backend.app.models import ...` 使用本文件列出的名字，
而不直接深入各个模型实现文件，从而把内部文件组织方式变成可自由调整的细节。

【导出内容分三类】
1. 能力常量与解析函数（来自 .registry）：模型 ID 集合、别名规范化、构造器；
2. 当前可训练的模型类（CNN1D/ResNet/Transformer 等 nn.Module 封装）；
3. profile 相关（来自 .profiles，文件末尾追加导出）：按 N/L 分档的超参数解析。

注意 cnn_mamba1d 只导出 ModelDependencyError / mamba_available 探测函数：
mamba-ssm 依赖缺失时包仍可正常导入，训练入口再以明确错误拒绝该模型。
"""

# ---- 第 1 类导出：registry 的能力常量、别名解析与构造器 ----
from .registry import (
    ARCHITECTURE_VERSION,
    DEEP_MODEL_TYPES,
    TARGET_MODEL_TYPES,
    TARGET_DEEP_MODEL_TYPES,
    TARGET_TRADITIONAL_MODEL_TYPES,
    TRADITIONAL_MODEL_TYPES,
    ModelNotImplementedForVersion,
    build_deep_model,
    build_dscarnet_model,
    build_logistic_regression,
    build_pca_lda,
    build_traditional_model,
    canonical_model_type,
    model_family,
)
# ---- 第 2 类导出：当前可训练的深度模型类 ----
from .pca_mlp import PCAMLPClassifier
from .cnn1d import CNN1DDocumentV2
from .cnn_se1d import CNNSE1DDocumentV2
from .cnn_transformer1d import CNNTransformer1D
from .cnn_mamba1d import ModelDependencyError, mamba_available
from .inception1d import Inception1DDocumentV2
from .resnet1d import ResNet1DDocumentV2
from .tcn1d import TCN1DDocumentV2

# 白名单：只有列在 __all__ 中的名字才会被 `from backend.app.models import *` 导出。
__all__ = [
    "ARCHITECTURE_VERSION",
    "DEEP_MODEL_TYPES",
    "TARGET_MODEL_TYPES",
    "TARGET_DEEP_MODEL_TYPES",
    "TARGET_TRADITIONAL_MODEL_TYPES",
    "TRADITIONAL_MODEL_TYPES",
    "ModelNotImplementedForVersion",
    "build_deep_model",
    "build_dscarnet_model",
    "build_logistic_regression",
    "build_pca_lda",
    "PCAMLPClassifier",
    "CNN1DDocumentV2",
    "CNNSE1DDocumentV2",
    "CNNTransformer1D",
    "ModelDependencyError",
    "mamba_available",
    "Inception1DDocumentV2",
    "ResNet1DDocumentV2",
    "TCN1DDocumentV2",
    "build_traditional_model",
    "canonical_model_type",
    "model_family",
]

# ---- 第 3 类导出：profile 分档（追加到 __all__ 末尾）----
from .profiles import ModelProfile, build_model_profile, model_range_warnings

__all__ += ["ModelProfile", "build_model_profile", "model_range_warnings"]

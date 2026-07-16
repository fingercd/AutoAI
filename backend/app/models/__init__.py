"""分类模型 registry、profile 和当前模型类的公共导出。"""

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
from .pca_mlp import PCAMLPClassifier
from .cnn1d import CNN1DDocumentV2
from .cnn_se1d import CNNSE1DDocumentV2
from .cnn_transformer1d import CNNTransformer1D
from .cnn_mamba1d import ModelDependencyError, mamba_available
from .inception1d import Inception1DDocumentV2
from .resnet1d import ResNet1DDocumentV2
from .tcn1d import TCN1DDocumentV2

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

from .profiles import ModelProfile, build_model_profile, model_range_warnings

__all__ += ["ModelProfile", "build_model_profile", "model_range_warnings"]

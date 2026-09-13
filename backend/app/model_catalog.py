"""Lightweight model identity declarations; no training runtime imports."""
from __future__ import annotations
from dataclasses import dataclass
import importlib.util
from pathlib import Path

ARCHITECTURE_VERSION = "docx-classification-v2"

@dataclass(frozen=True)
class ModelDeclaration:
    id: str
    display_name: str
    family: str
    execution_family: str
    module: str
    aliases: tuple[str, ...]
    dependencies: tuple[str, ...]
    implemented: bool = True
    legacy_agent: bool = False

MODEL_DECLARATIONS = (
    ModelDeclaration('pls_da', 'PLS-DA', 'traditional_ml', 'traditional_ml', 'backend.app.models.pls_da', ('pls', 'pls-da', 'pls_da'), ('numpy', 'scipy', 'sklearn'), True, False),
    ModelDeclaration('pca_lda', 'PCA-LDA', 'traditional_ml', 'traditional_ml', 'backend.app.models.pca_lda', ('pca_lda',), ('numpy', 'scipy', 'sklearn'), True, False),
    ModelDeclaration('logistic_regression', 'Logistic Regression', 'traditional_ml', 'traditional_ml', 'backend.app.models.logistic_regression', ('logistic_regression', 'logistic-regression', 'logreg'), ('numpy', 'scipy', 'sklearn'), True, True),
    ModelDeclaration('svm', 'SVM', 'traditional_ml', 'traditional_ml', 'backend.app.models.svm', ('svm', 'support_vector_machine'), ('numpy', 'scipy', 'sklearn'), True, True),
    ModelDeclaration('random_forest', 'Random Forest', 'traditional_ml', 'traditional_ml', 'backend.app.models.random_forest', ('random_forest', 'random-forest', 'rf'), ('numpy', 'scipy', 'sklearn'), True, True),
    ModelDeclaration('xgboost', 'XGBoost', 'traditional_ml', 'traditional_ml', 'backend.app.models.xgboost', ('xgboost', 'xgb'), ('numpy', 'scipy', 'sklearn', 'xgboost'), True, False),
    ModelDeclaration('pca_mlp', 'PCA-MLP', 'basic_deep', 'deep_learning', 'backend.app.models.pca_mlp', ('pca_mlp',), ('numpy', 'scipy', 'sklearn', 'torch'), True, False),
    ModelDeclaration('cnn1d', '1D CNN', 'basic_deep', 'deep_learning', 'backend.app.models.cnn1d', ('1d-cnn', '1dcnn', 'cnn1d'), ('numpy', 'scipy', 'sklearn', 'torch'), True, False),
    ModelDeclaration('cnn1d_se', '1D CNN-SE', 'convolutional', 'deep_learning', 'backend.app.models.cnn_se1d', ('cnn1d_se', 'cnn-se'), ('numpy', 'scipy', 'sklearn', 'torch'), True, False),
    ModelDeclaration('resnet1d', '1D ResNet', 'convolutional', 'deep_learning', 'backend.app.models.resnet1d', ('resnet1d', '1d-resnet'), ('numpy', 'scipy', 'sklearn', 'torch'), True, False),
    ModelDeclaration('inception1d', '1D Inception', 'convolutional', 'deep_learning', 'backend.app.models.inception1d', ('inception1d', '1d-inception'), ('numpy', 'scipy', 'sklearn', 'torch'), True, False),
    ModelDeclaration('tcn1d', '1D TCN', 'convolutional', 'deep_learning', 'backend.app.models.tcn1d', ('tcn1d', '1d-tcn'), ('numpy', 'scipy', 'sklearn', 'torch'), True, False),
    ModelDeclaration('cnn_transformer1d', 'CNN-Transformer', 'long_range', 'deep_learning', 'backend.app.models.cnn_transformer1d', ('transformer', 'transformer1d', '1d-transformer', 'cnn_transformer1d'), ('numpy', 'scipy', 'sklearn', 'torch'), True, False),
    ModelDeclaration('cnn_mamba1d', 'CNN-Mamba', 'long_range', 'deep_learning', 'backend.app.models.cnn_mamba1d', ('cnn_mamba1d',), ('numpy', 'scipy', 'sklearn', 'torch', 'mamba_ssm'), False, False),
    ModelDeclaration('dscarnet', 'DSCARNet', 'two_dimensional_mapping', 'deep_learning', 'backend.app.models.dscarnet', ('dscarnet', 'dscar_net'), ('numpy', 'scipy', 'sklearn', 'torch', 'aggmap'), True, False),
)
MODELS_BY_ID = {m.id: m for m in MODEL_DECLARATIONS}
MODEL_ALIASES = {alias: m.id for m in MODEL_DECLARATIONS for alias in m.aliases}
LEGACY_AGENT_MODELS = tuple(m.id for m in MODEL_DECLARATIONS if m.legacy_agent)
TARGET_MODEL_TYPES = set(MODELS_BY_ID)
TARGET_TRADITIONAL_MODEL_TYPES = {m.id for m in MODEL_DECLARATIONS if m.execution_family == "traditional_ml"}
TARGET_DEEP_MODEL_TYPES = TARGET_MODEL_TYPES - TARGET_TRADITIONAL_MODEL_TYPES
SUPPORTED_MODEL_TYPES = {m.id for m in MODEL_DECLARATIONS if m.implemented}
TRADITIONAL_MODEL_TYPES = TARGET_TRADITIONAL_MODEL_TYPES & SUPPORTED_MODEL_TYPES
DEEP_MODEL_TYPES = TARGET_DEEP_MODEL_TYPES & SUPPORTED_MODEL_TYPES
RETIRED_OR_REGRESSION_MODEL_TYPES = {'mlp_baseline', 'unet1d', 'k-nearest-neighbors', 'mlp', 'knn', 'plsr', 'svr', 'unet'}

class ModelNotImplementedForVersion(ValueError):
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

def model_availability(model_id: str) -> tuple[bool, str | None]:
    model = MODELS_BY_ID[model_id]
    if not model.implemented:
        return False, "not_implemented_for_version"
    module_file = Path(__file__).parent / "models" / (model.module.rsplit(".", 1)[-1] + ".py")
    if not module_file.is_file():
        return False, "implementation_missing"
    for name in model.dependencies:
        try:
            found = importlib.util.find_spec(name)
        except (ImportError, ValueError, AttributeError):
            found = None
        if found is None:
            return False, "dependency_missing_" + name
    return True, None

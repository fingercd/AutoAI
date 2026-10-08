"""Single public capability switch for the versioned Word training scheme."""

# 经典新版训练已接通；未声明版本的兼容 API 与历史结果保持原语义。
FEATURE_ENGINEERING_ENABLED = True


def supported_feature_schemes(model_type):
    """Versioned schemes; PCA classifiers already fit their own fold-local PCA."""
    from .feature_engineering import MODELS, SCHEMES
    if model_type not in MODELS:
        return []
    return [key for key, _ in SCHEMES
            if model_type not in {'cnn1d', 'pca_lda', 'pca_svm'} or not key.startswith('pca_')]


def unsupported_feature_reason(model_type):
    if model_type == 'cnn1d':
        return 'CNN 保留光谱顺序，不使用 PCA 输入'
    return '该模型已内置 PCA，不叠加外层 PCA；请选择全特征或相邻点合并'


def validate_training_options(profile, scheme, model_type):
    from .feature_engineering import SCHEMES
    if profile not in ('quick', 'full'):
        raise ValueError('training_profile 必须是 quick 或 full')
    if scheme not in dict(SCHEMES):
        raise ValueError('未知特征处理方案')
    if scheme not in supported_feature_schemes(model_type):
        raise ValueError(unsupported_feature_reason(model_type))


def training_scheme():
    """Public capability; the classic UI must not invent its own enable switch."""
    from dataclasses import asdict
    from .classification_policy import DEEP_TRAINING_DEFAULTS
    from .feature_engineering import MODELS, SCHEMES
    return {'enabled': FEATURE_ENGINEERING_ENABLED, 'version': 'word-0904',
            'traditional_scheme_count': 7, 'cnn_scheme_count': 4,
            'default_profile': 'quick', 'default_feature_scheme': 'full',
            'profiles': [{'id': 'quick', 'name': '快速训练（单方案、一次验证）'},
                         {'id': 'full', 'name': '完整比较（多方案、五次验证）'}],
            'feature_schemes': [{'id': key, 'name': name, 'cnn_supported': not key.startswith('pca_')}
                                for key, name in SCHEMES],
            'quick_candidate_count': 3,
            'supported_feature_schemes': {model: supported_feature_schemes(model) for model in sorted(MODELS)},
            'defaults': asdict(DEEP_TRAINING_DEFAULTS)}

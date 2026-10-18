"""Single public capability switch for the versioned Word training scheme."""

# 经典新版训练已接通；未声明版本的兼容 API 与历史结果保持原语义。
FEATURE_ENGINEERING_ENABLED = True


def inapplicable_reason(model_type, scheme):
    from .feature_engineering import DEEP_MODELS
    if scheme.startswith('pca_'):
        if model_type in DEEP_MODELS:
            return 'CNN 不支持 PCA，请选择全特征或相邻点合并'
        if model_type in {'pca_lda', 'pca_svm'}:
            return '该模型内置 PCA，不重复叠加外层 PCA，请选择全特征或相邻点合并'
    return None


def validate_training_options(profile, scheme, model_type):
    from .feature_engineering import ALL_SCHEMES
    if profile not in ('quick', 'full'):
        raise ValueError('training_profile 必须是 quick 或 full')
    if scheme not in dict(ALL_SCHEMES):
        raise ValueError('未知特征处理方案')
    reason = inapplicable_reason(model_type, scheme)
    if profile == 'quick' and reason:
        raise ValueError(reason)


def training_scheme():
    """Public capability; the classic UI must not invent its own enable switch."""
    from dataclasses import asdict
    from .classification_policy import DEEP_TRAINING_DEFAULTS
    from .feature_engineering import ALL_SCHEMES, STUDENT_SCHEMES
    from .feature_engineering import MODELS
    return {'enabled': FEATURE_ENGINEERING_ENABLED, 'version': 'word-0904',
            'traditional_scheme_count': 7, 'cnn_scheme_count': 4,
            'default_profile': 'quick', 'default_feature_scheme': 'full',
            'student_default_feature_scheme': 'bin_5',
            'profiles': [{'id': 'quick', 'name': '快速训练（单方案、一次验证）'},
                         {'id': 'full', 'name': '完整比较（多方案、五次验证）'}],
            'feature_schemes': [{'id': key, 'name': name, 'cnn_supported': not key.startswith('pca_'),
                                'student_visible': key in dict(STUDENT_SCHEMES),
                                'unsupported_models': ['cnn1d', 'cnn1d_se', 'resnet1d', 'pca_lda', 'pca_svm'] if key.startswith('pca_') else [],
                                'inapplicable_reasons': {model: reason for model in sorted(MODELS) if (reason := inapplicable_reason(model, key))}} for key, name in ALL_SCHEMES],
            'quick_candidate_count': 3,
            'defaults': asdict(DEEP_TRAINING_DEFAULTS)}

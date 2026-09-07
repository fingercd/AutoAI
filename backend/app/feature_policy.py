"""Single public capability switch for the versioned Word training scheme."""

# 经典新版训练已接通；未声明版本的兼容 API 与历史结果保持原语义。
FEATURE_ENGINEERING_ENABLED = True


def training_scheme():
    """Public capability; the classic UI must not invent its own enable switch."""
    from dataclasses import asdict
    from .classification_policy import DEEP_TRAINING_DEFAULTS
    return {'enabled': FEATURE_ENGINEERING_ENABLED, 'version': 'word-0904',
            'traditional_scheme_count': 7, 'cnn_scheme_count': 4,
            'defaults': asdict(DEEP_TRAINING_DEFAULTS)}

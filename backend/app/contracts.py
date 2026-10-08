"""HTTP 训练请求与内部兼容配置的轻量契约。

身份字段有意不出现在请求模型中；本地模式的 Principal 只能由服务端依赖注入。
`extra='forbid'` 同时阻止旧客户端偷偷传入 owner_id/tenant_id。
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TrainingRunRequest(BaseModel):
    """创建 queued Run 时允许的顶层字段。"""
    model_config = ConfigDict(extra='forbid')

    dataset_id: str | None = None
    data_path: str | None = None
    test_dataset_id: str | None = None
    test_data_path: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    strict_config: bool = False


class TrainingBatchRequest(BaseModel):
    """Multi-model request.  Identity and filesystem paths remain server-owned."""
    model_config = ConfigDict(extra='forbid')

    dataset_id: str | None = None
    data_path: str | None = None
    test_dataset_id: str | None = None
    test_data_path: str | None = None
    model_types: list[str] = Field(min_length=1, max_length=50)
    # 新请求只允许“一模型一个 Run”。仓储/结果投影仍能读取历史 R>1 批次，
    # 但公开创建接口不再接受新的重复训练。
    repeat_count: int = Field(default=1, ge=1, le=1)
    base_seed: int = 42
    config: dict[str, Any] = Field(default_factory=dict)
    strict_config: bool = False


class TrainingPreflightRequest(BaseModel):
    """Dataset-only preview; never enqueues or fits a model."""
    model_config = ConfigDict(extra='forbid')
    task_type: Literal['run', 'batch'] = 'run'
    dataset_id: str
    test_dataset_id: str | None = None
    model_types: list[str] = Field(default_factory=list, max_length=50)
    base_seed: int = 42
    config: dict[str, Any] = Field(default_factory=dict)


class TrainingConfigValidationError(ValueError):
    """训练配置在创建 queued Run 前即可确定的错误。"""


_KNOWN_TRAINING_CONFIG_FIELDS = frozenset(
    {
        'experiment_version',
        'training_profile', 'feature_scheme',
        'epochs', 'batch_size', 'learning_rate', 'weight_decay', 'scheduler_factor',
        'scheduler_patience', 'min_learning_rate', 'seed', 'split_seed', 'model_seed', 'normalization', 'split_mode',
        'split_train', 'split_valid', 'split_test', 'class_balance', 'model_type',
        'early_stopping_patience', 'dropout', 'hidden_size', 'transformer_heads',
        'unet_depth', 'dscarnet_inception_blocks', 'dscarnet_pca_components',
        'dscarnet_cluster_channels', 'dscarnet_input_mode', 'knn_n_neighbors',
        'knn_weights', 'knn_metric', 'knn_p', 'random_forest_n_estimators',
        'random_forest_search_iterations', 'random_forest_max_depth',
        'random_forest_min_samples_leaf', 'svm_c', 'svm_gamma', 'xgboost_n_estimators',
        'xgboost_max_depth', 'xgboost_learning_rate', 'xgboost_subsample',
        'xgboost_colsample_bytree', 'xgboost_reg_lambda', 'pls_components',
        'pca_components', 'logistic_c', 'svm_kernel', 'random_forest_max_features',
        'random_forest_oob_score', 'xgboost_min_child_weight', 'xgboost_gamma',
        'feature_selection_enabled', 'feature_window_count', 'feature_top_k',
        'feature_n_repeats', 'feature_eval_split', 'spls_components', 'spls_keepx',
        'logistic_l1_ratio',
    }
)

_MODEL_ALIASES = {
    'pls': 'pls_da', 'pls-da': 'pls_da', 'pls_da': 'pls_da',
    'spls_da': 'spls_da', 'spls-da': 'spls_da',
    'pca_lda': 'pca_lda',
    'logistic_regression': 'logistic_regression', 'logistic-regression': 'logistic_regression',
    'logreg': 'logistic_regression',
    'svm': 'svm', 'support_vector_machine': 'svm',
    'pca_svm': 'pca_svm', 'pca-svm': 'pca_svm',
    'random_forest': 'random_forest', 'random-forest': 'random_forest', 'rf': 'random_forest',
    'xgboost': 'xgboost', 'xgb': 'xgboost',
    'pca_mlp': 'pca_mlp',
    '1d-cnn': 'cnn1d', '1dcnn': 'cnn1d', 'cnn1d': 'cnn1d',
    'cnn1d_se': 'cnn1d_se', 'cnn-se': 'cnn1d_se',
    'resnet1d': 'resnet1d', '1d-resnet': 'resnet1d',
    'inception1d': 'inception1d', '1d-inception': 'inception1d',
    'tcn1d': 'tcn1d', '1d-tcn': 'tcn1d',
    'transformer': 'cnn_transformer1d', 'transformer1d': 'cnn_transformer1d',
    '1d-transformer': 'cnn_transformer1d', 'cnn_transformer1d': 'cnn_transformer1d',
    'cnn_mamba1d': 'cnn_mamba1d',
    'dscarnet': 'dscarnet', 'dscar_net': 'dscarnet',
}


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise TrainingConfigValidationError(f'{field} 必须是数值')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TrainingConfigValidationError(f'{field} 必须是数值') from exc
    if not math.isfinite(number):
        raise TrainingConfigValidationError(f'{field} 必须是有限数值')
    return number


class TrainingSpec:
    """把旧字典配置包成稳定的内部访问接口。"""
    def __init__(self, values: dict[str, Any], *, warnings: tuple[str, ...] = ()) -> None:
        self.values = dict(values)
        self.warnings = warnings

    @classmethod
    def from_strict(cls, values: dict[str, Any]) -> 'TrainingSpec':
        # The versioned scheme owns candidate parameters and CNN scheduling.
        # Accept only controls that actually affect that scheme.
        allowed = {'experiment_version', 'training_profile', 'feature_scheme',
                   'model_type', 'normalization', 'split_mode', 'split_train',
                   'split_valid', 'split_test', 'seed', 'split_seed', 'model_seed', 'class_balance'}
        unsupported = sorted(set(values) - allowed)
        if unsupported:
            raise TrainingConfigValidationError('严格模式不支持这些参数：' + ', '.join(unsupported))
        if values.get('experiment_version') != 'word-0904':
            raise TrainingConfigValidationError('严格模式必须明确指定 experiment_version=word-0904')
        if not values.get('model_type'):
            raise TrainingConfigValidationError('严格模式必须明确选择 model_type')
        if values.get('class_balance', 'none') != 'none':
            raise TrainingConfigValidationError('0904 固定策略不支持覆盖 class_balance')
        for key in ('split_train', 'split_valid', 'split_test'):
            if key in values:
                number = _finite_number(values[key], field=key)
                if not number.is_integer():
                    raise TrainingConfigValidationError(f'{key} 必须是整数')
        if values.get('split_mode') not in (None, 'stratified_holdout', 'leave_one_sample_id_cv',
                                             'external_test_holdout', 'leave_one_sample_id_cv_with_external_test'):
            raise TrainingConfigValidationError('严格模式必须使用四种明确的评估模式之一')
        if values.get('training_profile') == 'full' and values.get('feature_scheme', 'full') != 'full':
            raise TrainingConfigValidationError('完整比较会运行所有适用方案；feature_scheme 请使用 full')
        return cls(dict(values))

    @classmethod
    def from_legacy(cls, values: dict[str, Any] | None) -> 'TrainingSpec':
        raw = dict(values or {})
        unknown = sorted(str(key) for key in raw if key not in _KNOWN_TRAINING_CONFIG_FIELDS)
        clean = {key: value for key, value in raw.items() if key in _KNOWN_TRAINING_CONFIG_FIELDS}
        warnings = (
            (f"已忽略未知训练参数：{', '.join(unknown)}",)
            if unknown
            else ()
        )
        return cls(clean, warnings=warnings)

    def validated(self, *, has_external_test: bool) -> 'TrainingSpec':
        """验证不依赖实际数据内容的关键约束，并规范化兼容别名。"""

        values = dict(self.values)
        raw_model = str(values.get('model_type') or 'cnn1d').strip().lower()
        model_type = _MODEL_ALIASES.get(raw_model)
        if model_type is None:
            raise TrainingConfigValidationError(f'不支持的分类模型：{raw_model}')
        values['model_type'] = model_type
        if values.get('experiment_version') not in (None, 'word-0904'):
            raise TrainingConfigValidationError('未知建模方案版本')
        if values.get('experiment_version') == 'word-0904':
            from . import feature_policy
            if not feature_policy.FEATURE_ENGINEERING_ENABLED:
                raise TrainingConfigValidationError('特征工程暂时停用，请使用普通训练配置')
            from .feature_engineering import MODELS
            if model_type not in MODELS:
                raise TrainingConfigValidationError('0904 方案仅支持九类公开模型')
            if values.get('normalization', 'zscore') not in {'zscore', 'minmax'}:
                raise TrainingConfigValidationError('0904 方案要求 zscore 或 minmax 特征标准化')
            values.setdefault('training_profile', 'quick')
            values.setdefault('feature_scheme', 'full')
            try:
                feature_policy.validate_training_options(values['training_profile'], values['feature_scheme'], model_type)
            except ValueError as exc:
                raise TrainingConfigValidationError(str(exc)) from exc

        normalization = str(values.get('normalization') or 'zscore').strip().lower()
        if normalization not in {'zscore', 'minmax', 'area', 'none'}:
            raise TrainingConfigValidationError('normalization 必须是 zscore、minmax、area 或 none')
        values['normalization'] = normalization

        class_balance = str(values.get('class_balance') or 'none').strip().lower()
        if class_balance not in {'none', 'class_weight'}:
            raise TrainingConfigValidationError('class_balance 必须是 none 或 class_weight')
        values['class_balance'] = class_balance

        if model_type == 'dscarnet':
            mode = str(values.get('dscarnet_input_mode') or 'dual').strip().lower()
            if mode not in {'sar', 'car', 'dual'}:
                raise TrainingConfigValidationError('dscarnet_input_mode 必须是 sar、car 或 dual')
            values['dscarnet_input_mode'] = mode

        from .classification_policy import resolve_evaluation_policy

        try:
            policy = resolve_evaluation_policy(values, has_external_test=has_external_test)
        except (TypeError, ValueError) as exc:
            raise TrainingConfigValidationError(str(exc)) from exc
        values.update(
            split_mode=policy.strategy,
            split_train=policy.split_train,
            split_valid=policy.split_valid,
            split_test=policy.split_test,
        )

        positive_integer_fields = {
            'epochs', 'batch_size', 'scheduler_patience', 'early_stopping_patience',
            'hidden_size', 'transformer_heads', 'dscarnet_inception_blocks',
            'dscarnet_pca_components', 'dscarnet_cluster_channels',
            'random_forest_n_estimators', 'random_forest_search_iterations',
            'random_forest_min_samples_leaf', 'xgboost_n_estimators', 'xgboost_max_depth',
            'feature_window_count', 'feature_top_k', 'feature_n_repeats',
            'spls_components', 'spls_keepx',
        }
        for field in positive_integer_fields:
            if field not in values:
                continue
            number = _finite_number(values[field], field=field)
            if number < 1 or not number.is_integer():
                raise TrainingConfigValidationError(f'{field} 必须是大于 0 的整数')
            values[field] = int(number)

        positive_fields = {
            'learning_rate', 'min_learning_rate', 'svm_c', 'xgboost_learning_rate',
            'xgboost_reg_lambda', 'xgboost_min_child_weight', 'logistic_c',
        }
        for field in positive_fields:
            if field in values:
                number = _finite_number(values[field], field=field)
                if number <= 0:
                    raise TrainingConfigValidationError(f'{field} 必须大于 0')
                values[field] = number

        for field in ('xgboost_subsample', 'xgboost_colsample_bytree', 'scheduler_factor'):
            if field in values:
                number = _finite_number(values[field], field=field)
                if not 0 < number <= 1:
                    raise TrainingConfigValidationError(f'{field} 必须大于 0 且不超过 1')
                values[field] = number

        for field in ('weight_decay', 'xgboost_gamma'):
            if field in values:
                number = _finite_number(values[field], field=field)
                if number < 0:
                    raise TrainingConfigValidationError(f'{field} 必须大于等于 0')
                values[field] = number

        for field in ('seed', 'split_seed', 'model_seed'):
            if field in values:
                seed = _finite_number(values[field], field=field)
                if not seed.is_integer():
                    raise TrainingConfigValidationError(f'{field} 必须是整数')
                values[field] = int(seed)

        for field in ('pls_components', 'spls_components', 'spls_keepx', 'pca_components', 'random_forest_max_depth'):
            if field in values and values[field] is not None:
                number = _finite_number(values[field], field=field)
                if number < 1 or not number.is_integer():
                    raise TrainingConfigValidationError(f'{field} 必须是大于 0 的整数或 null')
                values[field] = int(number)

        if 'dropout' in values and values['dropout'] is not None:
            dropout = _finite_number(values['dropout'], field='dropout')
            if not 0 <= dropout < 1:
                raise TrainingConfigValidationError('dropout 必须大于等于 0 且小于 1')
            values['dropout'] = dropout

        if 'logistic_l1_ratio' in values:
            ratio = _finite_number(values['logistic_l1_ratio'], field='logistic_l1_ratio')
            if not 0 <= ratio <= 1:
                raise TrainingConfigValidationError('logistic_l1_ratio 必须在 0 到 1 之间')
            values['logistic_l1_ratio'] = ratio

        if 'svm_gamma' in values and isinstance(values['svm_gamma'], str):
            gamma = values['svm_gamma'].strip().lower()
            if gamma not in {'scale', 'auto'}:
                raise TrainingConfigValidationError('svm_gamma 必须是正数、scale 或 auto')
            values['svm_gamma'] = gamma
        elif 'svm_gamma' in values:
            gamma = _finite_number(values['svm_gamma'], field='svm_gamma')
            if gamma <= 0:
                raise TrainingConfigValidationError('svm_gamma 必须大于 0')
            values['svm_gamma'] = gamma

        return TrainingSpec(values, warnings=self.warnings)

    def to_legacy_dict(self) -> dict[str, Any]:
        return dict(self.values)

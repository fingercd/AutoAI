"""HTTP 训练请求与内部兼容配置的轻量契约。

身份字段有意不出现在请求模型中；本地模式的 Principal 只能由服务端依赖注入。
`extra='forbid'` 同时阻止旧客户端偷偷传入 owner_id/tenant_id。
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TrainingRunRequest(BaseModel):
    """创建 queued Run 时允许的顶层字段。"""
    model_config = ConfigDict(extra='forbid')

    dataset_id: str | None = None
    data_path: str | None = None
    test_dataset_id: str | None = None
    test_data_path: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class TrainingConfigValidationError(ValueError):
    """训练配置在创建 queued Run 前即可确定的错误。"""


_KNOWN_TRAINING_CONFIG_FIELDS = frozenset(
    {
        'epochs', 'batch_size', 'learning_rate', 'weight_decay', 'scheduler_factor',
        'scheduler_patience', 'min_learning_rate', 'seed', 'normalization', 'split_mode',
        'split_train', 'split_valid', 'split_test', 'class_balance', 'model_type',
        'early_stopping_patience', 'dropout', 'hidden_size', 'transformer_heads',
        'unet_depth', 'knn_n_neighbors',
        'knn_weights', 'knn_metric', 'knn_p', 'random_forest_n_estimators',
        'random_forest_search_iterations', 'random_forest_max_depth',
        'random_forest_min_samples_leaf', 'svm_c', 'svm_gamma', 'xgboost_n_estimators',
        'xgboost_max_depth', 'xgboost_learning_rate', 'xgboost_subsample',
        'xgboost_colsample_bytree', 'xgboost_reg_lambda', 'pls_components',
        'pca_components', 'logistic_c', 'svm_kernel', 'random_forest_max_features',
        'random_forest_oob_score', 'xgboost_min_child_weight', 'xgboost_gamma',
        'agent_config_policy_version', 'agent_config_policy_digest',
        'feature_selection_enabled', 'feature_window_count', 'feature_top_k',
        'feature_n_repeats', 'feature_eval_split',
    }
)

from .model_catalog import MODEL_ALIASES as _MODEL_ALIASES, RETIRED_MODEL_ALIASES


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
        if raw_model in RETIRED_MODEL_ALIASES:
            raise TrainingConfigValidationError('model_retired: DSCARNet 已退役，不再接受新训练')
        model_type = _MODEL_ALIASES.get(raw_model)
        if model_type is None:
            raise TrainingConfigValidationError(f'不支持的分类模型：{raw_model}')
        values['model_type'] = model_type

        normalization = str(values.get('normalization') or 'zscore').strip().lower()
        if normalization not in {'zscore', 'minmax', 'area', 'none'}:
            raise TrainingConfigValidationError('normalization 必须是 zscore、minmax、area 或 none')
        values['normalization'] = normalization

        class_balance = str(values.get('class_balance') or 'none').strip().lower()
        if class_balance not in {'none', 'class_weight'}:
            raise TrainingConfigValidationError('class_balance 必须是 none 或 class_weight')
        values['class_balance'] = class_balance


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
            'hidden_size', 'transformer_heads', 'random_forest_n_estimators', 'random_forest_search_iterations',
            'random_forest_min_samples_leaf', 'xgboost_n_estimators', 'xgboost_max_depth',
            'feature_window_count', 'feature_top_k', 'feature_n_repeats',
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

        for field in ('xgboost_subsample', 'xgboost_colsample_bytree'):
            if field in values:
                number = _finite_number(values[field], field=field)
                if not 0 < number <= 1:
                    raise TrainingConfigValidationError(f'{field} 必须大于 0 且不超过 1')
                values[field] = number

        if 'scheduler_factor' in values:
            factor = _finite_number(values['scheduler_factor'], field='scheduler_factor')
            if not 0 < factor < 1:
                raise TrainingConfigValidationError('scheduler_factor must be greater than 0 and less than 1')
            values['scheduler_factor'] = factor

        for field in ('weight_decay', 'xgboost_gamma'):
            if field in values:
                number = _finite_number(values[field], field=field)
                if number < 0:
                    raise TrainingConfigValidationError(f'{field} 必须大于等于 0')
                values[field] = number

        if 'seed' in values:
            seed = _finite_number(values['seed'], field='seed')
            if not seed.is_integer():
                raise TrainingConfigValidationError('seed 必须是整数')
            values['seed'] = int(seed)

        for field in ('pls_components', 'pca_components', 'random_forest_max_depth'):
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

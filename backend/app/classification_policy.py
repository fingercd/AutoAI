"""集中定义分类训练默认值与评估口径。

本模块只负责把请求中的划分选项归一化成不可歧义的策略；它不读取数据，
也不执行划分。训练代码必须以这里返回的策略为准，避免前端、API 和训练器
分别解释比例，尤其要保证 external test 与留一交叉验证不会同时启用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class DeepTrainingDefaults:
    """文档版深度模型共享的、不可变训练默认值。"""
    epochs: int = 200
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    min_learning_rate: float = 1e-6
    early_stopping_patience: int = 20
    seed: int = 42


DEEP_TRAINING_DEFAULTS = DeepTrainingDefaults()


@dataclass(frozen=True)
class EvaluationPolicy:
    """训练器实际执行的评估策略及 train/valid/test 权重。"""
    strategy: str
    split_train: int
    split_valid: int
    split_test: int
    cv_allowed: bool


_CV_MODES = {
    "leave_one_sample_id_cv",
    "leave_one_repeat_index_cv",
    "outer_leave_one_repeat_index_cv",
    "loocv",
    "loo",
}


def resolve_evaluation_policy(
    config_data: Mapping[str, object] | None,
    *,
    has_external_test: bool,
) -> EvaluationPolicy:
    """根据外部测试集和 split_mode 解析唯一合法的评估策略。"""
    config = dict(config_data or {})
    requested = str(config.get("split_mode") or "stratified_holdout").strip().lower()

    if has_external_test:
        if requested in _CV_MODES:
            raise ValueError("已提供独立测试集时不允许开启交叉验证")
        train = int(config.get("split_train", 8))
        valid = int(config.get("split_valid", 2))
        test = int(config.get("split_test", 0))
        if test != 0 or train <= 0 or valid <= 0 or train + valid != 10:
            raise ValueError("独立测试集模式要求主数据训练/验证比例相加必须等于 10，且内部测试比例为 0")
        return EvaluationPolicy("external_test_holdout", train, valid, 0, False)

    if requested in _CV_MODES:
        train = int(config.get("split_train", 8))
        valid = int(config.get("split_valid", 2))
        if train <= 0 or valid <= 0 or train + valid != 10:
            raise ValueError("留一交叉验证要求训练/验证比例相加必须等于 10")
        return EvaluationPolicy("leave_one_sample_id_cv", train, valid, 0, True)

    train = int(config.get("split_train", 8))
    valid = int(config.get("split_valid", 1))
    test = int(config.get("split_test", 1))
    if min(train, valid, test) <= 0 or train + valid + test != 10:
        raise ValueError("分层 holdout 要求训练、验证、测试比例均大于 0 且相加为 10")
    return EvaluationPolicy("stratified_holdout", train, valid, test, True)

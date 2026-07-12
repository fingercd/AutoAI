from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class EvaluationPolicy:
    strategy: str
    split_train: int
    split_valid: int
    split_test: int
    cv_allowed: bool


_CV_MODES = {
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
        return EvaluationPolicy("leave_one_repeat_index_cv", train, valid, 0, True)

    train = int(config.get("split_train", 8))
    valid = int(config.get("split_valid", 1))
    test = int(config.get("split_test", 1))
    if min(train, valid, test) <= 0 or train + valid + test != 10:
        raise ValueError("分层 holdout 要求训练、验证、测试比例均大于 0 且相加为 10")
    return EvaluationPolicy("stratified_holdout", train, valid, test, True)

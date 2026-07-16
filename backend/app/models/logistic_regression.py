"""Logistic Regression 分类基线构造器。"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def build_logistic_regression(
    c: float = 1.0,
    seed: int = 42,
    class_weight: str | None = None,
) -> LogisticRegression:
    """构造固定 lbfgs 求解器和可选类别权重的分类基线。"""
    return LogisticRegression(
        C=float(c),
        class_weight=class_weight,
        max_iter=1000,
        random_state=int(seed),
        solver="lbfgs",
    )

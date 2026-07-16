"""Random Forest 分类器构造器；搜索与 OOB 选优由 training.py 编排。"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def build_random_forest(
    n_estimators: int = 200,
    max_depth: int | None = 3,
    min_samples_leaf: int = 2,
    max_features: str | float = "sqrt",
    class_weight: str | None = None,
    seed: int = 42,
    oob_score: bool = False,
) -> RandomForestClassifier:
    """构造支持 OOB 审计和类别权重的随机森林。"""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=-1,
        bootstrap=True,
        oob_score=oob_score,
    )

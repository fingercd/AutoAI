from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def build_random_forest(
    n_estimators: int = 100,
    max_depth: int | None = 3,
    min_samples_leaf: int = 2,
    max_features: str | float = "sqrt",
    class_weight: str | None = None,
    seed: int = 42,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=-1,
    )

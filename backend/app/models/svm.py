from __future__ import annotations

from sklearn.svm import SVC


def build_svm(
    c: float = 1.0,
    gamma: str | float = 0.03,
    class_weight: str | None = None,
    seed: int = 42,
    kernel: str = "rbf",
) -> SVC:
    if isinstance(gamma, str) and gamma not in {"scale", "auto"}:
        gamma = float(gamma)
    safe_kernel = kernel if kernel in {"linear", "rbf"} else "rbf"
    return SVC(C=c, gamma=gamma, kernel=safe_kernel, probability=True, class_weight=class_weight, random_state=seed)

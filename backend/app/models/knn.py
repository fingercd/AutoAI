from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier


def build_knn(n_neighbors: int = 5, weights: str = "distance", metric: str = "minkowski", p: int = 2) -> KNeighborsClassifier:
    kwargs = {"n_neighbors": max(1, int(n_neighbors)), "weights": weights, "metric": metric}
    if metric == "minkowski":
        kwargs["p"] = max(1, int(p))
    return KNeighborsClassifier(**kwargs)

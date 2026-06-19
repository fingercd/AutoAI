from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier


def build_knn(n_neighbors: int = 5, weights: str = "distance", metric: str = "minkowski", p: int = 2) -> KNeighborsClassifier:
    safe_weights = weights if weights in {"uniform", "distance"} else "distance"
    safe_metric = metric if metric in {"minkowski", "euclidean", "manhattan", "cosine"} else "minkowski"
    kwargs = {"n_neighbors": max(1, int(n_neighbors)), "weights": safe_weights, "metric": safe_metric}
    if safe_metric == "minkowski":
        kwargs["p"] = max(1, int(p))
    return KNeighborsClassifier(**kwargs)

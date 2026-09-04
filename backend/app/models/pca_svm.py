"""Fold-local PCA + linear probabilistic SVM pipeline builder."""

from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def build_pca_svm(n_components: int = 2, c: float = 1.0, class_weight: str | None = None, seed: int = 42) -> Pipeline:
    """Return an unfitted PCA/SVC pipeline.

    Normalisation is deliberately owned by ``training.py`` and fitted inside
    every train fold.  PCA lives inside this pipeline, so its fit can only see
    the corresponding caller-provided fold matrix.
    """
    return Pipeline([
        ("pca", PCA(n_components=int(n_components), random_state=int(seed))),
        ("svm", SVC(kernel="linear", C=float(c), class_weight=class_weight, probability=True, random_state=int(seed))),
    ])

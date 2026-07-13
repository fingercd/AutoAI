from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline


def build_pca_lda(n_components: int = 2) -> Pipeline:
    return Pipeline(
        [
            ("pca", PCA(n_components=max(1, int(n_components)))),
            ("lda", LinearDiscriminantAnalysis()),
        ]
    )

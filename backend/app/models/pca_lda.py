"""PCA 降维后接 LDA 的传统分类 Pipeline 构造器。"""

from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline


def build_pca_lda(n_components: int = 2) -> Pipeline:
    """构造 PCA 与 LDA 串联的 sklearn Pipeline。"""
    return Pipeline(
        [
            ("pca", PCA(n_components=max(1, int(n_components)))),
            ("lda", LinearDiscriminantAnalysis()),
        ]
    )

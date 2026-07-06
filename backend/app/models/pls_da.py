from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression


class PLSDAClassifier:
    def __init__(self, n_components: int = 2) -> None:
        self.n_components = max(1, int(n_components))
        self.model = PLSRegression(n_components=self.n_components, scale=False, max_iter=500, tol=1e-6)
        self.classes_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PLSDAClassifier":
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)
        y_one_hot = np.zeros((len(y), len(self.classes_)), dtype=np.float32)
        for col, class_id in enumerate(self.classes_):
            y_one_hot[y == class_id, col] = 1.0
        self.model.fit(np.asarray(x, dtype=np.float32), y_one_hot)
        return self

    def _responses(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(self.model.predict(np.asarray(x, dtype=np.float32)), dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return values

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("PLS-DA model is not fitted")
        responses = self._responses(x)
        return self.classes_[np.argmax(responses, axis=1)]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        responses = self._responses(x)
        responses = responses - responses.max(axis=1, keepdims=True)
        exp = np.exp(responses)
        return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def build_pls_da(n_components: int = 2) -> PLSDAClassifier:
    return PLSDAClassifier(n_components=n_components)

"""A compact supervised sparse PLS-DA classifier.

The estimator performs sparsification *while each latent component is solved*:
the loading vector is thresholded to its ``keepX`` strongest variables before
the score/deflation step.  It is therefore not a post-hoc top-K selection on a
normal PLS-DA fit.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(np.clip(shifted, -700.0, 700.0))
    return exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), 1e-12)


class SPLSDAClassifier(ClassifierMixin, BaseEstimator):
    """mixOmics-style sparse PLS-DA using one-hot response deflation.

    ``keepX`` is the maximum number of non-zero feature weights for every
    component.  All learned centring/scaling, selected variables and component
    matrices are fitted only on the estimator's ``fit`` input, which lets the
    training service safely instantiate it inside each grouped inner fold.
    """

    def __init__(self, n_components: int = 2, keepX: int = 50, max_iter: int = 200, tol: float = 1e-8):
        self.n_components = n_components
        self.keepX = keepX
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, x: np.ndarray, y: np.ndarray):
        values, labels = check_X_y(x, y, ensure_2d=True, dtype=np.float64)
        if int(self.n_components) < 1:
            raise ValueError("n_components 必须至少为 1")
        if int(self.keepX) < 1:
            raise ValueError("keepX 必须至少为 1")
        self.classes_, encoded = np.unique(labels, return_inverse=True)
        if self.classes_.size < 2:
            raise ValueError("sPLS-DA 至少需要两个类别")
        self.x_mean_ = values.mean(axis=0)
        self.x_scale_ = np.maximum(values.std(axis=0), 1e-12)
        x_residual = (values - self.x_mean_) / self.x_scale_
        y_one_hot = np.eye(self.classes_.size, dtype=np.float64)[encoded]
        self.y_mean_ = y_one_hot.mean(axis=0)
        y_residual = y_one_hot - self.y_mean_

        max_components = min(int(self.n_components), values.shape[0] - 1, values.shape[1])
        if max_components < 1:
            raise ValueError("样本或特征数不足，无法拟合 sPLS-DA")
        keep = min(int(self.keepX), values.shape[1])
        weights: list[np.ndarray] = []
        loadings: list[np.ndarray] = []
        responses: list[np.ndarray] = []
        selected: list[list[int]] = []
        for _component in range(max_components):
            covariance = x_residual.T @ y_residual
            if not np.any(np.isfinite(covariance)) or np.linalg.norm(covariance) <= 1e-12:
                break
            # Initialise the response direction from the dominant cross-block
            # singular vector, then alternate X/Y supervised PLS directions.
            _, _, vh = np.linalg.svd(covariance, full_matrices=False)
            u = y_residual @ vh[0]
            previous = None
            for _ in range(int(self.max_iter)):
                raw_w = x_residual.T @ u
                if np.linalg.norm(raw_w) <= 1e-12:
                    break
                top = np.argpartition(np.abs(raw_w), -keep)[-keep:]
                sparse_w = np.zeros_like(raw_w)
                sparse_w[top] = raw_w[top]
                norm = np.linalg.norm(sparse_w)
                if norm <= 1e-12:
                    break
                sparse_w /= norm
                score = x_residual @ sparse_w
                response_weight = y_residual.T @ score
                response_norm = np.linalg.norm(response_weight)
                if response_norm <= 1e-12:
                    break
                response_weight /= response_norm
                next_u = y_residual @ response_weight
                if previous is not None and np.linalg.norm(sparse_w - previous) <= float(self.tol):
                    u = next_u
                    break
                previous = sparse_w
                u = next_u
            if previous is None:
                break
            w = previous
            score = x_residual @ w
            score_norm_sq = float(score @ score)
            if score_norm_sq <= 1e-12:
                break
            p = (x_residual.T @ score) / score_norm_sq
            q = (y_residual.T @ score) / score_norm_sq
            x_residual = x_residual - np.outer(score, p)
            y_residual = y_residual - np.outer(score, q)
            weights.append(w)
            loadings.append(p)
            responses.append(q)
            selected.append(sorted(np.flatnonzero(np.abs(w) > 0).astype(int).tolist()))
        if not weights:
            raise ValueError("sPLS-DA 未能提取有效监督稀疏成分")
        self.x_weights_ = np.column_stack(weights)
        self.x_loadings_ = np.column_stack(loadings)
        self.y_loadings_ = np.column_stack(responses)
        # W (P'W)^-1 Q' maps scaled X to one-hot response space.
        self.coef_ = self.x_weights_ @ np.linalg.pinv(self.x_loadings_.T @ self.x_weights_) @ self.y_loadings_.T
        self.selected_feature_indices_by_component = selected
        self.actual_n_components_ = len(weights)
        self.actual_keepX_ = keep
        self.n_features_in_ = values.shape[1]
        return self

    def _response_scores(self, x: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "coef_")
        values = check_array(x, ensure_2d=True, dtype=np.float64)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("sPLS-DA 特征数与拟合时不一致")
        return ((values - self.x_mean_) / self.x_scale_) @ self.coef_ + self.y_mean_

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return _softmax(self._response_scores(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]


def build_spls_da(n_components: int = 2, keepX: int = 50) -> SPLSDAClassifier:
    return SPLSDAClassifier(n_components=n_components, keepX=keepX)

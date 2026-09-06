"""Fold-local feature transformations for the explicitly versioned 0904 plan."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

EXPERIMENT_VERSION = 'word-0904'
SCHEMES = [('full', '全特征'), ('bin_5', 'Binning 5'), ('bin_10', 'Binning 10'), ('bin_20', 'Binning 20'), ('pca_90', 'PCA 90%'), ('pca_95', 'PCA 95%'), ('pca_99', 'PCA 99%')]
MODELS = {'pls_da', 'logistic_regression', 'svm', 'random_forest', 'xgboost', 'cnn1d'}


def bin_mean(x: np.ndarray, width: int) -> np.ndarray:
    """Preserve the partial last bin, using its actual number of points."""
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2 or width < 1:
        raise ValueError('Binning 需要二维特征和正整数箱宽')
    return np.column_stack([values[:, start:start + width].mean(axis=1) for start in range(0, values.shape[1], width)])


@dataclass
class FeatureTransform:
    scheme_id: str = 'full'
    normalization: str = 'zscore'

    def _base(self, x: np.ndarray) -> np.ndarray:
        return bin_mean(x, int(self.scheme_id.split('_')[1])) if self.scheme_id.startswith('bin_') else np.asarray(x, dtype=np.float64)

    def fit(self, x: np.ndarray) -> 'FeatureTransform':
        if self.scheme_id not in dict(SCHEMES):
            raise ValueError('未知特征工程方案')
        if self.normalization not in {'zscore', 'minmax'}:
            raise ValueError('0904 方案要求 zscore 或 minmax 特征标准化')
        values = self._base(x)
        if not np.isfinite(values).all():
            raise ValueError('特征包含非有限数值')
        self.input_features_ = np.asarray(x).shape[1]
        self.scaler_ = MinMaxScaler() if self.normalization == 'minmax' else StandardScaler()
        scaled = self.scaler_.fit_transform(values)
        self.pca_ = None
        if self.scheme_id.startswith('pca_'):
            if len(values) < 2 or not np.any(np.var(scaled, axis=0) > 0):
                raise ValueError('PCA 需要至少两条记录和非零方差')
            self.pca_ = PCA(n_components=int(self.scheme_id.split('_')[1]) / 100, svd_solver='full')
            scaled = self.pca_.fit_transform(scaled)
        self.output_features_ = scaled.shape[1]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        values = self.scaler_.transform(self._base(x))
        if self.pca_ is not None:
            values = self.pca_.transform(values)
        return np.asarray(values, dtype=np.float32)

    def metadata(self) -> dict:
        return {'scheme_id': self.scheme_id, 'scheme_name': dict(SCHEMES)[self.scheme_id], 'normalization': self.normalization, 'input_features': self.input_features_, 'output_features': self.output_features_, 'pca_components': int(self.pca_.n_components_) if self.pca_ is not None else None, 'explained_variance_ratio': self.pca_.explained_variance_ratio_.tolist() if self.pca_ is not None else None}


class FeatureClassifier:
    """Private persisted traditional model includes its fitted transformation."""
    def __init__(self, transform: FeatureTransform, estimator):
        self.transformer = transform
        self.estimator = estimator
        self.classes_ = estimator.classes_

    def predict(self, x):
        return self.estimator.predict(self.transformer.transform(x))

    def predict_proba(self, x):
        return self.estimator.predict_proba(self.transformer.transform(x))

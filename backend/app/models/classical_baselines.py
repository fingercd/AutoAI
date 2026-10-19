"""Additional genuine classifiers; Lasso/Ridge mean regularized logistic classification."""
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def build_shrinkage_lda():
    return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')


def build_regularized_logistic(config, class_weight):
    estimator = LogisticRegression(C=config.logistic_c, penalty='l1' if config.model_type == 'lasso' else 'l2',
                                   solver='liblinear', max_iter=5000, random_state=config.seed,
                                   class_weight=class_weight)
    # Explicit OvR supports multiclass without relying on deprecated implicit
    # liblinear multiclass behavior. Avoid nested process pools inside a Worker.
    return OneVsRestClassifier(estimator, n_jobs=1)


def build_knn(config):
    return KNeighborsClassifier(n_neighbors=config.knn_n_neighbors, weights=config.knn_weights,
                                metric=config.knn_metric, p=config.knn_p,
                                n_jobs=int(os.environ.get('AUTOAI_WORKER_THREADS', '1')))


def build_extra_trees(config, class_weight):
    return ExtraTreesClassifier(n_estimators=config.extra_trees_n_estimators,
                                max_depth=config.extra_trees_max_depth,
                                min_samples_leaf=config.extra_trees_min_samples_leaf,
                                max_features=config.extra_trees_max_features,
                                bootstrap=False, random_state=config.seed, class_weight=class_weight,
                                n_jobs=int(os.environ.get('AUTOAI_WORKER_THREADS', '1')))

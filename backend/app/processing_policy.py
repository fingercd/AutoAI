"""Finite, executable preprocessing choices for classification models.

This module owns capability checks. Model hyperparameters and the numerical
implementation remain in their existing authoritative modules.
"""
from __future__ import annotations

from itertools import product
import hashlib
import json
from typing import Any, Iterable

PROCESSING_POLICY_VERSION = "finite-processing-v1"
NORMALIZATIONS = ("zscore", "minmax", "area", "none")
BALANCES = ("none", "class_weight")
_WEIGHTED_TRADITIONAL = frozenset({"logistic_regression", "svm", "random_forest"})
_UNWEIGHTED_TRADITIONAL = frozenset({"pls_da", "pca_lda"})
_DEEP = frozenset({
    "pca_mlp", "cnn1d", "cnn1d_se", "resnet1d", "inception1d",
    "tcn1d", "cnn_transformer1d",
})
EXECUTABLE_MODELS = _WEIGHTED_TRADITIONAL | _UNWEIGHTED_TRADITIONAL | {"xgboost"} | _DEEP


def validate_processing(
    model_id: str,
    normalization: str,
    class_balance: str,
    *,
    class_count: int | None = None,
) -> tuple[str, str]:
    """Reject unsupported combinations before fit, without silent fallbacks."""
    if type(model_id) is not str or model_id not in EXECUTABLE_MODELS:
        raise ValueError("processing_model_unavailable")
    if type(normalization) is not str or normalization not in NORMALIZATIONS:
        raise ValueError("invalid_normalization")
    if type(class_balance) is not str or class_balance not in BALANCES:
        raise ValueError("invalid_class_balance")
    if class_count is not None and (type(class_count) is not int or class_count < 2):
        raise ValueError("invalid_class_count")
    if class_balance == "class_weight":
        if model_id in _UNWEIGHTED_TRADITIONAL:
            raise ValueError("class_weight_unsupported_for_model")
        if model_id == "xgboost" and class_count is not None and class_count != 2:
            raise ValueError("class_weight_unsupported_for_multiclass_xgboost")
    return normalization, class_balance


def legal_processing(model_id: str, *, class_count: int) -> tuple[tuple[str, str], ...]:
    """Return the complete finite domain in stable order."""
    return tuple(
        (normalization, balance)
        for normalization, balance in product(NORMALIZATIONS, BALANCES)
        if _is_legal(model_id, normalization, balance, class_count)
    )


def processing_execution_digest(model_id: str, normalization: str, class_balance: str) -> str:
    validate_processing(model_id, normalization, class_balance)
    body = dict(policy_version=PROCESSING_POLICY_VERSION, model_id=model_id,
                normalization=normalization, class_balance=class_balance)
    return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',', ':'),
        ensure_ascii=False,allow_nan=False).encode('utf-8')).hexdigest()


def _is_legal(model_id: str, normalization: str, balance: str, class_count: int) -> bool:
    try:
        validate_processing(model_id, normalization, balance, class_count=class_count)
    except ValueError:
        return False
    return True


def freeze_fixed_processing(
    allowed_models: Iterable[str], supplied: dict[str, Any] | None = None,
) -> dict[str, dict[str, str]]:
    """Freeze exactly one explicit or default processing choice per model."""
    models = tuple(allowed_models)
    if len(models) != len(set(models)):
        raise ValueError("duplicate_allowed_model")
    if supplied is not None and (type(supplied) is not dict or set(supplied) != set(models)):
        raise ValueError("fixed_processing_model_set_mismatch")
    result: dict[str, dict[str, str]] = {}
    for model_id in sorted(models):
        item = {"normalization": "zscore", "class_balance": "none"} if supplied is None else supplied[model_id]
        if type(item) is not dict or set(item) != {"normalization", "class_balance"}:
            raise ValueError("invalid_fixed_processing_member")
        normalization, balance = validate_processing(
            model_id, item["normalization"], item["class_balance"]
        )
        result[model_id] = {"normalization": normalization, "class_balance": balance}
    return result

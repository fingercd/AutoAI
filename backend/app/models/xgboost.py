from __future__ import annotations

from typing import Any

import numpy as np


def build_xgboost(
    y: np.ndarray,
    class_count: int,
    class_balance: str,
    seed: int,
    n_estimators: int = 50,
    max_depth: int = 2,
    learning_rate: float = 0.1,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    reg_lambda: float = 2.0,
) -> Any:
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise ValueError("The current environment does not have xgboost installed, so XGBoost cannot be trained.") from exc

    params: dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_lambda": reg_lambda,
        "eval_metric": "logloss" if class_count == 2 else "mlogloss",
        "random_state": seed,
        "n_jobs": 1,
    }
    if class_count == 2:
        params["objective"] = "binary:logistic"
        if class_balance == "class_weight":
            counts = np.bincount(y, minlength=class_count).astype(np.float32)
            params["scale_pos_weight"] = float(counts[0] / max(counts[1], 1.0))
    else:
        params["objective"] = "multi:softprob"
        params["num_class"] = class_count
    return XGBClassifier(**params)

"""延迟导入 XGBoost 并按二分类/多分类配置目标函数。

模块职责
--------
本模块提供 ``build_xgboost``，按类别数（二分类/多分类）与类别均衡策略构造
一个 ``xgboost.XGBClassifier`` 实例。它对应能力目录中的 ``xgboost`` 目标
模型，属于传统机器学习模型。

在系统中的位置
--------------
- 上游：``backend/app/services/training.py`` 在训练/搜索流程中调用本函数，
  传入已编码的训练标签 ``y``、类别数与超参数；搜索编排、数据划分、评估口径
  （stratified_holdout / leave_one_sample_id_cv / external_test_holdout）
  全部由 training.py 负责，本模块不感知。
- 依赖：``xgboost`` 是可选重依赖。项目要求在 import 失败时给出可读的拒绝
  信息而不是让 ImportError 裸奔（类似 ``cnn_mamba1d`` 因依赖缺失标记
  ``available=false`` 的思路），因此导入被延迟到函数内部。

关键设计约束
------------
- 二分类与多分类使用不同的 objective 与 eval_metric，必须在构造期一次性
  配好，避免 sklearn 接口包装下的目标函数错配。
- 类别均衡通过 ``scale_pos_weight``（仅二分类有效）实现，与 ``class_balance``
  请求参数联动；多分类不设该参数（XGBoost 无直接等价物）。
- 默认超参数偏保守（树浅、学习率小、带 subsample/colsample 与 L2 正则），
  适配光谱/色谱小样本高维场景。
"""

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
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
) -> Any:
    """按类别数量与 class_balance 构造 XGBClassifier。

    参数
    ----
    y:
        已做类别编码（整数 0..class_count-1）的训练标签向量。仅在
        二分类且启用类别权重时用于统计正负类样本数，计算
        ``scale_pos_weight``。
    class_count:
        类别总数。等于 2 走二分类分支，否则走多分类分支。
    class_balance:
        类别均衡策略；取值 ``"class_weight"`` 时在二分类下启用
        ``scale_pos_weight``，其他取值视为等权。
    seed:
        随机种子，保证 boosting 过程可复现。
    n_estimators / max_depth / learning_rate / subsample /
    colsample_bytree / reg_lambda / min_child_weight / gamma:
        XGBoost 常规超参数，默认值为训练服务的小样本保守基线
        （浅树 + 行/列子采样 + L2 正则，抑制过拟合）。

    返回
    ----
    Any
        未拟合的 ``XGBClassifier`` 实例。返回类型标注为 ``Any`` 是因为
        xgboost 为延迟导入，模块级无法引用其类型。

    异常
    ----
    ValueError
        当前环境未安装 xgboost（或导入失败）时抛出，消息面向用户说明
        无法训练的原因；由 training.py 捕获后转为 Run 失败状态。
    """
    try:
        # 延迟导入：xgboost 是可选重依赖，模块级导入会让整个 models 包在
        # 缺失环境下无法加载；放在函数内才能把"依赖缺失"隔离为本模型的失败。
        from xgboost import XGBClassifier
    except Exception as exc:
        # 捕获宽泛 Exception 而非仅 ImportError：xgboost 导入还可能因原生
        # 库（libxgboost）加载失败抛出 OSError 等，业务上这些都算"不可用"。
        raise ValueError("The current environment does not have xgboost installed, so XGBoost cannot be trained.") from exc

    # 通用超参数：eval_metric 按任务类型选择 logloss/mlogloss，与 objective 配套；
    # n_jobs 固定为 1，避免与外层训练编排的并行策略叠加造成线程超额订阅。
    params: dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_lambda": reg_lambda,
        "min_child_weight": min_child_weight,
        "gamma": gamma,
        "eval_metric": "logloss" if class_count == 2 else "mlogloss",
        "random_state": seed,
        "n_jobs": 1,
    }
    if class_count == 2:
        # 二分类：logistic 目标直接输出正类概率。
        params["objective"] = "binary:logistic"
        if class_balance == "class_weight":
            # 少数类（正类，编码 1）权重 = 负类数 / 正类数，是 XGBoost 处理
            # 类别不均衡的标准做法；max(..., 1.0) 防止训练折中某类缺失导致除零。
            counts = np.bincount(y, minlength=class_count).astype(np.float32)
            params["scale_pos_weight"] = float(counts[0] / max(counts[1], 1.0))
    else:
        # 多分类：softprob 输出各类别概率分布，num_class 必须显式给出，
        # 否则 XGBoost 无法确定 softmax 维度。
        params["objective"] = "multi:softprob"
        params["num_class"] = class_count
    return XGBClassifier(**params)

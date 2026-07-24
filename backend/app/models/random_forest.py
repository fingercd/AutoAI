"""Random Forest 分类器构造器；搜索与 OOB 选优由 training.py 编排。

模块职责
--------
本模块只负责一件事：按给定超参数构造一个 sklearn 的 ``RandomForestClassifier``
实例。它是 15 个分类目标模型之一 ``random_forest`` 的构造入口，属于传统机器
学习模型（与 ``pls_da``、``svm``、``xgboost`` 等同级）。

在系统中的位置
--------------
- 上游：``backend/app/services/training.py``（训练编排器）在超参数搜索循环中
  反复调用 ``build_random_forest``，每组候选超参数构造一个新实例进行拟合与
  评估；本模块自身不做数据读取、不做搜索、不做评估。
- 下游：返回的模型实例由 training.py 负责 ``fit``、交叉验证/holdout 评估，
  并最终序列化为 ``model.pkl``。
- 预处理输入为 ``wide-feature-v2`` 宽表转换出的特征矩阵，但本模块不感知该
  契约，只接收已经定好的超参数。

关键设计约束
------------
- 所有超参数都有默认值，默认值即训练服务的"安全基线"（浅树、正则化偏强），
  以适应光谱/色谱小样本、高维特征、易过拟合的典型场景。
- 固定 ``bootstrap=True``，这是启用 OOB（out-of-bag）评分的前提；OOB 是否
  计算由 ``oob_score`` 开关控制，便于 training.py 做不占用验证折的选优审计。
- 固定 ``n_jobs=-1`` 使用全部 CPU 核心并行建树，单棵树之间天然独立。
- 随机性完全由传入的 ``seed`` 控制，保证同一评估口径下结果可复现。
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def build_random_forest(
    n_estimators: int = 200,
    max_depth: int | None = 3,
    min_samples_leaf: int = 2,
    max_features: str | float = "sqrt",
    class_weight: str | None = None,
    seed: int = 42,
    oob_score: bool = False,
) -> RandomForestClassifier:
    """构造支持 OOB 审计和类别权重的随机森林。

    参数
    ----
    n_estimators:
        森林中决策树的数量。树越多集成方差越低但训练越慢；默认 200 是
        小样本光谱数据上精度与耗时的折中。
    max_depth:
        单棵树的最大深度，``None`` 表示不限制。默认 3（浅树）是有意的
        强正则化：高维小样本下深树极易过拟合。
    min_samples_leaf:
        叶节点最小样本数。默认 2 进一步平滑叶节点预测，抑制噪声样本
        被单独分裂出来。
    max_features:
        每次分裂候选特征数，``"sqrt"``（特征总数开平方）是分类任务的
        经典默认，兼顾分裂质量与树间多样性。
    class_weight:
        类别权重策略（如 ``"balanced"``），用于类别不均衡时放大少数类
        样本的损失权重；``None`` 表示等权。
    seed:
        随机种子，同时控制 bootstrap 抽样与特征子空间抽样，保证可复现。
    oob_score:
        是否计算 OOB 评分。开启后 sklearn 用每棵树未参与训练的袋外样本
        估计泛化精度，供 training.py 做不额外消耗验证折的选优参考。

    返回
    ----
    RandomForestClassifier
        未拟合的 sklearn 分类器实例，由调用方（training.py）负责训练。

    说明
    ----
    本函数不校验超参数合理性（非法组合由 sklearn 在 fit 时报错），也不
    触碰数据；它只是把"超参数组合"翻译成"可训练的模型对象"。
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=seed,
        # 使用全部 CPU 核心并行建树；树之间相互独立，并行无正确性风险。
        n_jobs=-1,
        # 强制有放回抽样：这是 OOB 评分存在的前提（每棵树约 1/3 样本留作袋外），
        # 也是随机森林降低方差的核心机制，因此写死而非暴露为参数。
        bootstrap=True,
        oob_score=oob_score,
    )

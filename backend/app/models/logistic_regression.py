"""Logistic Regression 分类基线构造器。

模块定位：
    本文件实现传统机器学习模型 ``logistic_regression``（逻辑回归）
    的工厂函数，是当前前端 10 项目录与后端能力目录共用的传统模型之一，
    常作为线性分类基线与其他模型对比。

系统协作：
    - 由 ``backend/app/models/registry.py`` 按训练配置调用
      ``build_logistic_regression`` 构造，正则强度来自配置
      ``logistic_c``，随机种子来自配置 ``seed``，类别权重由
      registry 按配置解析后传入；
    - 输入特征来自预处理输出的 ``wide-feature-v2`` 宽表，外层
      训练流程先用仅由训练集拟合的标准化器处理特征——逻辑回归
      的 saga 求解对特征量纲敏感，标准化是该模型正常收敛的
      前提，但不在本文件职责内。

关键设计约束：
    - sklearn 逻辑回归原生提供 ``predict_proba``（softmax /
      sigmoid 后验），可直接满足下游 Log-loss 评估与可解释性
      分析对概率输出的统一要求；
    - 求解器固定为 ``saga`` + elastic-net；`C` 与 `l1_ratio` 都在
      分组内层 5 折中选择，避免把验证或测试数据泄漏到正则选择。
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def build_logistic_regression(
    c: float = 1.0,
    seed: int = 42,
    class_weight: str | None = None,
    l1_ratio: float = 0.5,
) -> LogisticRegression:
    """构造固定 saga/elastic-net 求解器和可选类别权重的分类基线。

    参数：
        c: 正则化强度的倒数 C（sklearn 惯例）：C 越大正则越弱、
           模型越贴近训练数据；越小则系数收缩越强、模型越平滑。
           ``float(c)`` 兼容配置层传入字符串数字的情况。
        seed: 随机种子，固定 ``random_state`` 控制 SAGA 的随机更新顺序。
        class_weight: 类别权重，通常为 None 或 "balanced"；后者按
           类别频率反比加权，缓解类别不平衡。

    返回：
        配置完成的 sklearn ``LogisticRegression`` 实例（未拟合）。
        ``max_iter=10000``、``tol=1e-3`` 为高维相关特征提供足够迭代预算，
        仍由 SAGA 的相对系数更新停止条件判断收敛，不隐藏警告。
    """
    return LogisticRegression(
        C=float(c),
        class_weight=class_weight,
        max_iter=10000,
        tol=1e-3,
        random_state=int(seed),
        solver="saga",
        penalty="elasticnet",
        l1_ratio=float(l1_ratio),
        multi_class="auto",
    )

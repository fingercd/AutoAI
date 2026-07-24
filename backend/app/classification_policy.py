"""集中定义分类训练默认值与评估口径。

【模块职责与系统位置】
本模块是训练流水线中"评估口径（evaluation policy）"的唯一权威定义点，
位于训练配置校验（contracts.py 的 TrainingSpec.validated）与训练执行器之间。
训练请求进入后端后，先由 TrainingSpec 调用本模块的 resolve_evaluation_policy，
把用户在前端选择的 split_mode 和 train/valid/test 比例归一化为一个
不可歧义的 EvaluationPolicy；之后的训练器必须严格按该策略执行数据划分。

【协作模块】
- contracts.py：TrainingSpec.validated 调用 resolve_evaluation_policy，
  并把解析结果回写进 values（split_mode/split_train/split_valid/split_test）。
- 训练执行器（training 相关模块）：消费 EvaluationPolicy 决定如何划分
  train/valid/test，以及是否允许交叉验证。

【关键设计约束】
本模块只负责把请求中的划分选项归一化成不可歧义的策略；它不读取数据，
也不执行划分。训练代码必须以这里返回的策略为准，避免前端、API 和训练器
分别解释比例，尤其要保证 external test 与留一交叉验证不会同时启用。
三种合法评估口径：
- stratified_holdout：无独立测试集时的分层留出（train/valid/test 均 > 0 且和为 10）。
- leave_one_sample_id_cv：留一样品交叉验证（train/valid 和为 10，test=0）。
- external_test_holdout：有独立测试集时，主数据只分 train/valid（和为 10），
  独立测试集充当最终 test，此时禁止开启交叉验证。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class DeepTrainingDefaults:
    """文档版深度模型共享的、不可变训练默认值。

    frozen=True 表示实例创建后字段不可修改，保证所有深度模型
    （cnn1d/resnet1d/dscarnet 等）在未被用户显式覆盖时使用完全一致的
    超参数基线，避免不同代码路径各自散落一份默认值。
    """
    # 训练轮数上限；配合 early_stopping_patience 提前终止，通常不会跑满。
    epochs: int = 200
    # 批大小 8：光谱样品数量有限，小批量有助于稳定收敛。
    batch_size: int = 8
    # Adam 初始学习率。
    learning_rate: float = 1e-3
    # L2 权重衰减系数，抑制过拟合。
    weight_decay: float = 1e-4
    # ReduceLROnPlateau 的衰减因子与容忍轮数：验证指标停滞 10 轮后 lr *= 0.5。
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    # 学习率下限，防止调度器把 lr 衰减到 0 导致训练停摆。
    min_learning_rate: float = 1e-6
    # 早停容忍轮数：验证集指标连续 20 轮无改善即停止训练。
    early_stopping_patience: int = 20
    # 全局随机种子，保证划分/初始化可复现。
    seed: int = 42


# 模块级单例：所有训练入口共享同一份默认值。
DEEP_TRAINING_DEFAULTS = DeepTrainingDefaults()


@dataclass(frozen=True)
class EvaluationPolicy:
    """训练器实际执行的评估策略及 train/valid/test 权重。

    该对象由 resolve_evaluation_policy 生成，是训练器划分数据的唯一依据：
    strategy 为三种评估口径之一（stratified_holdout / leave_one_sample_id_cv /
    external_test_holdout）；split_train/split_valid/split_test 为归一化后的
    比例权重（总和恒为 10）；cv_allowed 标记该口径是否允许交叉验证。
    """
    # 评估口径名称，训练器据此选择划分逻辑。
    strategy: str
    # train/valid/test 三个比例权重，总和恒等于 10。
    split_train: int
    split_valid: int
    split_test: int
    # 是否允许在该口径下启用交叉验证；external_test_holdout 恒为 False。
    cv_allowed: bool


# 所有被识别为“留一交叉验证”的 split_mode 写法集合。
# 除当前正式口径 leave_one_sample_id_cv 外，还保留历史别名
# （leave_one_repeat_index_cv / loocv / loo 等），保证旧配置文件仍被正确
# 解释为 CV 而不是静默退回 stratified_holdout。
_CV_MODES = {
    "leave_one_sample_id_cv",
    "leave_one_repeat_index_cv",
    "outer_leave_one_repeat_index_cv",
    "loocv",
    "loo",
}


def resolve_evaluation_policy(
    config_data: Mapping[str, object] | None,
    *,
    has_external_test: bool,
) -> EvaluationPolicy:
    """根据外部测试集和 split_mode 解析唯一合法的评估策略。

    参数：
        config_data: 训练请求中的 config 字典（可为 None）。只读取其中的
            split_mode 与 split_train/split_valid/split_test，其余键忽略。
        has_external_test: 请求是否携带了独立测试集（test_dataset_id 等）。
            该标志决定第一优先级分支：有独立测试集时强制走 external_test_holdout。

    返回：
        EvaluationPolicy：训练器据此划分数据；比例权重总和恒为 10。

    异常：
        ValueError: 比例不合法（不为正、总和不为 10）或口径冲突
            （独立测试集 + 交叉验证）时抛出；调用方 contracts.py 会将其
            包装为 TrainingConfigValidationError 返回 4xx。

    设计意图：三种口径互斥且优先级固定——external test > CV > stratified
    holdout，避免前端、API 和训练器各自解释比例产生分歧。
    """
    # 拷贝为普通 dict，避免改动调用方传入的 Mapping。
    config = dict(config_data or {})
    requested = str(config.get("split_mode") or "stratified_holdout").strip().lower()

    # 分支一：有独立测试集。此时 CV 与外部 test 语义冲突，直接拒绝；
    # 主数据只划分 train/valid（默认 8:2），内部 test 必须为 0，
    # 因为最终测试由独立测试集承担，不允许再从主数据里切一份 test。
    if has_external_test:
        if requested in _CV_MODES:
            raise ValueError("已提供独立测试集时不允许开启交叉验证")
        train = int(config.get("split_train", 8))
        valid = int(config.get("split_valid", 2))
        test = int(config.get("split_test", 0))
        if test != 0 or train <= 0 or valid <= 0 or train + valid != 10:
            raise ValueError("独立测试集模式要求主数据训练/验证比例相加必须等于 10，且内部测试比例为 0")
        # cv_allowed=False：已用独立测试集，禁止再开交叉验证。
        return EvaluationPolicy("external_test_holdout", train, valid, 0, False)

    # 分支二：无独立测试集且用户请求了留一交叉验证。
    # 每折留出 1 个 Sample_ID 作 test，其余按 train:valid（默认 8:2）划分，
    # 因此内部 test 权重恒为 0，train+valid 必须等于 10。
    if requested in _CV_MODES:
        train = int(config.get("split_train", 8))
        valid = int(config.get("split_valid", 2))
        if train <= 0 or valid <= 0 or train + valid != 10:
            raise ValueError("留一交叉验证要求训练/验证比例相加必须等于 10")
        return EvaluationPolicy("leave_one_sample_id_cv", train, valid, 0, True)

    # 分支三（默认）：分层 holdout。train/valid/test 默认 8:1:1，
    # 三者都必须为正且和为 10；训练器按 Sample_ID 整组分层划分。
    train = int(config.get("split_train", 8))
    valid = int(config.get("split_valid", 1))
    test = int(config.get("split_test", 1))
    if min(train, valid, test) <= 0 or train + valid + test != 10:
        raise ValueError("分层 holdout 要求训练、验证、测试比例均大于 0 且相加为 10")
    return EvaluationPolicy("stratified_holdout", train, valid, test, True)

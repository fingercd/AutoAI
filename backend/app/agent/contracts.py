"""Agent Session HTTP 契约（Pydantic）。

设计要点：
- 所有 Agent 请求体 ``extra='forbid'``：Agent 显式不允许扩展字段，绝不静默丢弃
  未知键（与现有 ``TrainingRunRequest`` 保持一致风格）。
- Session 创建后所有字段都被锁定；Experiment 只接收 ``model_type``、
  ``normalization``、``class_balance`` 与 ``parent_run_id``、``rationale``。
- 响应模型不在此定义——服务层直接构造 dict，避免 Pydantic 模型把
  ``arbitrary_types_allowed`` 关掉之后还把额外字段塞进序列化结果。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


AGENT_API_CONTRACT_VERSION = 'agent-session-v1'
AGENT_API_CAPABILITIES = (
    'create_session',
    'create_experiment',
    'read_session',
    'read_feedback',
    'finalize_session',
)


class AgentModuleFlags(BaseModel):
    """Session-scoped feature switches used for reproducible A/B runs."""

    model_config = ConfigDict(extra='forbid')

    evidence_card: bool = False
    dynamic_preprocessing: bool = False
    restricted_strategy_pool: bool = False
    bounded_hpo: bool = False
    fail_fast_guard: bool = False
    constrained_code_evolution: bool = False
    feedback_diagnosis: bool = False
    limited_replanning: bool = False
    uncertainty_selection: bool = False
    case_memory: bool = False
    budget_control: bool = False


class AgentContextPolicy(BaseModel):
    """Controls where a Session may read/write learned planning context."""

    model_config = ConfigDict(extra='forbid')

    source_role: Literal['development', 'benchmark', 'domain'] = 'development'
    case_write: bool = False


# 第一版锁定的传统模型白名单；与 ``backend/app/models/registry.py`` 的
# TRADITIONAL_MODEL_TYPES 取交集得到 6 个，这里只开放 3 个简单稳定基线。
_AGENT_ALLOWED_MODELS = frozenset({'logistic_regression', 'svm', 'random_forest'})
# selection_metric 第一版只允许 macro_f1 / balanced_accuracy——这两个值在
# ``training.py:1156-1218`` 的 ``_traditional_candidate_configs`` 选优路径上
# 都是天然支持的。
_AGENT_SELECTION_METRICS = frozenset({'macro_f1', 'balanced_accuracy'})
# 评估口径只允许 stratified_holdout（外部测试集由 Session 锁定后不可改）。
_AGENT_SPLIT_MODES = frozenset({'stratified_holdout'})
# normalization 与 class_balance 的合法值取自现有 ``TrainingSpec.validated``。
_AGENT_NORMALIZATIONS = frozenset({'zscore', 'minmax', 'area', 'none'})
_AGENT_CLASS_BALANCES = frozenset({'none', 'class_weight'})


class AgentEvaluationBlock(BaseModel):
    """Session 内锁定的评估口径（split_mode + 比例）。"""

    model_config = ConfigDict(extra='forbid')

    split_mode: Literal['stratified_holdout'] = 'stratified_holdout'
    split_train: int = Field(8, ge=1, le=9)
    split_valid: int = Field(1, ge=1, le=9)
    split_test: int = Field(1, ge=1, le=9)


class CreateAgentSessionRequest(BaseModel):
    """``POST /api/agent/sessions`` 的请求体。"""

    model_config = ConfigDict(extra='forbid')

    dataset_id: str = Field(..., min_length=1)
    selection_metric: str = Field(...)
    allowed_models: list[str] = Field(..., min_length=1)
    max_runs: int = Field(..., ge=1, le=10)
    seed: int = Field(42, ge=0)
    evaluation: AgentEvaluationBlock = Field(default_factory=AgentEvaluationBlock)
    modules: AgentModuleFlags = Field(default_factory=AgentModuleFlags)
    context_policy: AgentContextPolicy = Field(default_factory=AgentContextPolicy)

    def validate(self) -> None:
        if self.selection_metric not in _AGENT_SELECTION_METRICS:
            raise ValueError(
                f'selection_metric 必须来自 {sorted(_AGENT_SELECTION_METRICS)}'
            )
        unknown_models = [
            model for model in self.allowed_models if model not in _AGENT_ALLOWED_MODELS
        ]
        if unknown_models:
            raise ValueError(
                f'allowed_models 仅支持 {sorted(_AGENT_ALLOWED_MODELS)},'
                f' 当前收到 {unknown_models}'
            )
        ratios = (self.evaluation.split_train, self.evaluation.split_valid,
                  self.evaluation.split_test)
        if sum(ratios) != 10 or min(ratios) <= 0:
            raise ValueError('evaluation 比例三项必须为正且相加等于 10')
        if self.context_policy.source_role == 'benchmark' and self.context_policy.case_write:
            raise ValueError('benchmark session 禁止写入案例库')


class CreateAgentExperimentRequest(BaseModel):
    """``POST /api/agent/sessions/{session_id}/experiments`` 的请求体。"""

    model_config = ConfigDict(extra='forbid')

    model_type: str = Field(...)
    normalization: str = Field('zscore')
    class_balance: str = Field('none')
    parent_run_id: str | None = None
    rationale: str | None = Field(None, max_length=2000)

    def validate(self) -> None:
        if self.model_type not in _AGENT_ALLOWED_MODELS:
            raise ValueError(
                f'model_type 仅支持 {sorted(_AGENT_ALLOWED_MODELS)}'
            )
        if self.normalization not in _AGENT_NORMALIZATIONS:
            raise ValueError(
                f'normalization 仅支持 {sorted(_AGENT_NORMALIZATIONS)}'
            )
        if self.class_balance not in _AGENT_CLASS_BALANCES:
            raise ValueError(
                f'class_balance 仅支持 {sorted(_AGENT_CLASS_BALANCES)}'
            )


class FinalizeAgentSessionRequest(BaseModel):
    """``POST /api/agent/sessions/{session_id}/finalize`` 的请求体。"""

    model_config = ConfigDict(extra='forbid')

    selected_run_id: str = Field(..., min_length=1)

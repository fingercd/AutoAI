"""Run 状态、Principal 与记录契约的公共导出。

本包（backend.app.runs）是训练 Run 的领域层：

- contracts 定义状态机使用的不可变领域对象（RunRecord / Principal）与旧状态映射；
- status_projection 把规范 RunRecord 单向投影成旧前端可读的 status.json；
- result_projection 生成结果页 run-result-v1 契约响应；
- artifacts / store 等模块负责产物清单校验与 SQLite 持久化。

这里只再导出最基础的契约符号，路由层与服务层统一从这里引用，
避免上层模块深入子模块细节、也降低循环依赖风险。
"""

# 只导出状态与记录契约；投影、存储等能力由调用方按需显式导入对应子模块。
from .contracts import LEGACY_STATUS, Principal, RunRecord, RunState

# 显式声明公共 API，防止 `from runs import *` 意外泄漏内部实现。
__all__ = ['LEGACY_STATUS', 'Principal', 'RunRecord', 'RunState']

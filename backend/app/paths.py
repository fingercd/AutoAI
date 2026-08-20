"""项目路径常量与本地存储目录初始化。

所有路径都从当前文件位置推导，避免把开发机盘符写入业务逻辑。这里仅创建目录，
不会创建数据库表；数据库 schema 分别由 DatasetRepository 和 RunRepository 管理。

【模块级说明】
- 职责：定义全项目共享的文件系统路径常量（存储根目录、两个 SQLite 库、
  上传/Run/预处理产物目录、前端静态目录、本地验证数据），并提供幂等的
  目录初始化函数 `ensure_storage()`。
- 系统位置：被 `main.py`（启动时初始化）、数据集与 Run 的 Repository、
  预处理流水线和训练 Worker 共同导入，是所有模块对“文件放在哪”的唯一
  事实来源（single source of truth）。
- 关键设计约束：路径全部相对于本文件位置推导，代码可在任意机器/盘符下
  原样运行；本模块只保证目录存在，绝不触碰数据库 schema 或写入业务数据，
  保证“导入本模块”本身除定义常量外没有其他副作用。
"""

from pathlib import Path


# 项目根目录：本文件位于 backend/app/paths.py，向上三级（parents[2]）即仓库根。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 本地存储根目录：所有运行时数据（数据库、上传文件、训练产物）都收敛在
# storage/ 下，便于统一备份、清理，并被 .gitignore 排除在版本库之外。
STORAGE_DIR = PROJECT_ROOT / "storage"
# 训练 Run 元数据库：Run 状态机（queued/running/succeeded 等）由 RunRepository 维护。
RUNS_DATABASE = STORAGE_DIR / "runs.sqlite3"
# 数据集元数据库：上传/预处理产出的数据集登记信息由 DatasetRepository 维护。
DATASETS_DATABASE = STORAGE_DIR / "datasets.sqlite3"
# Agent Session/Experiment 元数据库：与 runs.sqlite3 平级，agent 包独立维护。
AGENT_DATABASE = STORAGE_DIR / "agent.sqlite3"
# 用户上传的原始光谱/色谱文件存放目录。
UPLOADS_DIR = STORAGE_DIR / "uploads"
# 训练 Run 的工作目录根：每个 Run 的产物（模型、指标、可解释性文件、Manifest）
# 按 run_id 分子目录存放在这里。
RUNS_DIR = STORAGE_DIR / "runs"
# 预处理输出目录：wide-feature-v2 宽表 CSV 等预处理结果写入这里。
PREPROCESSED_DIR = STORAGE_DIR / "preprocessed"
# 前端静态资源目录：由 FastAPI 以 /static 挂载，与 API 同源托管。
STATIC_DIR = PROJECT_ROOT / "static"
# 本机验证数据：仅用于本地真实数据流程回归，不提交进 Git；server 模式训练
# 只接受 dataset_id，禁止回退到这个根目录文件。
DEFAULT_DATA = PROJECT_ROOT / "data.csv"


def ensure_storage() -> None:
    """幂等创建 Web、预处理和训练流程需要的本地目录。

    在应用启动（`main.py` 导入时）调用一次：`parents=True` 允许一次性补全
    多级缺失目录，`exist_ok=True` 保证重复调用、多进程并发启动都不会报错。
    注意这里只建目录，不创建数据库文件或表——schema 由各 Repository 在首次
    使用时自行负责，从而避免启动顺序耦合。
    """
    for path in (STORAGE_DIR, UPLOADS_DIR, RUNS_DIR, PREPROCESSED_DIR):
        path.mkdir(parents=True, exist_ok=True)

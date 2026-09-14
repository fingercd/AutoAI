"""路由共享依赖与数据引用解析。

新客户端优先传 dataset_id；data_path 只保留给受控本地兼容。两种引用最终都在
允许目录内解析，并由 Principal 限定作用域，不能借训练接口读取任意路径。

【在系统中的位置】
本模块是路由层（backend/app/routers/ 下的 training、runs 等路由）的共享依赖
集合，介于 FastAPI 请求对象与仓储层（datasets.repository / runs.repository）
之间。路由函数通过它把"客户端传进来的数据引用"（dataset_id 或旧式
data_path）统一解析成受控的本地文件路径，避免每个路由重复实现同一套校验。

【协作模块】
- ..contracts.TrainingRunRequest：训练请求入参模型，携带 dataset_id /
  data_path / test_dataset_id / test_data_path 四种互斥的数据引用。
- ..datasets.repository.DatasetRepository：按 dataset_id + Principal 作用域
  解析已登记的上传数据集，并给出上传时的原始文件名。
- ..runs.repository.RunRepository / ..runs.contracts.Principal：Run 记录仓储与
  调用者身份值对象。
- ..paths：集中定义 STORAGE / UPLOADS / PREPROCESSED / RUNS 等受控目录常量。

【关键设计约束】
- server 模式（Principal 带 owner_id/tenant_id）只接受 dataset_id，拒绝浏览器
  直传 data_path，也不回退根目录 data.csv；local 模式才保留 data_path 兼容。
- 所有旧式路径必须落在受控根目录之内，防止借训练接口读取服务器任意文件。
- run_id 必须是安全的单段名称，防止借 run 下载/删除接口逃出产物根目录。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException

from ..contracts import TrainingRunRequest
from ..datasets.repository import DatasetRepository
from ..paths import DATASETS_DATABASE, DEFAULT_DATA, PREPROCESSED_DIR, RUNS_DATABASE, RUNS_DIR, STORAGE_DIR, UPLOADS_DIR
from ..runs.contracts import Principal
from ..runs.repository import RunRepository


def _is_within(target: Path, root: Path) -> bool:
    """判断 target 是否位于 root 目录之内。

    原理：Path.relative_to 在 target 不是 root 的后代路径时抛 ValueError，
    借此把"路径包含关系"转化为异常捕获，避免手写字符串前缀比较的边界错误
    （例如 '/data2' 也会被 '/data' 的字符串前缀误匹配）。
    """
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_legacy_dataset_path(path: str | Path, *, allowed_roots: list[Path]) -> str:
    """验证兼容路径位于允许根目录并返回规范绝对路径。

    这是 local 模式旧客户端直传 data_path 的兼容入口，server 模式不会走到这里。
    校验分两步：先确认目标是真实存在的文件（否则 404），再确认它落在任一
    受控根目录之内（否则 403），把"任意路径读文件"的口子收窄到 uploads /
    preprocessed / 默认数据文件这几处。

    参数：
        path: 客户端传入的文件路径（相对/绝对均可，内部统一 resolve 成绝对路径）。
        allowed_roots: 允许落脚的根目录（或单文件）列表；根目录解析后若是文件，
            要求 target 与之精确相等，否则要求 target 位于目录内部。
    返回：规范化后的绝对路径字符串，供后续 pandas 直接读取。
    异常：HTTPException 404（文件不存在）/ 403（路径越出受控目录）。
    """
    target = Path(path).resolve()
    if not target.is_file():
        raise HTTPException(status_code=404, detail='训练数据文件不存在')
    def allowed(root: Path) -> bool:
        resolved_root = root.resolve()
        # 根目录解析后若是"文件"（如 DEFAULT_DATA），只允许精确指向该文件本身
        return target == resolved_root if resolved_root.is_file() else _is_within(target, resolved_root)

    # 任一受控根匹配即放行；全部不匹配说明路径越界，按 403 拒绝
    if not any(allowed(root) for root in allowed_roots):
        raise HTTPException(status_code=403, detail='旧 data_path 只能指向受控数据目录')
    return str(target)


@dataclass(frozen=True)
class TrainingDataReference:
    """已解析的主数据/独立测试数据引用及其上传文件名。

    主数据与独立测试集各自只取两种形态之一：dataset_id（新接口，指向已登记的
    上传数据集）或 legacy_path（旧接口，指向受控目录内的文件），二者互斥，
    未使用的一侧为 None。dataset_name 保留用户上传时的原始文件名，仅用于
    结果页与 Manifest 展示，不参与任何路径解析。
    """
    dataset_id: str | None
    legacy_path: str | None
    dataset_name: str
    test_dataset_id: str | None
    test_legacy_path: str | None
    test_dataset_name: str | None


def _dataset_repository() -> DatasetRepository:
    """创建并初始化指向共享 datasets.sqlite3 的仓储对象。

    每次调用新建轻量仓储：连接按操作打开，initialize() 只保证表结构存在。
    与 get_run_repository 同为"按请求建仓"风格，避免跨请求共享连接状态。
    """
    repository = DatasetRepository(DATASETS_DATABASE, storage_root=STORAGE_DIR)
    repository.initialize()
    return repository


def resolve_training_data_reference(payload: TrainingRunRequest, *, principal: Principal) -> TrainingDataReference:
    """校验互斥引用并解析一次训练所需的主数据和可选测试数据。

    校验顺序（fail-fast：先做便宜的互斥/模式检查，再触库解析）：
    1. dataset_id 与 data_path 互斥；test_dataset_id 与 test_data_path 互斥。
    2. server 模式（Principal 带 owner_id 或 tenant_id）必须使用 dataset_id /
       test_dataset_id，禁止浏览器直传路径，也禁止缺省回退根目录 data.csv。
    3. local 模式才允许 data_path，且路径必须落在受控根目录内。

    参数：
        payload: 训练请求体（TrainingRunRequest）。
        principal: 服务端注入的调用者身份；dataset_id 解析时按它做作用域隔离，
            防止跨 owner/tenant 读取他人数据集。
    返回：TrainingDataReference，主数据与可选测试数据各取 id 或路径之一。
    异常：HTTPException 422（互斥冲突 / server 模式违反引用约束）；路径相关的
        404/403 由 resolve_legacy_dataset_path 抛出；dataset_id 不存在或越权
        由 DatasetRepository.resolve 抛出。
    """
    # owner_id/tenant_id 任一存在即视为服务器受控作用域，身份约束随之收紧
    server_scoped = principal.owner_id is not None or principal.tenant_id is not None
    # 互斥约束 1：主数据只允许一种引用方式
    if payload.dataset_id and payload.data_path:
        raise HTTPException(status_code=422, detail='dataset_id 与 data_path 不能同时提供')
    # 互斥约束 2：独立测试集同理
    if payload.test_dataset_id and payload.test_data_path:
        raise HTTPException(status_code=422, detail='test_dataset_id 与 test_data_path 不能同时提供')
    # server 模式：禁止 data_path，且必须显式给出 dataset_id（不回退默认 data.csv）
    if server_scoped and (payload.data_path or not payload.dataset_id):
        raise HTTPException(status_code=422, detail='服务器模式必须使用上传接口返回的 dataset_id')
    # server 模式：独立测试集同样只接受 test_dataset_id
    if server_scoped and payload.test_data_path:
        raise HTTPException(status_code=422, detail='服务器模式的独立测试集必须使用 test_dataset_id')

    datasets = _dataset_repository()
    if payload.dataset_id:
        # 新接口：按 Principal 作用域解析已登记数据集，取上传时的原始文件名
        dataset = datasets.resolve(payload.dataset_id, principal=principal)
        dataset_id = payload.dataset_id
        legacy_path = None
        dataset_name = dataset.original_name
    else:
        # 旧接口（仅 local）：未传路径时缺省回退根目录 data.csv，但必须过受控根校验
        path = payload.data_path or str(DEFAULT_DATA)
        dataset_id = None
        legacy_path = resolve_legacy_dataset_path(
            path,
            allowed_roots=[UPLOADS_DIR, PREPROCESSED_DIR, DEFAULT_DATA],
        )
        dataset_name = Path(legacy_path).name

    if payload.test_dataset_id:
        # 独立测试集走与主数据相同的作用域解析规则
        test_dataset = datasets.resolve(payload.test_dataset_id, principal=principal)
        test_dataset_id = payload.test_dataset_id
        test_legacy_path = None
        test_dataset_name = test_dataset.original_name
    elif payload.test_data_path:
        test_dataset_id = None
        test_legacy_path = resolve_legacy_dataset_path(
            payload.test_data_path,
            allowed_roots=[UPLOADS_DIR, PREPROCESSED_DIR, DEFAULT_DATA],
        )
        test_dataset_name = Path(test_legacy_path).name
    else:
        # 未提供独立测试集：三个测试字段全部为 None，由训练侧决定评估口径
        test_dataset_id = None
        test_legacy_path = None
        test_dataset_name = None

    return TrainingDataReference(
        dataset_id=dataset_id,
        legacy_path=legacy_path,
        dataset_name=dataset_name,
        test_dataset_id=test_dataset_id,
        test_legacy_path=test_legacy_path,
        test_dataset_name=test_dataset_name,
    )


def get_run_repository() -> RunRepository:
    """为当前请求创建指向共享 SQLite 文件的轻量仓库对象。

    RunRepository 按请求新建：initialize() 保证 runs.sqlite3 表结构就绪，
    具体连接的打开/关闭由仓储内部按操作管理，不在请求之间共享。
    """
    repository = RunRepository(RUNS_DATABASE)
    repository.initialize()
    return repository


def get_run_dir(run_id: str) -> Path:
    """校验 run_id 是安全单段名称并解析其受控产物目录。

    为什么校验：run_id 来自 URL 路径参数，若允许 '..'、嵌套分隔符或空串，
    下载/删除接口就能借 RUNS_DIR / run_id 逃出产物根目录。这里要求
    Path(run_id).name == run_id（不含任何目录分隔符）且不是 '.' / '..'；
    不满足一律按 404 处理，不向客户端暴露路径细节。
    """
    if not run_id or Path(run_id).name != run_id or run_id in {'.', '..'}:
        raise HTTPException(status_code=404, detail='run 不存在')
    return RUNS_DIR / run_id


def get_agent_service(contract_version: str = 'agent-session-v1', *, storage=None) -> AgentService:
    """每次请求新建一个轻量服务实例，Repository 是 SQLite 句柄集合可以共享。

    与现有 ``get_run_repository`` 风格一致；Repository 本身不带跨请求状态。
    """
    from .. import paths
    from ..agent.repository import AgentSessionRepository
    from ..agent.service import AgentService
    from ..runs.submission import RunSubmissionService
    from ..runs.status_projection import project_status
    source = storage if storage is not None else paths
    AGENT_DATABASE, RUNS_DATABASE = source.AGENT_DATABASE, source.RUNS_DATABASE
    DATASETS_DATABASE, STORAGE_DIR, RUNS_DIR = source.DATASETS_DATABASE, source.STORAGE_DIR, source.RUNS_DIR
    sessions = AgentSessionRepository(AGENT_DATABASE)
    sessions.initialize()
    runs = RunRepository(RUNS_DATABASE)
    runs.initialize()
    datasets = DatasetRepository(DATASETS_DATABASE, storage_root=STORAGE_DIR)
    datasets.initialize()
    return AgentService(
        session_repository=sessions,
        run_repository=runs,
        dataset_repository=datasets,
        submission_service=RunSubmissionService(
            run_repository=runs,
            dataset_repository=datasets,
            run_dir=lambda run_id: RUNS_DIR / run_id,
            status_projector=project_status,
        ),
        run_root=RUNS_DIR, contract_version=contract_version,
    )

"""上传数据集的 SQLite 元数据仓库。

【模块职责】
本文件是 datasets 子包的核心实现：负责"用户上传的数据集"在服务端的全生命周期元数据管理，
包括注册（register）、按身份范围解析（resolve）、系统级解析（resolve_system）
以及内容完整性校验（verify_integrity）。

【在系统中的位置】
整体链路为：浏览器上传文件 → 保存到受控 storage 目录 → 本仓库登记 dataset_id 元数据 →
训练接口只接受 dataset_id（server 模式下禁止浏览器直接传 data_path）→
创建 Run 时通过本仓库把 dataset_id 解析回磁盘路径并记录哈希快照。
因此本模块是"训练只引用稳定数据集，不引用任意路径"这一安全契约的执行者。

【协作模块】
- ..runs.contracts.Principal：服务端注入的调用者身份（owner_id/tenant_id），
  本模块用它做多租户/多用户的数据隔离，身份绝不来自请求体。
- 上层 API 路由（datasets 上传接口、training 创建 Run 接口）调用本仓库。
- 数据文件本身存放在 storage/uploads、storage/preprocessed 等受控目录。

【关键设计约束】
- 数据文件保存在受控 storage 目录，数据库只保存稳定 dataset_id、绝对路径、
  原文件名、内容哈希和服务端身份范围。
- 读取记录时会同时验证路径仍位于允许根目录，防止数据库内容被利用为任意文件读取入口。
- SQLite 仅作元数据索引；真正的光谱/色谱数据（wide-feature 宽表）始终在文件系统中。
"""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

# Principal 是服务端注入的调用者身份（owner_id/tenant_id），
# 用于数据集记录的归属隔离；注意它只来自服务端，绝不能从请求体读取。
from ..runs.contracts import Principal


@dataclass(frozen=True)
class DatasetRecord:
    """一个已持久化、可被训练 Run 稳定引用的数据集。

    这是数据集元数据在内存中的不可变表示（frozen=True 防止运行期被意外改写）。
    属性含义：
    - dataset_id: 稳定的数据集标识（形如 ``ds_<uuid hex>``），训练 Run 只引用它。
    - path: 数据文件在磁盘上的绝对路径（已 resolve）。
    - sha256: 注册时计算的文件内容哈希快照，用于之后检测"文件被换掉"。
    - original_name: 用户上传时的原始文件名，仅作展示，不参与定位。
    - owner_id / tenant_id: 服务端身份范围；两者共同决定该记录对哪个 Principal 可见。
    """
    dataset_id: str
    path: Path
    sha256: str
    original_name: str
    owner_id: str | None
    tenant_id: str | None


class DatasetIntegrityError(ValueError):
    """数据文件内容与 Run 创建时保存的哈希快照不一致。

    继承 ValueError 是为了让上层能把"数据被改动"与"路径不存在"（FileNotFoundError）、
    "越权访问"（PermissionError）区分开，分别映射为不同的 HTTP 错误。
    """


def _sha256(path: Path) -> str:
    """以 1MiB 分块流式计算文件的 SHA-256，避免一次性把大光谱文件读入内存。

    参数 path: 目标文件路径。返回值: 小写十六进制哈希字符串。
    ``iter(lambda: ..., b'')`` 是惯用法：反复读块直到读到空字节串（EOF）。
    """
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    """判断 path 是否等于 root 或位于 root 目录之内（纯路径前缀判断，不做 IO）。

    这是"受控目录"防线的基础构件；调用前双方都应当已 resolve，
    以免 ``..`` 等相对成分绕过检查。
    """
    return path == root or root in path.parents


class DatasetRepository:
    """管理 dataset schema、注册、按 Principal 查询与路径约束。

    设计要点：
    - 所有 SQL 都使用参数化占位符 ``?``，杜绝 SQL 注入。
    - 每次操作新开一个短连接（sqlite3 在单文件场景下足够简单可靠），
      ``with connection`` 退出时自动 commit/rollback。
    - 所有对外方法都先做"路径必须位于受控根目录"或"Principal 身份范围"检查，
      保证数据库记录不能被当成任意文件读取/引用的跳板。
    """
    def __init__(self, database_path: Path, *, storage_root: Path) -> None:
        # database_path: SQLite 元数据库文件路径；storage_root: 数据文件受控根目录。
        # storage_root 立即 resolve 成绝对路径，后续所有包含性判断都以它为准。
        self.database_path = Path(database_path)
        self.storage_root = Path(storage_root).resolve()
        # 确保数据库文件所在目录存在（首次部署时 storage 目录可能还没建）。
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def _connection(self) -> sqlite3.Connection:
        # timeout=30 秒：避免并发写时立刻抛 database is locked，给锁等待留出余量。
        # row_factory=Row 让查询结果可以按列名访问，配合 _record 构造 dataclass。
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        """建表（幂等）。服务启动时调用一次；IF NOT EXISTS 保证重复调用无副作用。

        表结构刻意只存元数据：主键 dataset_id、文件绝对路径、SHA-256 快照、
        原始文件名、归属身份（owner_id/tenant_id，历史记录可空）和创建时间。
        """
        with self._connection() as connection:
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    owner_id TEXT,
                    tenant_id TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                '''
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> DatasetRecord:
        """把 SQLite 行转换成 DatasetRecord；path 统一 resolve 成绝对路径，
        让后续 ``is_file`` 与受控路径判断不依赖当前工作目录。"""
        return DatasetRecord(
            dataset_id=str(row['dataset_id']),
            path=Path(row['path']).resolve(),
            sha256=str(row['sha256']),
            original_name=str(row['original_name']),
            owner_id=row['owner_id'],
            tenant_id=row['tenant_id'],
        )

    def register(self, path: Path, *, original_name: str, principal: Principal) -> DatasetRecord:
        """注册一个已落入受控目录的数据文件，返回带稳定 dataset_id 的记录。

        参数：path 数据文件路径；original_name 用户上传时的原始文件名；
        principal 服务端身份，记录的归属范围直接取自它。
        异常：文件不存在 → FileNotFoundError；路径不在受控目录 → PermissionError。
        """
        target = Path(path).resolve()
        if not target.is_file():
            raise FileNotFoundError(target)
        # 关键安全闸：只允许登记受控 storage 内的文件，防止把任意服务器路径登记成数据集。
        if not self._is_controlled_path(target):
            raise PermissionError(f'dataset path is outside controlled storage: {target}')
        # dataset_id 形如 ds_<32位hex>，全局唯一且不可猜测，训练 Run 只引用这个 ID。
        dataset_id = f'ds_{uuid.uuid4().hex}'
        record = DatasetRecord(
            dataset_id=dataset_id,
            path=target,
            # 注册即计算内容哈希快照，之后 verify_integrity 靠它发现文件被替换。
            sha256=_sha256(target),
            original_name=original_name,
            owner_id=principal.owner_id,
            tenant_id=principal.tenant_id,
        )
        with self._connection() as connection:
            connection.execute(
                '''
                INSERT INTO datasets (dataset_id, path, sha256, original_name, owner_id, tenant_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (
                    record.dataset_id,
                    str(record.path),
                    record.sha256,
                    record.original_name,
                    record.owner_id,
                    record.tenant_id,
                ),
            )
        return record

    def resolve(self, dataset_id: str, *, principal: Principal) -> DatasetRecord:
        """按 dataset_id 取出记录，并强制校验调用者 Principal 的归属范围。

        这是 server 模式下训练接口取数据的入口：身份不匹配直接 PermissionError，
        实现多用户/多租户之间的数据隔离（身份只来自服务端注入，不信请求体）。
        """
        with self._connection() as connection:
            row = connection.execute('SELECT * FROM datasets WHERE dataset_id = ?', (dataset_id,)).fetchone()
        if row is None:
            raise FileNotFoundError(f'dataset {dataset_id} does not exist')
        record = self._record(row)
        # 归属校验：owner 与 tenant 都必须完全一致，历史 owner 为空的记录因此默认不可见。
        if record.owner_id != principal.owner_id or record.tenant_id != principal.tenant_id:
            raise PermissionError(f'dataset {dataset_id} is outside the principal scope')
        # 元数据存在但文件已被删除/移动时，视为不存在，避免训练读到悬空路径。
        if not record.path.is_file():
            raise FileNotFoundError(record.path)
        return record

    def resolve_system(self, dataset_id: str | None, *, legacy_path: str | None) -> DatasetRecord:
        """服务端内部使用的解析入口，不做 Principal 归属校验。

        两种来源按优先级处理：
        1. dataset_id：查库取记录（同样要求文件仍存在），用于 local 模式等
           由服务端自己持有身份的场景。
        2. legacy_path：兼容历史直接传路径的调用；路径必须落在受控目录内，
           并即时构造一条 owner/tenant 为 None 的临时记录（dataset_id 为空串）。
        两者都缺失时抛 FileNotFoundError。因为是内部入口，调用方必须自己保证
        已经做过身份/权限判断，不能把它暴露给未经鉴权的请求。
        """
        if dataset_id:
            with self._connection() as connection:
                row = connection.execute('SELECT * FROM datasets WHERE dataset_id = ?', (dataset_id,)).fetchone()
            if row is None:
                raise FileNotFoundError(f'dataset {dataset_id} does not exist')
            record = self._record(row)
            if not record.path.is_file():
                raise FileNotFoundError(record.path)
            return record
        if not legacy_path:
            raise FileNotFoundError('no dataset reference supplied')
        target = Path(legacy_path).resolve()
        if not target.is_file():
            raise FileNotFoundError(target)
        # legacy 路径同样不能越出受控目录，防止读取服务器上任意文件。
        if not self._is_controlled_path(target):
            raise PermissionError(f'dataset path is outside controlled storage: {target}')
        return DatasetRecord(
            dataset_id='',
            path=target,
            sha256=_sha256(target),
            original_name=target.name,
            owner_id=None,
            tenant_id=None,
        )

    @staticmethod
    def verify_integrity(record: DatasetRecord, *, expected_sha256: str | None = None) -> str:
        """重新计算文件哈希，并同时校验注册记录和可选 Run 快照。

        参数 expected_sha256：创建 Run 时保存的内容快照（可空）；
        不传时退化为只校验注册时的哈希。返回实际计算出的哈希。
        两处不一致都抛 DatasetIntegrityError，提示用户重新上传。
        """
        actual = _sha256(record.path)
        # expected 优先取 Run 快照，否则用注册哈希作为基准。
        expected = expected_sha256 or record.sha256
        if expected and actual != expected:
            raise DatasetIntegrityError(
                f'dataset {record.dataset_id or record.original_name} 内容已变化；请重新上传后创建 Run'
            )
        # 双保险：即使没传 Run 快照，也要保证文件仍与注册时一致。
        if record.sha256 and actual != record.sha256:
            raise DatasetIntegrityError(
                f'dataset {record.dataset_id or record.original_name} 与注册哈希不一致'
            )
        return actual

    def snapshot(
        self,
        record: DatasetRecord,
        *,
        dataset_id: str | None = None,
        name: str | None = None,
    ) -> dict[str, str | None]:
        """生成 Run 创建时使用的统一数据集指纹快照。

        解析和哈希仍由本仓库负责，Agent Adapter 与训练路由共享同一套
        ``verify_integrity`` 逻辑，避免两个入口记录不同的文件指纹。
        """
        actual_sha256 = self.verify_integrity(record)
        return {
            'dataset_id': dataset_id or record.dataset_id,
            'name': name or record.original_name,
            'sha256': actual_sha256,
        }

    def _is_controlled_path(self, path: Path) -> bool:
        """判断路径是否位于任一"允许根"之内。

        允许范围刻意收窄为四类：storage 根目录本身、uploads、preprocessed，
        以及仓库根目录下的 data.csv（本地验证数据的历史例外）。
        注意每个 root 都先 resolve，保证与调用方 resolve 过的 path 可比较。
        """
        allowed_roots = (
            self.storage_root,
            self.storage_root / 'uploads',
            self.storage_root / 'preprocessed',
            self.storage_root.parent / 'data.csv',
        )
        return any(_is_within(path, root.resolve()) for root in allowed_roots)

"""建模 CSV 上传路由。

上传文件先落到随机命名的受控目录，再做完整解析校验，最后注册稳定 dataset_id。
失败时删除本次临时上传，避免无效文件进入可引用的数据集目录。
"""

# ---------------------------------------------------------------------------
# 模块说明（教学注释）
#
# 本文件是数据集上传的 HTTP 路由层，负责把浏览器上传的建模 CSV 变成可被训练
# 引用的“受管数据集”。它在整个系统中的位置：
#   前端（AI 建模页）上传 CSV → 本路由保存+校验+注册 → 得到稳定 dataset_id →
#   训练创建接口（routers/runs.py 的 create_run）在 server 模式下只接受该
#   dataset_id，绝不接受浏览器直传的文件路径。
#
# 协作模块：
#   - parsers.summarize_modeling_csv：完整解析 CSV，校验 wide-feature-v1/v2
#     宽表结构（Label/Sample_ID 等列、唯一递增的坐标表头等），产出前端摘要；
#   - datasets.repository.DatasetRepository：把校验通过的文件登记进
#     datasets.sqlite3，分配稳定 dataset_id 并记录归属 Principal；
#   - http.principal.get_principal：服务端注入身份，上传的数据集按
#     owner/tenant 隔离，请求体不接受身份字段；
#   - paths.UPLOADS_DIR / STORAGE_DIR / DATASETS_DATABASE：受控存储位置。
#
# 关键设计约束：
#   - “先落盘到随机名 → 再校验 → 后注册”的三段式流程：任何一步失败都删除
#     临时文件，保证无效 CSV 永远不会以任何身份进入数据集目录。
#   - 响应中服务器本地路径只在 local 模式给出；server 模式只返回稳定 ID，
#     并从摘要里剥掉 path，避免把主机目录结构泄露到浏览器。
# ---------------------------------------------------------------------------

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from ..datasets.repository import DatasetRepository
from ..http.principal import get_principal
from ..parsers import summarize_modeling_csv
from ..paths import DATASETS_DATABASE, STORAGE_DIR, UPLOADS_DIR
from ..runs.contracts import Principal

router = APIRouter()


@router.post('/api/datasets/upload')
def upload_dataset(
    request: Request,
    file: UploadFile = File(...),
    principal: Principal = Depends(get_principal),
) -> dict[str, object]:
    """保存并校验一个建模 CSV，返回稳定 ID 与前端摘要。"""
    # 第 1 段：落盘。文件名用 uuid4 随机生成（不信任客户端文件名，防路径
    # 注入与同名覆盖），只保留原扩展名；上传内容以二进制流式拷贝，
    # 不在内存中整读大文件。
    suffix = Path(file.filename or 'upload.csv').suffix or '.csv'
    target = UPLOADS_DIR / f'{uuid.uuid4().hex}{suffix}'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('wb') as handle:
        shutil.copyfileobj(file.file, handle)
    try:
        # 第 2 段：完整解析校验（宽表结构、表头坐标、Label/Sample_ID 等），
        # 并生成给前端预览用的统计摘要。
        summary = summarize_modeling_csv(target)
        # 第 3 段：注册进数据集仓库，分配稳定 dataset_id，归属当前 Principal。
        repository = DatasetRepository(DATASETS_DATABASE, storage_root=STORAGE_DIR)
        repository.initialize()
        dataset = repository.register(target, original_name=file.filename or target.name, principal=principal)
    except Exception as exc:
        # 校验或注册任一失败：删除本次临时文件，保证“坏数据不落库、不留盘”；
        # 统一以 400 返回解析/注册错误消息（均为面向用户的校验文案）。
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # 拷贝一份摘要再对外加工，避免污染内部对象。
    public_summary = dict(summary)
    response: dict[str, object] = {
        'dataset_id': dataset.dataset_id,
        'dataset_name': dataset.original_name,
        'summary': public_summary,
    }
    # 旧本机前端仍可使用受控路径兼容字段；服务器模式只公开稳定 ID，避免把
    # 主机目录结构带入浏览器或结果页面。
    if request.app.state.security_settings.mode == 'local':
        response['dataset_path'] = str(dataset.path)
    else:
        public_summary.pop('path', None)
    return response

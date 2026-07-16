"""建模 CSV 上传路由。

上传文件先落到随机命名的受控目录，再做完整解析校验，最后注册稳定 dataset_id。
失败时删除本次临时上传，避免无效文件进入可引用的数据集目录。
"""

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
    suffix = Path(file.filename or 'upload.csv').suffix or '.csv'
    target = UPLOADS_DIR / f'{uuid.uuid4().hex}{suffix}'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('wb') as handle:
        shutil.copyfileobj(file.file, handle)
    try:
        summary = summarize_modeling_csv(target)
        repository = DatasetRepository(DATASETS_DATABASE, storage_root=STORAGE_DIR)
        repository.initialize()
        dataset = repository.register(target, original_name=file.filename or target.name, principal=principal)
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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

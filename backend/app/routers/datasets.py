from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ..datasets.repository import DatasetRepository
from ..http.principal import get_principal
from ..parsers import summarize_modeling_csv
from ..paths import DATASETS_DATABASE, STORAGE_DIR, UPLOADS_DIR
from ..runs.contracts import Principal

router = APIRouter()


@router.post('/api/datasets/upload')
def upload_dataset(file: UploadFile = File(...), principal: Principal = Depends(get_principal)) -> dict[str, object]:
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
    return {'dataset_id': dataset.dataset_id, 'dataset_path': str(dataset.path), 'summary': summary}

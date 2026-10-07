"""Preview a strict Run/Batch without creating training artifacts."""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from ..contracts import TrainingBatchRequest, TrainingPreflightRequest, TrainingRunRequest
from ..http.principal import get_principal
from ..runs.contracts import Principal, public_error_message
from ..training_preflight import inspect_training_data, strict_spec
from ..version import WORKER_CONTRACT_VERSION
from .deps import get_run_repository, resolve_training_data_reference

router = APIRouter()


@router.post('/api/training/preflight')
def preflight(payload: TrainingPreflightRequest, principal: Principal = Depends(get_principal)):
    result = {'contract': 'training-preflight-v1', 'runnable': False, 'errors': [], 'warnings': []}
    try:
        if payload.task_type == 'run':
            if payload.model_types:
                raise ValueError('单 Run 请在 config.model_type 中选择模型')
            request = TrainingRunRequest(dataset_id=payload.dataset_id, test_dataset_id=payload.test_dataset_id,
                                         config=payload.config, strict_config=True)
            specs = [strict_spec(request.config, has_external_test=bool(payload.test_dataset_id))]
            request.config = specs[0].values
        else:
            from .batches import _batch_config
            request = TrainingBatchRequest(dataset_id=payload.dataset_id, test_dataset_id=payload.test_dataset_id,
                                           config=payload.config, model_types=payload.model_types,
                                           base_seed=payload.base_seed, strict_config=True)
            models, config, _ = _batch_config(request, has_external_test=bool(payload.test_dataset_id))
            # Seed fields belong to the Batch envelope rather than its config.
            request.model_types = models
            request.config = {key: value for key, value in config.items()
                              if key not in {'split_seed', 'feature_selection_enabled'}}
            specs = [strict_spec({**request.config, 'model_type': model, 'seed': payload.base_seed},
                                 has_external_test=bool(payload.test_dataset_id)) for model in models]
        data_ref = resolve_training_data_reference(request, principal=principal)
        data = inspect_training_data(data_ref, specs, principal=principal)
        worker = get_run_repository().worker_health(now=datetime.now(timezone.utc), expected_contract_version=WORKER_CONTRACT_VERSION)
        if not worker.get('available'):
            result['warnings'].append('Worker 当前不可用；配置有效，但客户端应等待 Worker 恢复后提交')
        if worker.get('available') and not worker.get('compatible'):
            result['errors'].append({'code': 'worker_contract_mismatch', 'message': 'Worker 与 Web 契约不兼容'})
        result.update(data=data, worker=worker, normalized_configs=[spec.values for spec in specs],
                      submit_payload=request.model_dump(exclude_none=True), run_count=len(specs),
                      feature_scheme_counts=[1 if spec.values['training_profile'] == 'quick' else
                                             4 if spec.values['model_type'] == 'cnn1d' else 7 for spec in specs])
        result['runnable'] = not result['errors']
    except (ValueError, HTTPException, FileNotFoundError, PermissionError) as exc:
        message = exc.detail if isinstance(exc, HTTPException) else str(exc)
        result['errors'].append({'code': 'invalid_training_plan', 'message': public_error_message(str(message))})
    return result

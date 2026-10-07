"""Read-only validation using the same partition builders as training."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .contracts import TrainingConfigValidationError, TrainingSpec
from .datasets.repository import DatasetRepository
from .parsers import load_modeling_csv
from .paths import DATASETS_DATABASE, STORAGE_DIR


def strict_spec(values: dict, *, has_external_test: bool) -> TrainingSpec:
    if values.get('split_mode') in {'external_test_holdout', 'leave_one_sample_id_cv_with_external_test'} and not has_external_test:
        raise TrainingConfigValidationError('所选评估模式要求独立测试集')
    spec = TrainingSpec.from_strict(values).validated(has_external_test=has_external_test)
    from .routers.catalog import get_models_catalog
    capabilities = {item['id']: item for item in get_models_catalog()['models']}
    model = capabilities[spec.values['model_type']]
    if not model['available']:
        raise TrainingConfigValidationError(model['unavailable_reason'])
    spec.values.setdefault('seed', 42)
    spec.values.setdefault('split_seed', spec.values['seed'])
    spec.values.setdefault('model_seed', spec.values['seed'])
    return spec


def inspect_training_data(data_ref, specs: list[TrainingSpec], *, principal) -> dict:
    from . import training as t
    from .training_experiments import grouped_experiment_folds

    datasets = DatasetRepository(DATASETS_DATABASE, storage_root=STORAGE_DIR)
    datasets.initialize()

    def load(dataset_id, path):
        record = datasets.resolve(dataset_id, principal=principal) if dataset_id else datasets.resolve_system(None, legacy_path=path)
        datasets.verify_integrity(record)
        return load_modeling_csv(record.path)

    data = load(data_ref.dataset_id, data_ref.legacy_path)
    external = load(data_ref.test_dataset_id, data_ref.test_legacy_path) if (data_ref.test_dataset_id or data_ref.test_legacy_path) else None
    labels = sorted(set(data.labels))
    if len(labels) < 2:
        raise ValueError('分类至少需要两个类别')
    y = np.asarray([labels.index(label) for label in data.labels], dtype=np.int64)
    groups = data.frame['Sample_ID'].astype(str).to_numpy()
    if external is not None:
        t._validate_external_test_dataset(labels, len(data.intensity[0]), data.x_axis[0], external)
        if set(groups).intersection(external.frame['Sample_ID'].astype(str)):
            raise ValueError('独立测试集与主数据包含相同 Sample_ID，不能跨集合使用同一样品')

    first = specs[0].values
    policy = t.resolve_evaluation_policy(first, has_external_test=external is not None)
    t._validate_split_ratio_config(policy)
    config = t.TrainConfig(**first)
    if policy.strategy in t._CV_EVALUATION_STRATEGIES:
        folds = t._leave_one_sample_id_folds(y, groups, config)
    else:
        splits = t._split_indices(y, groups, config, labels)
        required = ('train', 'valid') if external is not None else ('train', 'valid', 'test')
        t._validate_required_splits(splits, y, labels, required)
        folds = [{'fold_index': 1, 'splits': splits}]
    for fold in folds:
        t._validate_required_splits(fold['splits'], y, labels, ('train', 'valid'))
        for spec in specs:
            if spec.values['training_profile'] == 'full' and spec.values['model_type'] != 'cnn1d':
                pool = sorted(set(fold['splits']['train']) | set(fold['splits']['valid']))
                grouped_experiment_folds(y, groups, pool, config.split_seed, labels)
            if spec.values['model_type'] == 'cnn1d':
                length = len(data.intensity[0])
                scheme = spec.values['feature_scheme']
                if spec.values['training_profile'] == 'quick' and scheme.startswith('bin_'):
                    width = int(scheme.split('_')[1])
                    length = (length + width - 1) // width
                if length < 8:
                    raise ValueError('CNN 所选特征方案处理后的特征数不足 8，无法执行池化')

    def summary(dataset):
        frame = dataset.frame
        grouped = frame.drop_duplicates('Sample_ID')
        return {'measurement_count': len(frame), 'sample_count': len(grouped),
                'feature_count': len(dataset.intensity[0]),
                'class_sample_counts': dict(Counter(grouped['Label'].astype(str))),
                'class_measurement_counts': dict(Counter(dataset.labels))}

    def partition(indices):
        ids = set(groups[indices])
        return {'measurement_count': len(indices), 'sample_count': len(ids),
                'sample_ids': sorted(ids),
                'class_sample_counts': dict(Counter(group_labels[sid] for sid in ids))}

    group_labels = dict(zip(groups, data.labels))
    fold_summaries = [{'fold_index': fold['fold_index'],
                       'partitions': {name: partition(indices) for name, indices in fold['splits'].items()}}
                      for fold in folds]
    if policy.strategy == 'external_test_holdout':
        fold_summaries[0]['partitions']['test'] = {**summary(external), 'source': 'external_test',
                                                  'sample_ids': sorted(set(external.frame['Sample_ID'].astype(str)))}
    return {'primary': summary(data), 'external_test': summary(external) if external is not None else None,
            'evaluation_strategy': policy.strategy, 'fold_count': len(folds),
            'folds': fold_summaries}

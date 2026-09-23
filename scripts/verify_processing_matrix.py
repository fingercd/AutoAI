"""Execute every finite processing member on controlled, imbalanced data.

This is a component acceptance run. It does not produce formal experiment Runs
or Test estimates. Keep its JSON evidence under work/, outside Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from backend.app.models import build_traditional_model, model_family
from backend.app.processing_policy import EXECUTABLE_MODELS, legal_processing
from backend.app.training import (
    TrainConfig, _fit_deep_fold, _fit_x_normalizer, _transform_x_with_normalizer,
)


def controlled_data(class_count: int):
    rng=np.random.default_rng(5200+class_count)
    train_labels=[0]*8+[label for label in range(1,class_count) for _ in range(4)]
    valid_labels=list(range(class_count))
    y=np.asarray(train_labels+valid_labels,dtype=np.int64)
    x=rng.uniform(.25,1.25,size=(len(y),128)).astype(np.float32)
    x+=y[:,None]*np.linspace(.02,.1,128,dtype=np.float32)[None,:]
    return x,y,dict(train=list(range(len(train_labels))),
                    valid=list(range(len(train_labels),len(y))))


def exercise(model_type: str, normalization: str, class_balance: str,
             class_count: int, scratch: Path) -> None:
    x_raw,y,splits=controlled_data(class_count)
    normalizer=_fit_x_normalizer(x_raw[splits['train']],normalization)
    x=_transform_x_with_normalizer(x_raw,normalizer)
    config=TrainConfig(model_type=model_type,normalization=normalization,
        class_balance=class_balance,seed=42,epochs=1,batch_size=8,
        random_forest_n_estimators=5,xgboost_n_estimators=3)
    if model_family(model_type)=='traditional_ml':
        model=build_traditional_model(config,y[splits['train']],class_count)
        model.fit(x[splits['train']],y[splits['train']])
        prediction=np.asarray(model.predict(x[splits['valid']]))
        if prediction.shape!=(class_count,) or not np.isfinite(prediction).all():
            raise ValueError('traditional prediction invalid')
    else:
        scratch.mkdir(parents=True,exist_ok=True)
        model,history=_fit_deep_fold(config=config,model_type=model_type,x=x,y=y,
            splits=splits,label_names=[str(i) for i in range(class_count)],
            run_dir=scratch,sample_count=len(splits['train']))
        if not history or not np.isfinite(history[-1]['valid_loss']):
            raise ValueError('deep validation loss invalid')


def verify(output: Path) -> dict:
    torch.set_num_threads(1)
    records=[]
    for class_count in (2,3):
        x,y,splits=controlled_data(class_count)
        data_sha=hashlib.sha256(x.tobytes()+y.tobytes()+
            np.asarray(splits['train'],dtype=np.int64).tobytes()).hexdigest()
        for model_type in sorted(EXECUTABLE_MODELS):
            for normalization,class_balance in legal_processing(model_type,class_count=class_count):
                start=time.monotonic()
                row=dict(class_count=class_count,model_type=model_type,
                    normalization=normalization,class_balance=class_balance,
                    controlled_data_sha256=data_sha)
                try:
                    exercise(model_type,normalization,class_balance,class_count,
                        output.parent/'processing-matrix-tmp'/f'{class_count}-{model_type}-{normalization}-{class_balance}')
                    row['status']='passed'
                    row['error_type']=None
                except Exception as error:
                    row['status']='failed'
                    row['error_type']=type(error).__name__
                row['elapsed_seconds']=round(time.monotonic()-start,4)
                records.append(row)
    result=dict(schema_version='processing-execution-matrix-v1',
        scope='controlled component fit; no formal Run or Test estimate',
        planned=len(records),passed=sum(r['status']=='passed' for r in records),
        failed=sum(r['status']=='failed' for r in records),records=records)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('work/step5_local_acceptance/processing-matrix.json'))
    args=parser.parse_args()
    result=verify(args.output)
    print(json.dumps({key:result[key] for key in ('planned','passed','failed')}))
    return 0 if result['failed']==0 else 1


if __name__=='__main__':
    raise SystemExit(main())

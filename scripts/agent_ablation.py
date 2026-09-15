"""One reproducible ablation row against an existing HTTP service and worker.

This is a small resumable command, not a sweep scheduler. Credentials come only
from AUTOAI_API_TOKEN / AUTOAI_LLM_TOKEN. Public records never contain those values.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from contextlib import contextmanager

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    with temporary.open('w',encoding='utf-8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


@contextmanager
def experiment_lock(directory, experiment_id):
    """Linux server command: kernel lock releases automatically after a crash."""
    import fcntl
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/(experiment_id+'.lock')).open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def baseline(client, record, path, timeout):
    """An ambiguous ordinary POST is never automatically repeated."""
    if record['status'] in ('attempt_started','submission_uncertain') and not record.get('run_id'):
        record.update(status='submission_uncertain',failure_reason='ordinary_submission_requires_manual_reconciliation')
        save(path,record)
        return
    if not record.get('run_id'):
        record['status']='attempt_started';save(path,record)
        try:
            response=client.post('/api/training/runs',json=record['training_request'])
            response.raise_for_status()
            run_id=response.json()['run_id']
        except Exception:
            record.update(status='submission_uncertain',failure_reason='ordinary_submission_response_not_bound')
            save(path,record)
            return
        record.update(run_id=run_id,status='submitted');save(path,record)
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        response=client.get('/api/training/runs/'+record['run_id']);response.raise_for_status()
        state=response.json()['state']
        record['run_status']=state
        if state not in ('queued','running'):
            record.update(status='completed' if state=='succeeded' else state,
                failure_reason=None if state=='succeeded' else 'ordinary_training_'+state)
            save(path,record);return
        time.sleep(.2)
    record.update(status='submitted',failure_reason='poll_deadline');save(path,record)


def terminal_results(client,record):
    """Offline only: call after a fixed baseline or finalized Agent decision."""
    response=client.get('/api/training/runs/'+record['run_id']+'/result');response.raise_for_status()
    result=response.json()
    metrics=result.get('metrics',{})
    record['offline_test']={key:value for key,value in metrics.get('primary',{}).items()
        if key in ('accuracy','balanced_accuracy','macro_f1','macro_precision','macro_recall','weighted_f1')}
    validation=metrics.get('direct',{}).get('valid')
    if validation is not None:
        record['validation']={key:value for key,value in validation.items() if key in ('accuracy','balanced_accuracy','macro_f1','macro_precision','macro_recall','weighted_f1')}
    record['artifacts']=[{key:item[key] for key in ('name','download_url','integrity') if key in item}
        for item in result.get('artifacts',[])]
    # Actual partition and dataset hashes, without persisting sample IDs or paths.
    for item in record['artifacts']:
        if item.get('name') not in ('split.json','model_metadata.json') or not item.get('download_url'):continue
        if not item['download_url'].startswith('/api/training/runs/'+record['run_id']+'/'):
            raise ValueError('artifact URL is outside the bound Run')
        payload=client.get(item['download_url']);payload.raise_for_status();content=payload.json()
        if item['name']=='split.json':
            if isinstance(content,list):
                record['actual_split_digest']=digest([{k:fold[k] for k in ('fold_index','train_sample_ids','valid_sample_ids','test_sample_ids','partition_digest') if k in fold} for fold in content])
                record['partition_digests']=[fold.get('partition_digest','unknown') for fold in content]
        elif isinstance(content,dict):
            audit=content.get('execution_audit') or {}
            record['execution_digest']=digest(audit)
            if audit.get('evaluation_plan'):
                record['plan']=audit['evaluation_plan']
                record['dataset_digest']=audit['evaluation_plan']['dataset_sha256']
                record['actual_split_digest']=audit['partition_digest']


def run(args):
    from scripts.agent_step2_acceptance import code_binding
    import httpx
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',args.experiment_id):raise ValueError('invalid experiment ID')
    if args.kind=='agent' and (not args.llm_url or not args.llm_model):raise ValueError('LLM endpoint and model are required')
    configs=json.loads(args.model_configs.read_text()) if args.model_configs else {}
    configuration=dict(kind=args.kind,dataset_id=args.dataset_id,seed=args.seed,allowed_models=sorted(args.allowed_models),
        model_configs=configs,normalization='zscore',class_balance='none',decision_mode=args.decision_mode if args.kind=='agent' else 'fixed_baseline',
        evidence=not args.hide_evidence_context,risks=not args.hide_risk_context,knowledge=args.knowledge,
        max_runs=1,max_llm_calls=args.max_llm_calls,max_api_calls=args.max_api_calls,timeout=args.timeout,
        backend_binding=digest(args.backend_url),scope_binding=digest(args.scope_key))
    llm=None
    if args.kind=='agent':
        from agent_poc.orchestration.llm import LLMConfig
        llm=LLMConfig(args.llm_url,args.llm_model,protocol=args.protocol,timeout=180,max_tokens=1024,prompt_version='agent-decision-knowledge-v1')
        configuration['llm']=llm.public_config()
        configuration['llm_binding']=llm.fingerprint()
    folder=args.storage.resolve()/args.experiment_id;path=folder/'record.json'
    # Lock covers initial record creation and ordinary submission as well as Graph.
    with experiment_lock(args.storage.resolve(),args.experiment_id):
        if path.exists():
            record=json.loads(path.read_text())
            if record['configuration_digest']!=digest(configuration):raise ValueError('experiment configuration is frozen')
        else:
            code=code_binding()
            record=dict(experiment_id=args.experiment_id,configuration=configuration,configuration_digest=digest(configuration),
                source={key:code[key] for key in ('head','dirty_diff_sha256','source_digest')},
                thread_id=args.experiment_id,status='prepared',failure_reason=None,
                seed=args.seed,scope_binding=configuration['scope_binding'],dataset_digest='unknown',actual_split_digest='unknown',
                validation='unknown',llm_calls=0,api_calls='unknown',tokens='unknown',elapsed_seconds=0,artifacts=[],
                modules=dict(soft_evidence_filter='not_applicable',dynamic_preprocessing='unavailable'),
                label='same-domain structured Plain Agent' if args.kind=='agent' and args.decision_mode=='structured_config' and args.hide_evidence_context and args.hide_risk_context and args.knowledge=='off' else args.kind)
            if args.kind=='baseline':
                record['training_request']=dict(dataset_id=args.dataset_id,config=dict(model_type='logistic_regression',
                    normalization='zscore',class_balance='none',seed=args.seed,split_mode='stratified_holdout',
                    split_train=8,split_valid=1,split_test=1,feature_selection_enabled=False))
                record['baseline_policy']='LR fixed before Agent; retain existing internal search'
                record['tokens']=0
            else:
                from agent_poc.orchestration.graph import _key
                record['session_request_id']=_key('session',args.experiment_id)
                record['experiment_request_id']=_key('experiment',args.experiment_id)
            save(path,record)  # Experiment identity exists before any request.
        started=time.monotonic()
        token=os.environ.get('AUTOAI_API_TOKEN')
        headers={'Authorization':'Bearer '+token} if token else {}
        http_calls=[0]
        def count_request(request):http_calls[0]+=1
        with httpx.Client(base_url=args.backend_url,headers=headers,timeout=30,trust_env=False,event_hooks={'request':[count_request]}) as client:
            try:
                if record['status']!='completed':
                    if args.kind=='baseline':baseline(client,record,path,args.timeout)
                    else:
                        from agent_poc.orchestration.runtime import RuntimeConfig,start_task,resume_task,read_status
                        runtime=RuntimeConfig(args.backend_url,args.scope_key,llm,backend_token=token,
                            llm_token=os.environ.get('AUTOAI_LLM_TOKEN'),api_timeout=30)
                        checkpoints=folder/'checkpoints'
                        try:state=read_status(storage=checkpoints,thread_id=args.experiment_id)
                        except Exception as error:
                            # Only absent storage allows start; corrupt checkpoints must stop.
                            if any(checkpoints.glob('*.sqlite*')):raise
                            state=None
                        if state is None:
                            state=start_task(runtime,dataset_id=args.dataset_id,allowed_models=args.allowed_models,
                                model_configs=configs,storage=checkpoints,thread_id=args.experiment_id,task_id=args.experiment_id,
                                seed=args.seed,wait=True,knowledge=args.knowledge=='on',decision_mode=args.decision_mode,
                                evidence_context=not args.hide_evidence_context,risk_context=not args.hide_risk_context,
                                timeout_seconds=args.timeout,max_llm_calls=args.max_llm_calls,max_api_calls=args.max_api_calls)
                        else:state=resume_task(runtime,storage=checkpoints,thread_id=args.experiment_id,wait=True)
                        record.update(status=state['lifecycle']['status'],terminal_reason=state['lifecycle'].get('reason_code'),
                            failure_reason=None if state['lifecycle']['status']=='completed' else state['lifecycle'].get('reason_code'),
                            principal_fingerprint=state['identity']['principal_fingerprint'],
                            session_id=state['identity']['session_id'],session_request_id=state['identity']['session_request_id'],
                            experiment_request_id=state['identity']['experiment_request_id'],run_id=state['execution']['run_id'],
                            dataset_digest=state['task']['dataset_fingerprint'],actual_split_digest=state['task']['split_fingerprint'],
                            plan=state['recipes']['evaluation_plan'],validation=state['feedback'],
                            budget=state['budget'],llm_calls=state['budget']['llm_calls'],api_calls=state['budget']['api_calls'],
                            tokens={key:state['budget'][key] for key in ('input_tokens','output_tokens','cached_tokens')},versions=state['versions'],
                            frozen_model_configs=state['capabilities']['frozen_snapshot']['model_configs'])
                if record['status']=='completed' and record.get('run_id'):
                    terminal_results(client,record)
            except Exception as error:
                # Do not write exception messages, URLs, headers or raw responses.
                record['failure_reason']=type(error).__name__
                if record['status'] not in ('submission_uncertain','attempt_started'):record['status']='needs_attention'
            finally:
                record['record_http_calls']=record.get('record_http_calls',0)+http_calls[0]
                if args.kind=='baseline':record['api_calls']=record['record_http_calls']
                record['elapsed_seconds']+=time.monotonic()-started
                save(path,record)
        print(json.dumps(dict(experiment_id=args.experiment_id,status=record['status'],run_id=record.get('run_id'))))
        return 0 if record['status']=='completed' else 2


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-id',required=True)
    p.add_argument('--kind',choices=['baseline','agent'],required=True)
    p.add_argument('--backend-url',required=True)
    p.add_argument('--dataset-id',required=True)
    p.add_argument('--scope-key',required=True,help='Stable operator-provided scope identity; never a token')
    p.add_argument('--storage',type=Path,default=REPO/'work'/'ablation-records')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--allowed-models',nargs='+',default=['logistic_regression','svm'])
    p.add_argument('--model-configs',type=Path)
    p.add_argument('--decision-mode',choices=['recipe_id','structured_config'],default='recipe_id')
    p.add_argument('--knowledge',choices=['on','off'],default='off')
    p.add_argument('--hide-evidence-context',action='store_true')
    p.add_argument('--hide-risk-context',action='store_true')
    p.add_argument('--llm-url')
    p.add_argument('--llm-model')
    p.add_argument('--protocol',choices=['json_action','native_tools'],default='json_action')
    p.add_argument('--timeout',type=float,default=600)
    p.add_argument('--max-llm-calls',type=int,default=6)
    p.add_argument('--max-api-calls',type=int,default=180)
    return run(p.parse_args(argv))


if __name__=='__main__':raise SystemExit(main())


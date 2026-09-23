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
DECISION_TERMINAL_STATES=frozenset({'completed','failed','cancelled','timed_out','needs_attention'})


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
    """A nonblocking OS lock that releases when the process exits."""
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/(experiment_id+'.lock')).open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b'\0')
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            try:yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:yield
            finally:fcntl.flock(handle,fcntl.LOCK_UN)


def summarize_measurements(record):
    """Known values cover settled attempts only; interrupted intervals stay unknown."""
    legacy = record.get('legacy_measurements')
    known_calls = legacy.get('record_http_calls', 0) if legacy else 0
    known_seconds = legacy.get('elapsed_seconds', 0) if legacy else 0
    known_calls = known_calls if type(known_calls) is int else 0
    known_seconds = known_seconds if type(known_seconds) in (int, float) else 0
    incomplete = bool(legacy)
    for attempt in record['attempts']:
        measurement = attempt['measurement']
        if measurement['status'] == 'complete':
            known_calls += measurement['http_calls']
            known_seconds += measurement['elapsed_seconds']
        else:
            incomplete = True
    record['measurement'] = dict(status='incomplete' if incomplete else 'complete',
        known_http_calls=known_calls, known_elapsed_seconds=known_seconds,
        coverage='settled attempts plus recorded legacy values; interrupted intervals excluded',
        http_call_scope='recording client request hooks; Agent execution calls use the Graph journal')
    record['record_http_calls'] = 'unknown' if incomplete else known_calls
    record['elapsed_seconds'] = 'unknown' if incomplete else known_seconds
    if record['configuration']['kind'] == 'baseline':
        record['api_calls'] = record['record_http_calls']


def begin_attempt(record, path, source):
    """Persist identity/source before I/O, including rejected resume attempts."""
    if 'attempts' not in record:
        record['legacy_measurements'] = dict(record_http_calls=record.get('record_http_calls', 'unknown'),
            elapsed_seconds=record.get('elapsed_seconds', 'unknown'), status='unverified',
            reason='legacy record has no durable attempt settlement markers')
        record['attempts'] = []
    for previous in record['attempts']:
        if previous['status'] == 'running':
            previous.update(status='interrupted', reason='previous_attempt_not_settled')
            previous['measurement'] = dict(status='incomplete', http_calls='unknown', elapsed_seconds='unknown')
    allowed = source == record['source']
    attempt = dict(attempt_id=len(record['attempts'])+1, source=source,
        source_policy='exact originating CLI source binding required',
        remote_execution_versions=dict(web='unknown', worker='unknown'),
        started_at=time.time(), status='running' if allowed else 'rejected_source_change',
        measurement=dict(status='pending' if allowed else 'complete', http_calls=0, elapsed_seconds=0))
    record['attempts'].append(attempt)
    summarize_measurements(record)
    save(path, record)
    return attempt


def finish_attempt(record, path, attempt, calls, elapsed):
    attempt.update(status='settled', outcome=record['status'], finished_at=time.time(),
        measurement=dict(status='complete', http_calls=calls, elapsed_seconds=elapsed))
    summarize_measurements(record)
    save(path, record)


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


def summarize_plan(rows, records, *, factor='knowledge', sides=('on','off'), pair_conditions=None):
    """Offline, equally weighted runs. Missing/failed rows remain in denominators."""
    import math
    import statistics
    identities = [(r['task_id'], r['family'], r[factor], r['seed'], r['repeat']) for r in rows]
    if len(identities) != len(set(identities)) or len({r['experiment_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate experiment identity')
    if any(r['family'] not in ('ML', 'DL') or r[factor] not in sides for r in rows):
        raise ValueError('invalid planned cell')
    metrics = ('macro_f1', 'balanced_accuracy', 'accuracy')
    groups = {}
    values = {}
    for side in sides:
        planned = [r for r in rows if r[factor] == side]
        group = dict(planned=len(planned), succeeded=0, rows=[], metrics={})
        for row in planned:
            record = records.get(row['experiment_id'])
            successful = bool(record and record.get('status') == 'completed' and record.get('run_id'))
            observed = record.get('offline_test', {}) if successful else {}
            valid = successful and all(type(observed.get(m)) in (int, float) and math.isfinite(observed[m])
                                       and 0 <= observed[m] <= 1 for m in metrics)
            item = dict(**row, status=record.get('status') if record else 'not_started',
                        run_id=record.get('run_id') if record else None,
                        failure_reason=record.get('failure_reason') if record else 'not_started',
                        metrics=observed if valid else None)
            group['rows'].append(item)
            if valid:
                group['succeeded'] += 1
                values[(row['task_id'], row['family'], side, row['seed'], row['repeat'])] = observed
        for metric in metrics:
            aggregates = {}
            for family in ('ML', 'DL', 'all'):
                selected = [r for r in group['rows'] if family == 'all' or r['family'] == family]
                successful = [r['metrics'][metric] for r in selected if r['metrics'] is not None]
                conditions_ok=(pair_conditions is None or all(
                    pair_conditions[(r['task_id'],r['family'],r['seed'],r['repeat'])]['status']=='matched'
                    for r in selected))
                aggregates[family] = dict(planned=len(selected), denominator=len(successful),
                    complete_mean=statistics.mean(successful) if successful and len(successful) == len(selected) and conditions_ok else None,
                    successful_subset_mean=statistics.mean(successful) if successful else None,
                    run_sample_std=statistics.stdev(successful) if len(successful) > 1 else None)
            group['metrics'][metric] = aggregates
        groups[side] = group
    paired = []
    bases = sorted({(r['task_id'], r['family'], r['seed'], r['repeat']) for r in rows})
    for task, family, seed, repeat in bases:
        condition=(pair_conditions or {}).get((task,family,seed,repeat),{'status':'matched','reason':None})
        on = values.get((task, family, sides[0], seed, repeat))
        off = values.get((task, family, sides[1], seed, repeat))
        paired.append(dict(task_id=task, family=family, seed=seed, repeat=repeat,
                           condition_status=condition['status'],exclusion_reason=condition['reason'] if condition['status']!='matched' else None,
                           difference={m:on[m]-off[m] for m in metrics} if on and off and condition['status']=='matched' else None))
    complete = [p for p in paired if p['difference'] is not None]
    return dict(groups=groups, paired=paired, paired_count=len(complete), unpaired_count=len(paired)-len(complete),
                paired_mean_difference={m:statistics.mean(p['difference'][m] for p in complete) if complete else None for m in metrics},
                paired_complete_mean_difference={m:statistics.mean(p['difference'][m] for p in complete) if len(complete)==len(paired) else None for m in metrics},
                uncertainty_note='Run sample standard deviation; ML/DL for the same task are not independent datasets.')


def register_processing_plan(config, experiment_id):
    """Freeze the 10 x 2 x 2 comparison before any Agent decision."""
    from backend.app.processing_policy import EXECUTABLE_MODELS, freeze_fixed_processing
    from backend.app.models import model_family
    if type(config) is not dict or set(config) != {'tasks','model_pools','fixed_processing','seed','repeat','conditions'}:
        raise ValueError('invalid processing registration')
    tasks=config['tasks']
    pools=config['model_pools']
    if type(tasks) is not list or len(tasks)!=10 or type(pools) is not dict or set(pools)!={'ML','DL'}:
        raise ValueError('processing plan requires ten tasks and both model families')
    models={family:tuple(pool) for family,pool in pools.items()}
    if (set(models['ML'])|set(models['DL'])!=EXECUTABLE_MODELS or
            set(models['ML'])&set(models['DL']) or
            any(len(pool)!=len(set(pool)) for pool in models.values()) or
            any(model_family(model)!=('traditional_ml' if family=='ML' else 'deep_learning')
                for family,pool in models.items() for model in pool)):
        raise ValueError('processing model pools differ from executable families')
    fixed=freeze_fixed_processing(sorted(EXECUTABLE_MODELS),config['fixed_processing'])
    if type(config['seed']) is not int or config['seed']!=42 or type(config['repeat']) is not int or config['repeat']<0:
        raise ValueError('processing plan requires seed 42 and nonnegative repeat')
    seen=set()
    for task in tasks:
        if type(task) is not dict or set(task)!={'task_id','dataset_id','dataset_sha256'}:
            raise ValueError('invalid planned task')
        if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',task['task_id']) or
                not re.fullmatch(r'[a-f0-9]{64}',task['dataset_sha256']) or
                not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}',task['dataset_id'])):
            raise ValueError('invalid task identity')
        if task['task_id'] in seen:
            raise ValueError('duplicate planned task')
        seen.add(task['task_id'])
    rows=[]
    for task_index,task in enumerate(tasks):
        for family_index,family in enumerate(('ML','DL')):
            modes=('dynamic','fixed') if (task_index+family_index)%2==0 else ('fixed','dynamic')
            for mode in modes:
                rows.append(dict(experiment_id=f'{experiment_id}-{len(rows)+1:02d}',
                    task_id=task['task_id'],dataset_id=task['dataset_id'],
                    dataset_sha256=task['dataset_sha256'],family=family,processing_mode=mode,
                    seed=42,repeat=config['repeat'],allowed_models=list(models[family])))
    conditions=config['conditions']
    if (type(conditions) is not dict or set(conditions)!={
            'backend_binding','scope_binding','llm_binding','model_configs_digest'} or
            any(type(value) is not str or not re.fullmatch(r'[a-f0-9]{64}',value)
                for value in conditions.values()) or
            any('token' in str(key).lower() or 'secret' in str(key).lower()
                                                 for key in conditions)):
        raise ValueError('registration conditions must be public')
    result=dict(schema_version='processing-ablation-plan-v1',experiment_id=experiment_id,
        created_at=time.time(),rows=rows,model_pools={key:list(value) for key,value in models.items()},
        fixed_processing=fixed,conditions=conditions,
        comparison='dynamic_minus_fixed',test_collection='after_all_decisions')
    result['plan_digest']=digest({key:value for key,value in result.items() if key not in ('plan_digest','created_at')})
    return result


def _processing_pair_conditions(plan,records):
    checks={}
    for task_id,family,seed,repeat in sorted({
            (r['task_id'],r['family'],r['seed'],r['repeat']) for r in plan['rows']}):
        pair=[row for row in plan['rows'] if (row['task_id'],row['family'],row['seed'],row['repeat'])==
              (task_id,family,seed,repeat)]
        if len(pair)!=2 or {row['processing_mode'] for row in pair}!={'dynamic','fixed'}:
            raise ValueError('missing registered processing pair')
        left,right=(records.get(row['experiment_id']) for row in pair)
        expected_sha=pair[0]['dataset_sha256']
        reason=None
        if not left or not right:reason='record_missing'
        elif left.get('status')!='completed' or right.get('status')!='completed':reason='decision_not_completed'
        elif not left.get('run_id') or not right.get('run_id'):reason='run_missing'
        elif left['run_id']==right['run_id']:reason='run_reused'
        elif left.get('dataset_digest')!=expected_sha or right.get('dataset_digest')!=expected_sha:
            reason='dataset_mismatch'
        elif not left.get('actual_split_digest') or left['actual_split_digest']=='unknown' or not right.get('actual_split_digest') or right['actual_split_digest']=='unknown':
            reason='split_digest_unavailable'
        elif left['actual_split_digest']!=right['actual_split_digest']:reason='split_mismatch'
        elif not left.get('frozen_model_configs') or not right.get('frozen_model_configs'):
            reason='model_configs_unavailable'
        elif left['frozen_model_configs']!=right['frozen_model_configs']:reason='model_configs_mismatch'
        checks[(task_id,family,seed,repeat)]=dict(status='matched' if reason is None else
            'unavailable' if reason in ('record_missing','decision_not_completed','run_missing','split_digest_unavailable','model_configs_unavailable') else 'mismatch',
            reason=reason)
    return checks


def summarize_processing_plan(plan, records):
    checks=_processing_pair_conditions(plan,records)
    result=summarize_plan(plan['rows'],records,factor='processing_mode',sides=('dynamic','fixed'),pair_conditions=checks)
    distributions={}
    costs={}
    for mode in ('dynamic','fixed'):
        items=[records.get(row['experiment_id']) for row in plan['rows'] if row['processing_mode']==mode]
        counts={}
        for record in items:
            action=(record or {}).get('actual_processing') or {}
            if action.get('model_type'):
                key=(action.get('model_type'),action.get('normalization'),action.get('class_balance'))
                label='/'.join(str(part) for part in key)
                counts[label]=counts.get(label,0)+1
        distributions[mode]=counts
        costs[mode]=dict(planned=len(items),known_http_calls=sum(
            (record or {}).get('measurement',{}).get('known_http_calls',0) for record in items),
            unknown_count=sum(not record or (record.get('measurement') or {}).get('status')!='complete'
                              for record in items))
    condition_checks=[dict(task_id=task,family=family,seed=seed,repeat=repeat,**condition)
        for (task,family,seed,repeat),condition in checks.items()]
    condition_counts={status:sum(item['status']==status for item in condition_checks)
                      for status in ('matched','mismatch','unavailable')}
    result.update(processing_distribution=distributions,costs=costs,condition_checks=condition_checks,
        condition_counts=condition_counts,
        comparison='dynamic_minus_fixed',complete=(all(group['succeeded']==group['planned']
            for group in result['groups'].values()) and all(item['status']=='matched' for item in condition_checks)))
    return result


def load_processing_plan(path):
    plan=json.loads(path.read_text(encoding='utf-8'))
    if plan.get('schema_version')!='processing-ablation-plan-v1' or len(plan.get('rows',[]))!=40:
        raise ValueError('invalid processing plan')
    if plan.get('plan_digest')!=digest({key:value for key,value in plan.items()
                                       if key not in ('plan_digest','created_at')}):
        raise ValueError('processing plan digest mismatch')
    return plan


def _confirm_terminal_decision(storage,row,record):
    """A row record is only a projection; the checkpoint owns decision state."""
    if record.get('status') not in DECISION_TERMINAL_STATES:
        raise ValueError('all planned decisions must finish before Test collection')
    from agent_poc.orchestration.runtime import read_status
    try:
        state=read_status(storage=storage/row['experiment_id']/'checkpoints',
            thread_id=row['experiment_id'])
    except Exception as error:
        raise ValueError('decision checkpoint unavailable before Test collection') from error
    lifecycle=state.get('lifecycle') or {}
    identity=state.get('identity') or {}
    task=state.get('task') or {}
    recovery=state.get('recovery') or {}
    execution=state.get('execution') or {}
    if (identity.get('thread_id')!=row['experiment_id'] or
            task.get('dataset_id')!=row['dataset_id'] or
            task.get('dataset_fingerprint')!=record.get('dataset_digest') or
            task.get('processing_mode')!=row['processing_mode'] or
            lifecycle.get('status')!=record['status'] or
            lifecycle.get('status') not in DECISION_TERMINAL_STATES or
            lifecycle.get('next_action') is not None or
            lifecycle.get('ended_at') is None or
            recovery.get('needs_human_review') is not False or
            (recovery.get('pending_operation') or {}).get('status') in ('prepared','in_flight','unknown') or
            execution.get('run_id')!=record.get('run_id')):
        raise ValueError('decision checkpoint is not closed or differs from record')
    if record['status']=='completed':
        finalization=state.get('finalization') or {}
        if (record.get('dataset_digest')!=row['dataset_sha256'] or
                not record.get('run_id') or finalization.get('status')!='confirmed' or
                finalization.get('selected_run_id')!=record['run_id'] or
                finalization.get('backend_session_state')!='finalized'):
            raise ValueError('completed decision lacks confirmed finalization')


def collect_processing_plan(plan_path,storage,backend_url):
    """Read Test only after all 40 decisions have a durable terminal status."""
    import httpx
    plan=load_processing_plan(plan_path)
    if plan_path.resolve()!=storage.resolve()/plan['experiment_id']/'plan.json':
        raise ValueError('processing plan and row storage differ')
    if digest(backend_url)!=plan['conditions']['backend_binding']:
        raise ValueError('collection backend differs from registration')
    records={}
    for row in plan['rows']:
        path=storage/row['experiment_id']/'record.json'
        if not path.exists():
            raise ValueError('all planned decisions must finish before Test collection')
        record=json.loads(path.read_text(encoding='utf-8'))
        configuration=record.get('configuration') or {}
        if (record['experiment_id']!=row['experiment_id'] or
                configuration.get('processing_mode')!=row['processing_mode'] or
                configuration.get('plan_digest')!=plan['plan_digest'] or
                configuration.get('dataset_id')!=row['dataset_id'] or
                configuration.get('seed')!=row['seed'] or
                configuration.get('allowed_models')!=sorted(row['allowed_models']) or
                configuration.get('fixed_processing')!={model:plan['fixed_processing'][model]
                                                        for model in row['allowed_models']} or
                any(configuration.get(key)!=plan['conditions'][key]
                    for key in ('backend_binding','scope_binding','llm_binding')) or
                digest(configuration.get('model_configs'))!=plan['conditions']['model_configs_digest']):
            raise ValueError('record does not match processing plan')
        if record.get('dataset_digest') not in (row['dataset_sha256'], 'unknown'):
            raise ValueError('dataset digest differs from registered task')
        _confirm_terminal_decision(storage,row,record)
        records[row['experiment_id']]=record
    token=os.environ.get('AUTOAI_API_TOKEN')
    headers={'Authorization':'Bearer '+token} if token else {}
    with httpx.Client(base_url=backend_url,headers=headers,timeout=30,trust_env=False) as client:
        for row in plan['rows']:
            record=records[row['experiment_id']]
            if record['status']!='completed' or not record.get('run_id') or record.get('offline_test'):
                continue
            path=storage/row['experiment_id']/'record.json'
            with experiment_lock(storage,row['experiment_id']):
                try:
                    terminal_results(client,record)
                    record['offline_test_failure']=None
                except Exception as error:
                    # The row remains a failure; no Test value is synthesized.
                    record.pop('offline_test',None)
                    record['offline_test_failure']=type(error).__name__
                save(path,record)
    summary=summarize_processing_plan(plan,records)
    save(plan_path.with_name('summary.json'),summary)
    return summary


def validate_planned_row(plan_path,args):
    plan=load_processing_plan(plan_path)
    if plan_path.resolve()!=args.storage.resolve()/plan['experiment_id']/'plan.json':
        raise ValueError('processing plan and row storage differ')
    rows=[row for row in plan['rows'] if row['experiment_id']==args.experiment_id]
    if len(rows)!=1:
        raise ValueError('experiment is not in processing plan')
    row=rows[0]
    if (args.kind!='agent' or not args.defer_test or args.dataset_id!=row['dataset_id'] or
            args.seed!=row['seed'] or args.processing_mode!=row['processing_mode'] or
            set(args.allowed_models)!=set(row['allowed_models']) or args.knowledge!='off' or
            args.decision_mode!='recipe_id' or args.hide_evidence_context or args.hide_risk_context):
        raise ValueError('CLI row differs from registered processing conditions')
    if not args.fixed_processing:
        raise ValueError('registered fixed processing file is required')
    from backend.app.processing_policy import freeze_fixed_processing
    fixed=json.loads(args.fixed_processing.read_text(encoding='utf-8'))
    if freeze_fixed_processing(sorted(fixed),fixed)!=plan['fixed_processing']:
        raise ValueError('fixed processing differs from registration')
    return plan,row


def run(args):
    from scripts.agent_step2_acceptance import code_binding
    import httpx
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',args.experiment_id):raise ValueError('invalid experiment ID')
    if getattr(args,'plan',None):
        plan,row=validate_planned_row(args.plan,args)
    if args.kind=='agent' and (not args.llm_url or not args.llm_model):raise ValueError('LLM endpoint and model are required')
    configs=json.loads(args.model_configs.read_text()) if args.model_configs else {}
    processing_mode=getattr(args,'processing_mode',None)
    if processing_mode is None and getattr(args,'fixed_processing',None) is not None:
        raise ValueError('fixed_processing requires processing_mode')
    fixed_processing=None
    if processing_mode is not None:
        from backend.app.processing_policy import freeze_fixed_processing
        supplied=json.loads(args.fixed_processing.read_text()) if getattr(args,'fixed_processing',None) else None
        if getattr(args,'plan',None):
            supplied={model:supplied[model] for model in args.allowed_models}
        fixed_processing=freeze_fixed_processing(args.allowed_models,supplied)
    configuration=dict(kind=args.kind,dataset_id=args.dataset_id,seed=args.seed,allowed_models=sorted(args.allowed_models),
        model_configs=configs,normalization='zscore',class_balance='none',decision_mode=args.decision_mode if args.kind=='agent' else 'fixed_baseline',
        evidence=not args.hide_evidence_context,risks=not args.hide_risk_context,knowledge=args.knowledge,
        max_runs=1,max_llm_calls=args.max_llm_calls,max_api_calls=args.max_api_calls,timeout=args.timeout,
        backend_binding=digest(args.backend_url),scope_binding=digest(args.scope_key))
    if processing_mode is not None:
        configuration.pop('normalization')
        configuration.pop('class_balance')
        configuration.update(processing_mode=processing_mode,fixed_processing=fixed_processing,
            defer_test=bool(getattr(args,'defer_test',False)))
    if getattr(args,'plan',None):
        configuration['plan_digest']=plan['plan_digest']
    query_config = None
    if getattr(args, 'query_mode', None) is not None or getattr(args, 'query_text', None) is not None or getattr(args, 'confirmed_domain', None) is not None:
        from agent_poc.clients.contracts import KnowledgeQueryConfig
        query_config = KnowledgeQueryConfig(query_mode=getattr(args, 'query_mode', None) or 'train_template', user_text=getattr(args, 'query_text', None), domain=getattr(args, 'confirmed_domain', None)).model_dump(mode='json')
        configuration['knowledge_query'] = query_config
    llm=None
    if args.kind=='agent':
        from agent_poc.orchestration.llm import LLMConfig
        llm=LLMConfig(args.llm_url,args.llm_model,protocol=args.protocol,timeout=180,max_tokens=1024,
            prompt_version='agent-decision-processing-v1' if processing_mode is not None else 'agent-decision-knowledge-v1',
            tokenizer_path=getattr(args,'llm_tokenizer',None),context_window=getattr(args,'context_window',None))
        configuration['llm']=llm.public_config()
        configuration['llm_binding']=llm.fingerprint()
    if getattr(args,'plan',None):
        expected=plan['conditions']
        actual=dict(backend_binding=configuration['backend_binding'],scope_binding=configuration['scope_binding'],
                    llm_binding=configuration.get('llm_binding'),model_configs_digest=digest(configs))
        if actual!=expected:
            raise ValueError('CLI execution conditions differ from registration')
    folder=args.storage.resolve()/args.experiment_id;path=folder/'record.json'
    # Lock covers initial record creation and ordinary submission as well as Graph.
    with experiment_lock(args.storage.resolve(),args.experiment_id):
        code=code_binding()
        source={key:code[key] for key in ('head','dirty_diff_sha256','source_digest')}
        if path.exists():
            record=json.loads(path.read_text())
            if record['configuration_digest']!=digest(configuration):raise ValueError('experiment configuration is frozen')
        else:
            record=dict(experiment_id=args.experiment_id,configuration=configuration,configuration_digest=digest(configuration),
                source=source, attempts=[],
                thread_id=args.experiment_id,status='prepared',failure_reason=None,
                seed=args.seed,scope_binding=configuration['scope_binding'],dataset_digest='unknown',actual_split_digest='unknown',
                validation='unknown',llm_calls=0,api_calls='unknown',tokens='unknown',elapsed_seconds=0,artifacts=[],
                modules=dict(soft_evidence_filter='not_applicable',dynamic_preprocessing=processing_mode or 'unavailable'),
                label='same-domain structured Plain Agent' if args.kind=='agent' and args.decision_mode=='structured_config' and args.hide_evidence_context and args.hide_risk_context and args.knowledge=='off' else args.kind)
            if args.kind=='baseline':
                baseline_processing=(fixed_processing or {}).get('logistic_regression',
                    {'normalization':'zscore','class_balance':'none'})
                record['training_request']=dict(dataset_id=args.dataset_id,config=dict(model_type='logistic_regression',
                    normalization=baseline_processing['normalization'],class_balance=baseline_processing['class_balance'],
                    seed=args.seed,split_mode='stratified_holdout',
                    split_train=8,split_valid=1,split_test=1,feature_selection_enabled=False))
                record['baseline_policy']='LR fixed before Agent; retain existing internal search'
                record['tokens']=0
            else:
                from agent_poc.orchestration.graph import _key
                record['session_request_id']=_key('session',args.experiment_id)
                record['experiment_request_id']=_key('experiment',args.experiment_id)
            save(path,record)  # Experiment identity exists before any request.
        attempt=begin_attempt(record,path,source)
        if attempt['status']=='rejected_source_change':
            print(json.dumps(dict(experiment_id=args.experiment_id,status='source_changed')))
            return 2
        started=time.monotonic()
        token=os.environ.get('AUTOAI_API_TOKEN')
        headers={'Authorization':'Bearer '+token} if token else {}
        http_calls=[0]
        def count_request(request):http_calls[0]+=1
        try:
            with httpx.Client(base_url=args.backend_url,headers=headers,timeout=30,trust_env=False,event_hooks={'request':[count_request]}) as client:
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
                                seed=args.seed,wait=True,knowledge=args.knowledge=='on',decision_mode=args.decision_mode,knowledge_query=query_config,
                                processing_mode=processing_mode,fixed_processing=fixed_processing,
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
                            frozen_model_configs=(state['capabilities']['frozen_snapshot'] or {}).get('model_configs'),
                            knowledge_snapshot=state['knowledge']['snapshot'], decision=state['decision'],
                            actual_recipe=state['execution'].get('submission_content'))
                        action=state['execution'].get('effective_action') or {}
                        record['actual_processing']={key:action.get(key) for key in ('model_type','normalization','class_balance')}
                        record['candidate_count']=len((state.get('recipes',{}).get('catalog') or {}).get('recipes',[]))
                if record['status']=='completed' and record.get('run_id') and not getattr(args,'defer_test',False):
                    terminal_results(client,record)
        except Exception as error:
            # Do not write exception messages, URLs, headers or raw responses.
            record['failure_reason']=type(error).__name__
            if record['status'] not in ('submission_uncertain','attempt_started'):record['status']='needs_attention'
        finally:
            finish_attempt(record,path,attempt,http_calls[0],time.monotonic()-started)
        print(json.dumps(dict(experiment_id=args.experiment_id,status=record['status'],run_id=record.get('run_id'))))
        return 0 if record['status']=='completed' else 2


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=='register-processing':
        q=argparse.ArgumentParser(description='Freeze a forty-row processing comparison')
        q.add_argument('--config',type=Path,required=True)
        q.add_argument('--storage',type=Path,required=True)
        q.add_argument('--experiment-id',required=True)
        a=q.parse_args(argv[1:])
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,45}',a.experiment_id):
            raise ValueError('invalid plan ID')
        with experiment_lock(a.storage.resolve(),a.experiment_id):
            path=a.storage.resolve()/a.experiment_id/'plan.json'
            if path.exists():raise ValueError('processing plan already exists')
            plan=register_processing_plan(json.loads(a.config.read_text(encoding='utf-8')),a.experiment_id)
            save(path,plan)
        print(json.dumps(dict(plan=str(path),plan_digest=plan['plan_digest'],rows=len(plan['rows']))))
        return 0
    if argv and argv[0]=='collect-processing':
        q=argparse.ArgumentParser(description='Collect Test after all decisions finish')
        q.add_argument('--plan',type=Path,required=True)
        q.add_argument('--storage',type=Path,required=True)
        q.add_argument('--backend-url',required=True)
        a=q.parse_args(argv[1:])
        summary=collect_processing_plan(a.plan,a.storage,a.backend_url)
        print(json.dumps(dict(complete=summary['complete'],paired_count=summary['paired_count'])))
        return 0
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
    p.add_argument('--processing-mode',choices=['fixed','dynamic'])
    p.add_argument('--fixed-processing',type=Path)
    p.add_argument('--defer-test',action='store_true')
    p.add_argument('--plan',type=Path)
    p.add_argument('--knowledge',choices=['on','off'],default='off')
    p.add_argument('--query-mode', choices=['train_template','user_text'])
    p.add_argument('--query-text')
    p.add_argument('--confirmed-domain')
    p.add_argument('--llm-tokenizer')
    p.add_argument('--context-window',type=int)
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

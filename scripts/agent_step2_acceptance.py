"""Isolated real HTTP/worker acceptance for direct actions and recipes; artifacts stay outside Git.

Run from the repository using its actual acceptance Python. No existing service,
Run database or model environment is modified. Tokens exist only in process memory.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))


def bind_storage(root: Path):
    """Called before any app/router/repository/worker import in each child."""
    import backend.app.paths as p
    root=root.resolve()
    if root==(REPO/'storage').resolve() or not root.name.startswith(('autoai-step2-','autoai-step3-','autoai-step4-')):
        raise ValueError('acceptance storage must be an explicitly isolated root')
    storage=root/'storage'
    for key,value in dict(STORAGE_DIR=storage,RUNS_DATABASE=storage/'runs.sqlite3',
        DATASETS_DATABASE=storage/'datasets.sqlite3',AGENT_DATABASE=storage/'agent.sqlite3',
        UPLOADS_DIR=storage/'uploads',RUNS_DIR=storage/'runs',PREPROCESSED_DIR=storage/'preprocessed').items():
        setattr(p,key,value)
    return storage


def save(root,name,value):
    (root/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def code_binding():
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    diff=subprocess.check_output(['git','diff','HEAD','--'],cwd=REPO)
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','--',
        'backend/app','agent_poc','scripts'],cwd=REPO,text=True).splitlines()
    sources={name:hashlib.sha256((REPO/name).read_bytes()).hexdigest()
        for name in sorted(set(names)) if name.endswith(('.py','.json')) and (REPO/name).is_file()}
    return dict(head=head,dirty_diff_sha256=hashlib.sha256(diff).hexdigest(),
        source_digest=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),sources=sources)


def _role(args):
    if args.role in ('worker','baseline'):
        reject_agent_imports()
    bind_storage(args.root)
    if args.role=='baseline':
        return baseline_http(args)
    if args.role=='web':
        import uvicorn
        from backend.app.main import app
        import backend.app.main as entry
        save(args.root,'web-runtime.json',dict(code=code_binding(),pid=os.getpid(),python=sys.executable,entry=entry.__file__,storage=str(args.root/'storage')))
        uvicorn.run(app,host='127.0.0.1',port=args.port,access_log=False,log_level='warning')
    elif args.role=='worker':
        from backend.app.runs.worker import main
        import backend.app.runs.worker as entry
        save(args.root,'worker-runtime.json',dict(code=code_binding(),pid=os.getpid(),python=sys.executable,entry=entry.__file__,storage=str(args.root/'storage'),threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}))
        sys.argv=[sys.argv[0],'--poll-seconds','0.1']
        main()


def run(args):
    root=args.root.resolve()
    root.mkdir(parents=True,exist_ok=False)
    storage=bind_storage(root)
    import httpx
    from backend.tests.modeling_data_factory import write_grouped_classification_csv
    from backend.app.runs.repository import RunRepository
    from backend.app.runs.artifacts import RunArtifactWriter
    if not args.backend_only:
        from agent_poc.clients.autoai_client import AutoAIClient
        from agent_poc.tools import ToolDispatcher
        from agent_poc.orchestration.llm import LLMConfig,LLMAdapter
        from agent_poc.orchestration.runtime import RuntimeConfig,start_task,resume_task

    os.environ['NO_PROXY']='127.0.0.1,localhost'
    os.environ['no_proxy']='127.0.0.1,localhost'
    token=secrets.token_urlsafe(40)
    env=dict(os.environ,AUTOAI_DEPLOYMENT_MODE='server',AUTOAI_API_TOKEN=token,
        AUTOAI_PRINCIPAL_ID='step3-acceptance' if args.recipes else 'step2-acceptance',AUTOAI_TENANT_ID='step3' if args.recipes else 'step2',
        AUTOAI_ALLOWED_ORIGINS='http://127.0.0.1:'+str(args.port),
        PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(REPO),
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    env.pop('STEP2_ISOLATED_STORAGE',None)
    base='http://127.0.0.1:'+str(args.port)
    handles=[]; children=[]
    report=dict(code=code_binding(),node=socket.gethostname(),python=sys.executable,
        python_version=sys.version,python_realpath=str(Path(sys.executable).resolve()),root=str(root),storage_realpath=str(storage.resolve()),port=args.port,
        packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        dataset_sources=[],models=[],llm=[],cleanup={})
    report['package_paths']={name:importlib.util.find_spec(name).origin for name in ('torch','numpy','scipy','sklearn','xgboost','pydantic')+(() if args.backend_only else ('langgraph',))}
    report['mount']=subprocess.run(['findmnt','-T',str(root),'-n','-o','FSTYPE,SOURCE,TARGET'],capture_output=True,text=True).stdout.strip()
    try:
        # Refuse a port already used by any service before creating children.
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',args.port))
        for role in ('web','worker'):
            handle=(root/(role+'.log')).open('w');handles.append(handle)
            child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--role',role,
                '--root',str(root),'--port',str(args.port)],cwd=REPO,env=env,stdout=handle,stderr=subprocess.STDOUT)
            children.append(child);report[role+'_pid']=child.pid
        save(root,'runtime_manifest.json',report)
        deadline=time.monotonic()+90
        while True:
            if any(p.poll() is not None for p in children):
                raise RuntimeError('acceptance child exited')
            try:
                health=httpx.get(base+'/health',timeout=3,trust_env=False).json()
                report['startup_health']={k:health[k] for k in ('status','worker') if k in health}
                save(root,'runtime_manifest.json',report)
                if health['worker']['available'] and health['worker']['compatible']:break
            except (httpx.HTTPError,KeyError,ValueError):pass
            if time.monotonic()>deadline:raise RuntimeError('startup deadline')
            time.sleep(0.5)
        if not args.backend_only:
            def client():return AutoAIClient(base,token=token,api_version='v2',max_retries=0,timeout=30)
            c=client();dispatcher=ToolDispatcher(c)
            catalog=dispatcher.dispatch('inspect_ml_capabilities',{})
            report['capability_snapshot']=catalog
        report['web_health']=health
        datasets={}
        for labels in (('A','B'),('A','B','C')):
            source=write_grouped_classification_csv(root/('synthetic-'+str(len(labels))+'.csv'),groups_per_class=10,repeats=2,feature_count=128,labels=labels)
            with source.open('rb') as handle:
                response=httpx.post(base+'/api/datasets/upload',headers={'Authorization':'Bearer '+token},files={'file':(source.name,handle,'text/csv')},timeout=30,trust_env=False)
            response.raise_for_status()
            datasets[len(labels)]=response.json()['dataset_id']
            report['dataset_sources'].append(dict(classes=len(labels),dataset_id=datasets[len(labels)],
                source='deterministic engineering synthetic grouped curves; no stochastic generator',
                path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),groups_per_class=10,repeats=2,features=128))
        save(root,'runtime_manifest.json',report)
        def audit(run_id):
            writer=RunArtifactWriter(storage/'runs'/run_id)
            manifest, descriptors=writer.descriptors(run_id=run_id)
            complete=not any(d['integrity'] in ('missing','corrupt') and (d.get('required') or d.get('applicable')) for d in descriptors)
            metadata=json.loads(writer.resolve_download('model_metadata.json').read_text())
            return dict(manifest_complete=complete,manifest_path=str(storage/'runs'/run_id/'manifest.json'),
                execution_device=metadata.get('execution_device'),classification_head=metadata.get('classification_head'),
                resolved_profile=metadata.get('model_profile'))
        if args.backend_only:
            request=root/'native-backend.json'
            save(root,request.name,dict(dataset_id=datasets[3],config=dict(model_type='logistic_regression',
                normalization='zscore',class_balance='none',seed=42,split_mode='stratified_holdout',
                split_train=8,split_valid=1,split_test=1,feature_selection_enabled=False)))
            completed=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--role','baseline',
                '--root',str(root),'--port',str(args.port),'--request-file',str(request)],cwd=REPO,env=env,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=330)
            (root/'native-backend.log').write_bytes(completed.stdout+completed.stderr)
            assert completed.returncode==0,'native backend failed'
            baseline=json.loads((root/'native-backend-result.json').read_text())
            baseline['audit']=audit(baseline['run_id'])
            assert baseline['audit']['manifest_complete']
            import sqlite3
            counts={}
            if (storage/'agent.sqlite3').exists():
                with sqlite3.connect('file:'+str(storage/'agent.sqlite3')+'?mode=ro',uri=True) as db:
                    tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                    for table in ('agent_sessions_v1','agent_experiment_reservations_v1'):
                        if table in tables:counts[table]=db.execute('SELECT count(*) FROM '+table).fetchone()[0]
            assert not any(counts.values())
            baseline['agent_table_counts']=counts
            baseline['driver_forbidden_imports']=[m for m in sys.modules if m.startswith(('agent_poc','langgraph','backend.app.agent','backend.app.knowledge','openai'))]
            assert baseline['driver_forbidden_imports']==[]
            report['native_backend']=baseline;report['completed']=True
            return 0
        for classes in (() if args.recipes else (2,3)):
            for model in catalog['models']:
                name=model['id']
                if not model['available']:continue
                if classes==3 and name not in ('xgboost','pca_mlp','cnn1d','cnn_transformer1d','dscarnet'):continue
                fixed={'epochs':2} if model['execution_family']=='deep_learning' else {}
                created=dispatcher.dispatch('start_ml_session',dict(dataset_id=datasets[classes],selection_metric='macro_f1',
                    allowed_models=[name],max_runs=1,seed=42,model_configs={name:fixed},client_request_id=f'session-{classes}-{name}'))
                sid=created['session_id']; request=dict(session_id=sid,model_type=name,client_request_id=f'experiment-{classes}-{name}')
                submitted=dispatcher.dispatch('submit_ml_experiment',request)
                rid=submitted['run_id']; timeline=[dict(state=submitted['state'],at=datetime.now(timezone.utc).isoformat())]
                deadline=time.monotonic()+300
                while True:
                    observation=dispatcher.dispatch('observe_ml_experiment',dict(session_id=sid,run_id=rid))
                    if timeline[-1]['state']!=observation['state']:
                        timeline.append(dict(state=observation['state'],at=datetime.now(timezone.utc).isoformat()))
                    if observation['state'] not in ('queued','running'):break
                    if time.monotonic()>deadline:raise RuntimeError('model execution deadline')
                    time.sleep(0.5)
                evidence=dict(model=name,classes=classes,session_id=sid,run_id=rid,thread_id=None,timeline=timeline,
                    dataset_id=datasets[classes],frozen=created['locked_config'],observation=observation)
                if observation['state']=='succeeded' and observation['validation']['status']=='ready':
                    evidence['finalize']=dispatcher.dispatch('finalize_ml_session',dict(session_id=sid,selected_run_id=rid))
                    evidence['audit']=audit(rid)
                report['models'].append(evidence)
                save(root,'runtime_manifest.json',report)
                print(json.dumps(dict(model=name,classes=classes,state=observation['state'],run_id=rid)),flush=True)
                if 'finalize' not in evidence:raise RuntimeError('model acceptance failed')
        if args.recipes:
            recipe_acceptance(args,root,storage,base,token,env,datasets,report,audit)
        if args.llm_url and not args.recipes:
            choices=[m['id'] for m in catalog['models'] if m['available']]
            groups=[choices,[m for m in choices if m not in ('logistic_regression','svm','random_forest')]]
            llm_config=LLMConfig(args.llm_url,args.llm_model,timeout=180,max_tokens=1024,prompt_version='agent-decision-step2-v1')
            runtime=RuntimeConfig(base,'step2-acceptance',llm_config,backend_token=token,api_timeout=30)
            class AuditedLLM(LLMAdapter):
                def propose(self,phase,context,**kwargs):
                    proposal=super().propose(phase,context,**kwargs)
                    calls.append(dict(phase=phase,context_digest=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest(),
                        tool=proposal.tool_name,arguments=proposal.arguments,rationale=proposal.rationale,usage=proposal.usage.__dict__))
                    save(root,'llm-calls-'+thread+'.json',calls)
                    return proposal
            for index,allowed in enumerate(groups):
                thread='real-llm-'+str(index);calls=[]
                config={m['id']:{'epochs':2} for m in catalog['models'] if m['id'] in allowed and m['execution_family']=='deep_learning'}
                adapter=AuditedLLM(llm_config)
                checkpoint=root/'checkpoints'/thread
                final=start_task(runtime,dataset_id=datasets[2],allowed_models=allowed,model_configs=config,
                    storage=checkpoint,thread_id=thread,wait=True,llm=adapter,timeout_seconds=600,max_api_calls=180,execution_profile='direct_action')
                entry=dict(thread=thread,allowed_models=allowed,status=final['lifecycle']['status'],lifecycle=final['lifecycle'],
                    run_id=final['execution']['run_id'],session_id=final['identity']['session_id'],
                    chosen_model=final['execution']['submission_content']['model_type'] if final['execution']['submission_content'] else None,
                    budget=final['budget'],versions=final['versions'],calls=calls)
                report['llm'].append(entry);save(root,'runtime_manifest.json',report)
                save(root,thread+'-state.json',final)
                print(json.dumps(dict(thread=thread,status=entry['status'],chosen_model=entry['chosen_model'])),flush=True)
                if entry['status']!='completed':raise RuntimeError('real LLM acceptance failed')
                resumed=resume_task(runtime,storage=checkpoint,thread_id=thread)
                assert resumed==final
                entry['terminal_resume_unchanged']=True
                entry['audit']=audit(entry['run_id'])
        report['completed']=True
        return 0
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
        for handle in handles:handle.close()
        report['cleanup']={str(c.pid):c.poll() for c in children}
        report['evidence_retained']=True
        save(root,'runtime_manifest.json',report)



def reject_agent_imports():
    """Prove ordinary execution does not import Agent or orchestration packages."""
    import importlib.abc
    class Forbidden(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if any(fullname==p or fullname.startswith(p+'.') for p in ('agent_poc','langgraph','backend.app.agent','backend.app.knowledge','openai')):
                raise ImportError('Agent dependency forbidden in ordinary execution')
    sys.meta_path.insert(0,Forbidden())


def baseline_http(args):
    import httpx
    body=json.loads(args.request_file.read_text(encoding='utf-8'))
    token=os.environ['AUTOAI_API_TOKEN']
    base='http://127.0.0.1:'+str(args.port)
    with httpx.Client(base_url=base,headers={'Authorization':'Bearer '+token},trust_env=False,timeout=30) as client:
        response=client.post('/api/training/runs',json=body);response.raise_for_status()
        run=response.json()['run_id'];deadline=time.monotonic()+300
        while True:
            response=client.get('/api/training/runs/'+run);response.raise_for_status()
            state=response.json()['state']
            if state not in ('queued','running'):break
            if time.monotonic()>deadline:raise RuntimeError('baseline execution deadline')
            time.sleep(.2)
        if state!='succeeded':raise RuntimeError('baseline training failed')
    forbidden=[m for m in sys.modules if m.startswith(('agent_poc','langgraph','backend.app.agent','backend.app.knowledge','openai'))]
    assert forbidden==[]
    save(args.root,args.request_file.stem+'-result.json',dict(run_id=run,state=state,forbidden_imports=forbidden))


def recipe_acceptance(args,root,storage,base,token,env,datasets,report,audit):
    if not args.llm_url:raise ValueError('recipe acceptance requires a real LLM endpoint')
    import sqlite3
    from agent_poc.clients.autoai_client import AutoAIClient
    from agent_poc.orchestration.llm import LLMConfig,LLMAdapter
    from agent_poc.orchestration.runtime import RuntimeConfig,start_task,resume_task,TaskInterrupted,read_status
    from agent_poc.orchestration.graph import Nodes
    from backend.app.runs.repository import RunRepository
    from backend.app.runs.artifacts import RunArtifactWriter
    repo=RunRepository(storage/'runs.sqlite3')
    cfg=LLMConfig(args.llm_url,args.llm_model,timeout=180,max_tokens=1024,prompt_version='agent-decision-knowledge-v1' if args.knowledge_ablation else 'agent-decision-recipes-v1')
    runtime=RuntimeConfig(base,'step3-acceptance',cfg,backend_token=token,api_timeout=30)
    def client():return AutoAIClient(base,token=token,api_version='v2',execution_profile='train-evidence-recipes-v1',protocol_revision='agent-recipes-revision-v2' if args.knowledge_ablation else 'agent-recipes-revision-v1',max_retries=0,timeout=30)
    scenarios=[('traditional',['logistic_regression','svm','random_forest'],True),
               ('deep',['cnn1d','cnn1d_se'],True),('recovery',['logistic_regression','svm'],False)]
    if args.knowledge_ablation:
        scenarios=[('knowledge-on',['logistic_regression','svm'],True),('knowledge-off',['logistic_regression','svm'],True)]
    ablation_states=[]
    for scenario,allowed,expose in scenarios:
        calls=[];thread=('step4-' if args.knowledge_ablation else 'step3-')+scenario;checkpoint=root/'checkpoints'/thread
        class AuditedLLM(LLMAdapter):
            def propose(self,phase,context,**kwargs):
                proposal=super().propose(phase,context,**kwargs)
                calls.append(dict(phase=phase,context=context,tool=proposal.tool_name,
                    arguments=proposal.arguments,rationale=proposal.rationale,usage=proposal.usage.__dict__))
                save(root,thread+'-llm.json',calls)
                return proposal
        llm=AuditedLLM(cfg)
        fixed={m:{'epochs':2} for m in allowed if m.startswith('cnn')}
        kwargs=dict(dataset_id=datasets[3],allowed_models=allowed,model_configs=fixed,storage=checkpoint,
            thread_id=thread,wait=True,client=client(),llm=llm,timeout_seconds=600,max_api_calls=180,
            evidence_context=expose,risk_context=expose,**({'knowledge':scenario=='knowledge-on'} if args.knowledge_ablation else {}))
        if scenario=='recovery':
            original=Nodes.submit
            def interrupt(self,state):raise KeyboardInterrupt()
            Nodes.submit=interrupt
            try:
                try:start_task(runtime,**kwargs)
                except TaskInterrupted:pass
                else:raise AssertionError('expected prepared interruption')
            finally:Nodes.submit=original
            prepared=read_status(storage=checkpoint,thread_id=thread)
            assert prepared['recovery']['pending_operation']['status']=='prepared'
            frozen=prepared['execution']['submission_content']
            final=resume_task(runtime,storage=checkpoint,thread_id=thread,wait=True,client=client(),llm=llm)
            assert final['execution']['submission_content']==frozen
        else:final=start_task(runtime,**kwargs)
        save(root,thread+'-state.json',final)
        if final['lifecycle']['status']!='completed':
            report['llm'].append(dict(scenario=scenario,status=final['lifecycle']['status'],lifecycle=final['lifecycle'],versions=final['versions'],budget=final['budget'],calls=calls))
            save(root,'runtime_manifest.json',report)
            raise RuntimeError('real LLM acceptance incomplete')
        assert len(calls)==2 and calls[0]['tool']=='submit_ml_experiment' and calls[1]['tool']=='finalize_ml_session'
        assert ('train_statistics' in calls[0]['context'])==expose
        if args.knowledge_ablation:
            refs=final['execution']['submission_content']['knowledge_refs']
            assert bool(refs)==(scenario=='knowledge-on'),'real on must actually cite provided knowledge'
            assert final['knowledge']['snapshot']['projection']==calls[0]['context']['knowledge']
            ablation_states.append(final)
            if len(ablation_states)==2:
                on,off=ablation_states
                assert on['evidence']==off['evidence']
                assert on['recipes']['catalog']==off['recipes']['catalog']
                assert on['recipes']['evaluation_plan']==off['recipes']['evaluation_plan']
                report['knowledge_ablation']=dict(evidence_equal=True,catalog_equal=True,plan_equal=True,
                    fixed_model_configs_equal=on['task']['model_configs']==off['task']['model_configs'],
                    on_knowledge=on['knowledge'],off_knowledge=off['knowledge'])
        assert resume_task(runtime,storage=checkpoint,thread_id=thread,client=client(),llm=llm)==final
        run=repo.get(final['execution']['run_id']);catalog=final['recipes']['catalog']
        recipe=next(r for r in catalog['recipes'] if r['recipe_id']==final['execution']['submission_content']['recipe_id'])
        def agent_counts():
            with sqlite3.connect('file:'+str(storage/'agent.sqlite3')+'?mode=ro',uri=True) as db:
                return [db.execute('SELECT count(*) FROM '+t).fetchone()[0] for t in ('agent_sessions_v1','agent_experiment_reservations_v1')]
        before=agent_counts()
        baseline_config={**recipe['fixed_execution_config'],'evaluation_plan_digest':final['recipes']['evaluation_plan']['plan_digest']}
        request=root/(thread+'-baseline.json')
        save(root,request.name,dict(dataset_id=datasets[3],config=baseline_config))
        completed=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--role','baseline',
            '--root',str(root),'--port',str(args.port),'--request-file',str(request)],cwd=REPO,env=env,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=330)
        (root/(thread+'-baseline.log')).write_bytes(completed.stdout+completed.stderr)
        assert completed.returncode==0,'pure backend process failed'
        assert agent_counts()==before
        baseline=json.loads((root/(request.stem+'-result.json')).read_text())
        other=repo.get(baseline['run_id'])
        assert all(run.config[k]==other.config[k] for k in baseline_config)
        assert run.dataset_snapshot['sha256']==other.dataset_snapshot['sha256']
        def read_artifact(record,name):
            return json.loads(RunArtifactWriter(storage/'runs'/record.run_id).resolve_download(name).read_text())
        a,b=read_artifact(run,'model_metadata.json'),read_artifact(other,'model_metadata.json')
        assert a['execution_audit']==b['execution_audit']
        for field in ('model_profile','classification_head','loss_function','output_dim','execution_device'):
            assert a.get(field)==b.get(field),field
        result_comparison=None
        if recipe['model_id'] in ('logistic_regression','svm','random_forest'):
            import math
            import pandas as pd
            def equal(left,right):
                if isinstance(left,dict):
                    assert left.keys()==right.keys()
                    for key in left:equal(left[key],right[key])
                elif isinstance(left,list):
                    assert len(left)==len(right)
                    for first,second in zip(left,right):equal(first,second)
                elif type(left) in (int,float):
                    assert math.isclose(left,right,rel_tol=1e-8,abs_tol=1e-10)
                else:assert left==right
            equal(read_artifact(run,'metrics.json'),read_artifact(other,'metrics.json'))
            tables=[pd.read_csv(RunArtifactWriter(storage/'runs'/record.run_id).resolve_download('predictions.csv')) for record in (run,other)]
            pd.testing.assert_frame_equal(*tables,check_exact=False,rtol=1e-8,atol=1e-10)
            result_comparison=dict(metrics_equal=True,predictions_equal=True,rtol=1e-8,atol=1e-10)
        agent_audit,baseline_audit=audit(run.run_id),audit(other.run_id)
        assert agent_audit['manifest_complete'] and baseline_audit['manifest_complete']
        sa,sb=read_artifact(run,'split.json'),read_artifact(other,'split.json')
        partition_fields=('fold_index','train_sample_ids','valid_sample_ids','test_sample_ids','partition_digest','final_fit_indices','best_params','selection_metric','preprocess','splits','external_test_indices')
        assert [{k:f[k] for k in partition_fields} for f in sa]==[{k:f[k] for k in partition_fields} for f in sb]
        entry=dict(scenario=scenario,classes=3,session_id=final['identity']['session_id'],thread_id=thread,
            knowledge=final['knowledge'] if args.knowledge_ablation else None,run_id=run.run_id,baseline_run_id=other.run_id,
            chosen_model=recipe['model_id'],recipe_id=recipe['recipe_id'],plan=final['recipes']['evaluation_plan'],
            agent_session_count_unchanged=True,baseline=baseline,paired_config_equal=True,paired_split_equal=True,
            paired_execution_audit_equal=True,execution_audit=a['execution_audit'],
            audit=agent_audit,baseline_audit=baseline_audit,result_comparison=result_comparison,model_profile=a['model_profile'],classification_head=a.get('classification_head'),versions=final['versions'],budget=final['budget'],
            terminal_resume_unchanged=True,prepared_recovery=scenario=='recovery',llm_calls=len(calls))
        report['llm'].append(entry);save(root,'runtime_manifest.json',report)
        print(json.dumps(dict(scenario=scenario,model=recipe['model_id'],status='completed',paired=True)),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--role',choices=('driver','web','worker','baseline'),default='driver')
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--port',type=int,default=18761)
    parser.add_argument('--recipes',action='store_true')
    parser.add_argument('--knowledge-ablation',action='store_true')
    parser.add_argument('--backend-only',action='store_true')
    parser.add_argument('--request-file',type=Path)
    parser.add_argument('--llm-url')
    parser.add_argument('--llm-model',default='qwen3-4b')
    args=parser.parse_args()
    if args.knowledge_ablation and not args.recipes:parser.error('knowledge ablation requires recipes')
    if args.backend_only:
        if args.recipes or args.llm_url:parser.error('backend-only cannot start Agent/LLM acceptance')
        reject_agent_imports()
    return run(args) if args.role=='driver' else _role(args)


if __name__=='__main__':
    raise SystemExit(main())

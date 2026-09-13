"""Isolated real HTTP/worker acceptance for step 2; artifacts stay outside Git.

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
    if root==(REPO/'storage').resolve() or not root.name.startswith('autoai-step2-'):
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
    return dict(head=head,dirty_diff_sha256=hashlib.sha256(diff).hexdigest())


def _role(args):
    bind_storage(args.root)
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
    from agent_poc.clients.autoai_client import AutoAIClient
    from agent_poc.tools import ToolDispatcher
    from agent_poc.orchestration.llm import LLMConfig,LLMAdapter
    from agent_poc.orchestration.runtime import RuntimeConfig,start_task,resume_task

    os.environ['NO_PROXY']='127.0.0.1,localhost'
    os.environ['no_proxy']='127.0.0.1,localhost'
    token=secrets.token_urlsafe(40)
    env=dict(os.environ,AUTOAI_DEPLOYMENT_MODE='server',AUTOAI_API_TOKEN=token,
        AUTOAI_PRINCIPAL_ID='step2-acceptance',AUTOAI_TENANT_ID='step2',
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
    report['package_paths']={name:importlib.util.find_spec(name).origin for name in ('torch','numpy','scipy','sklearn','xgboost','pydantic','langgraph')}
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
        for classes in (2,3):
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
        if args.llm_url:
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
                    storage=checkpoint,thread_id=thread,wait=True,llm=adapter,timeout_seconds=600,max_api_calls=180)
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


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--role',choices=('driver','web','worker'),default='driver')
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--port',type=int,default=18761)
    parser.add_argument('--llm-url')
    parser.add_argument('--llm-model',default='qwen3-4b')
    args=parser.parse_args()
    return run(args) if args.role=='driver' else _role(args)


if __name__=='__main__':
    raise SystemExit(main())

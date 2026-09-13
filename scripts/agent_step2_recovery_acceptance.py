"""Real HTTP/worker/LLM recovery at two durable windows, in new isolated storage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.agent_step2_acceptance import bind_storage, code_binding, save


def run(root, llm_url):
    root=root.resolve()
    root.mkdir(parents=True,exist_ok=False)
    bind_storage(root)
    import httpx
    from backend.tests.modeling_data_factory import write_grouped_classification_csv
    from agent_poc.clients.autoai_client import AutoAIClient, _HttpxTransport
    from agent_poc.orchestration.llm import LLMAdapter, LLMConfig
    from agent_poc.orchestration.runtime import RuntimeConfig,start_task,resume_task,TaskInterrupted
    from agent_poc.orchestration.graph import Nodes,build_graph
    from backend.app.runs.artifacts import RunArtifactWriter
    port=18761
    with socket.socket() as s:s.bind(('127.0.0.1',port))
    os.environ['NO_PROXY']=os.environ['no_proxy']='127.0.0.1,localhost'
    token=secrets.token_urlsafe(40)
    env=dict(os.environ,AUTOAI_DEPLOYMENT_MODE='server',AUTOAI_API_TOKEN=token,
        AUTOAI_PRINCIPAL_ID='step2-recovery',AUTOAI_TENANT_ID='step2',
        AUTOAI_ALLOWED_ORIGINS='http://127.0.0.1:18761',PYTHONPATH=str(REPO),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    env.pop('STEP2_ISOLATED_STORAGE',None)
    report=dict(code=code_binding(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        node=socket.gethostname(),python=sys.executable,root=str(root),port=port,tasks=[],cleanup={})
    children=[];handles=[]
    try:
        for role in ('web','worker'):
            handle=(root/(role+'.log')).open('w');handles.append(handle)
            child=subprocess.Popen([sys.executable,'-B',str(REPO/'scripts/agent_step2_acceptance.py'),
                '--role',role,'--root',str(root),'--port',str(port)],env=env,cwd=REPO,stdout=handle,stderr=subprocess.STDOUT)
            children.append(child);report[role+'_pid']=child.pid
        base='http://127.0.0.1:18761'
        deadline=time.monotonic()+90
        while True:
            try:
                health=httpx.get(base+'/health',trust_env=False,timeout=3).json()
                if health['worker']['available'] and health['worker']['compatible']:break
            except (httpx.HTTPError,ValueError,KeyError):pass
            if time.monotonic()>deadline:raise RuntimeError('startup deadline')
            time.sleep(0.5)
        report['health']=health
        source=write_grouped_classification_csv(root/'synthetic.csv',groups_per_class=10,repeats=2,feature_count=128,labels=('A','B'))
        with source.open('rb') as handle:
            response=httpx.post(base+'/api/datasets/upload',headers={'Authorization':'Bearer '+token},
                files={'file':(source.name,handle,'text/csv')},trust_env=False,timeout=30)
        response.raise_for_status();dataset=response.json()['dataset_id']
        report['data']=dict(dataset_id=dataset,sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source='deterministic engineering synthetic grouped curves')
        config=LLMConfig(llm_url,'qwen3-4b',timeout=180,max_tokens=512,prompt_version='agent-decision-step2-v1')
        runtime=RuntimeConfig(base,'step2-recovery',config,backend_token=token,api_timeout=30)
        for model,window in [('logistic_regression','submit_prepared'),('cnn1d','choose_end')]:
            calls=[];proposals=[]
            class Wire(_HttpxTransport):
                def request(self,method,url,**kwargs):
                    calls.append(dict(method=method,path=urlsplit(url).path,timeout=kwargs['timeout']))
                    return super().request(method,url,**kwargs)
            class AuditedLLM(LLMAdapter):
                def propose(self,phase,context,**kwargs):
                    proposal=super().propose(phase,context,**kwargs)
                    proposals.append(dict(phase=phase,tool=proposal.tool_name,arguments=proposal.arguments,
                        usage=proposal.usage.__dict__,context_digest=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()))
                    return proposal
            def client():return AutoAIClient(base,token=token,transport=Wire(),api_version='v2',max_retries=0,timeout=30)
            def factory(deps,saver):
                graph=build_graph(deps,saver)
                class Proxy:
                    def __getattr__(self,name):return getattr(graph,name)
                    def invoke(self,*args,**kwargs):
                        state=graph.invoke(*args,**kwargs)
                        if window=='choose_end' and state['lifecycle']['next_action']=='submit':raise KeyboardInterrupt()
                        return state
                return Proxy()
            original=Nodes.submit
            def interrupt(*args):raise KeyboardInterrupt()
            if window=='submit_prepared':Nodes.submit=interrupt
            checkpoint=root/'checkpoints'/model
            try:
                start_task(runtime,dataset_id=dataset,allowed_models=[model],model_configs={model:{'epochs':2}} if model=='cnn1d' else {},
                    storage=checkpoint,thread_id=model,client=client(),llm=AuditedLLM(config),graph_factory=factory,wait=True)
                raise AssertionError('window was not interrupted')
            except TaskInterrupted:pass
            finally:Nodes.submit=original
            assert len(calls)==3
            state=resume_task(runtime,storage=checkpoint,thread_id=model,client=client(),llm=AuditedLLM(config),wait=True)
            assert state['lifecycle']['status']=='completed',state['lifecycle']
            assert state['budget']['api_calls']['actual']==len(calls)
            count=len(calls)
            assert resume_task(runtime,storage=checkpoint,thread_id=model,client=client(),llm=AuditedLLM(config))==state
            assert len(calls)==count
            run=state['execution']['run_id']
            writer=RunArtifactWriter(root/'storage/runs'/run)
            _,descriptors=writer.descriptors(run_id=run)
            assert not any(d['integrity'] in ('missing','corrupt') and d.get('required') for d in descriptors)
            entry=dict(model=model,window=window,run_id=run,session_id=state['identity']['session_id'],
                status=state['lifecycle']['status'],physical_http=len(calls),journal_api=state['budget']['api_calls']['actual'],
                requests=calls,proposals=proposals,versions=state['versions'],terminal_resume_unchanged=True,manifest_complete=True)
            report['tasks'].append(entry);save(root,model+'-state.json',state)
            save(root,'recovery_manifest.json',report)
            print(json.dumps({k:entry[k] for k in ('model','window','run_id','status','physical_http','journal_api')}),flush=True)
        report['completed']=True
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
        for handle in handles:handle.close()
        report['cleanup']={str(p.pid):p.poll() for p in children}
        save(root,'recovery_manifest.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--llm-url',default='http://127.0.0.1:18762/v1')
    args=parser.parse_args()
    run(args.root,args.llm_url)

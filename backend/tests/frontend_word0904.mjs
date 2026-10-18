import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {test} from 'node:test';

const html=readFileSync(new URL('../../static/index.html',import.meta.url),'utf8');
const policySource=readFileSync(new URL('../../static/js/classic-training-policy.js',import.meta.url),'utf8');
function harness(memory=new Map()) {
  const nodes=new Map();
  const get=id=>{if(!nodes.has(id))nodes.set(id,{value:'',checked:false,classList:{toggle(){},remove(){},add(){}},setAttribute(){}});return nodes.get(id);};
  const storage={getItem:key=>memory.get(key),setItem:(key,value)=>memory.set(key,value)};
  const context=vm.createContext({console,window:{sessionStorage:storage},$:get});
  vm.runInContext(policySource,context);
  const source=['updateSplitOptionVisibility','updateTrainingWorkHint','configPayload','selectedRamanBaselineMethod','startTrain'].map(name=>{
    const start=html.search(new RegExp(`      (?:async )?function ${name}\\(`)),end=html.indexOf('\n      }',start);assert.ok(start>=0);return html.slice(start,end+8);
  }).join('\n');
  vm.runInContext(`let testDatasetId=null,testDatasetPath=null,models=['svm'];
    function selectedModelTypes(){return models;}
    let trainingScheme={enabled:true,version:'word-0904',default_profile:'quick',default_feature_scheme:'full',
      profiles:[{id:'quick'},{id:'full'}],feature_schemes:[{id:'full'},{id:'bin_5'},{id:'bin_10'},{id:'bin_20'},{id:'pca_90'},{id:'pca_95'},{id:'pca_99'}],
      defaults:{epochs:200,batch_size:8,learning_rate:.001,weight_decay:.0001,scheduler_factor:.5,scheduler_patience:10,min_learning_rate:.000001,early_stopping_patience:20,seed:42}};
    ${source}`,context);
  get('featureScheme').value='full';
  return {run:s=>vm.runInContext(s,context),get,memory,context};
}

test('student requests ignore stale CV, ratios, profile, normalization and seed',()=>{
  const h=harness(new Map([['specautoai.split-preferences.v1',JSON.stringify({cv:true,ratios:{holdout:[6,2,2]}})]]));
  h.get('cvEnabled').checked=true;h.get('splitTrain').value='6';h.get('splitValid').value='0';
  h.get('trainingProfile').value='full';h.get('seed').value='987';h.get('normalization').value='minmax';
  const p=h.run('configPayload()');
  assert.equal(p.split_mode,'stratified_holdout');assert.equal(p.split_train,8);assert.equal(p.split_valid,1);assert.equal(p.split_test,1);
  assert.equal(p.experiment_version,'word-0904');assert.equal(p.epochs,200);assert.equal(p.seed,42);
  assert.equal(p.training_profile,'quick');assert.equal(p.feature_scheme,'full');assert.equal(p.normalization,'zscore');
  for(const key of ['dropout','svm_c','random_forest_search_iterations','hidden_size'])assert.equal(key in p,false);
});

test('external dataset uses 8:2 and no internal Test; removing it restores holdout',()=>{
  const h=harness();
  for(const external of ["testDatasetId='external'","testDatasetId=null;testDatasetPath='external.csv'"]){
    h.run(`${external};updateSplitOptionVisibility();updateSplitOptionVisibility()`);
    const p=h.run('configPayload()');
    assert.equal(p.split_mode,'external_test_holdout');assert.equal(p.split_train,8);assert.equal(p.split_valid,2);assert.equal('split_test' in p,false);
    assert.match(h.get('splitModeHint').textContent,/独立测试集只用于最终评估/);
  }
  h.run('testDatasetId=null;testDatasetPath=null;updateSplitOptionVisibility()');
  assert.equal(h.run('configPayload().split_test'),1);
});

test('three feature choices are capability-filtered and map to exact backend IDs',()=>{
  const h=harness();
  assert.equal(h.run('JSON.stringify(window.SpecAutoAITrainingPolicy.studentFeatureSchemes(trainingScheme).map(s=>s.id))'),JSON.stringify(['full','bin_10','pca_95']));
  for(const feature of ['full','bin_10','pca_95']){
    h.get('featureScheme').value=feature;assert.equal(h.run('configPayload().feature_scheme'),feature);
  }
  h.get('featureScheme').value='bin_5';assert.throws(()=>h.run('configPayload()'),/不可用/);
  h.run("trainingScheme.feature_schemes=trainingScheme.feature_schemes.filter(s=>s.id!=='pca_95')");
  h.get('featureScheme').value='pca_95';assert.throws(()=>h.run('configPayload()'),/不可用/);
});

test('PCA and CNN fail for mixed model selections without silently changing feature',()=>{
  const h=harness();h.get('featureScheme').value='pca_95';h.run("models=['svm','cnn1d'];updateTrainingWorkHint()");
  assert.equal(h.get('featureScheme').value,'pca_95');assert.match(h.get('trainingWorkHint').textContent,/不支持 PCA/);
  assert.throws(()=>h.run('configPayload()'),/CNN 不支持 PCA/);
  h.run("models=['svm'];updateTrainingWorkHint()");assert.equal(h.run('configPayload().feature_scheme'),'pca_95');
});

test('unavailable or stale backend capability fails closed',()=>{
  for(const change of ['trainingScheme.enabled=false','trainingScheme.version="old"','delete trainingScheme.profiles']){
    const h=harness();h.run(change);assert.throws(()=>h.run('configPayload()'),/尚未就绪|重启后端与 Worker/);
  }
});

test('single and batch HTTP submission use the same defaults and keep dataset IDs',async()=>{
  for(const models of [['svm'],['svm','random_forest']]){
    const h=harness();const calls=[];
    h.context.api=async(url,opts)=>{calls.push([url,JSON.parse(opts.body)]);return {batch_id:'b',run_count:2};};
    h.context.createRun=async(body)=>{calls.push(['run',body]);return {run_id:'r'};};
    h.context.resetTrainingPanel=()=>{};h.context.clearRememberedCurrentRun=()=>{};
    h.context.rememberCurrentRun=()=>{};h.context.watchCurrentRun=()=>{};h.context.watchBatchComparison=()=>{};
    h.context.currentRunNotice=()=>({});h.context.trainingContractIssue=()=>null;
    h.context.window.SpecAutoAIHealth={ready:true};h.context.window.SpecAutoAIResults={markNewRun(){}};
    h.get('trainingProgress').replaceChildren=()=>{};h.get('seed').value='999';h.get('featureScheme').value='bin_10';
    h.run(`let trainingRequestInFlight=false,datasetId='main',datasetPath=null,currentRunId=null;models=${JSON.stringify(models)};testDatasetId='external'`);
    await h.run('startTrain()');
    assert.equal(calls.length,1);const [url,body]=calls[0];
    assert.equal(body.dataset_id,'main');assert.equal(body.test_dataset_id,'external');assert.equal('data_path' in body,false);
    assert.equal(body.config.training_profile,'quick');assert.equal(body.config.feature_scheme,'bin_10');
    assert.equal(body.config.split_mode,'external_test_holdout');
    if(models.length>1){assert.equal(url,'/api/training/batches');assert.equal(body.base_seed,42);assert.deepEqual(body.model_types,models);}
    else {assert.equal(body.config.seed,42);assert.equal(body.config.model_type,'svm');}
  }
});

test('Raman always submits arPLS even with stale hidden custom fields',()=>{
  const h=harness();h.get('ramanBaselineMethodMode').value='custom';h.get('ramanCustomMethod').value='poly';
  assert.equal(h.run('selectedRamanBaselineMethod()'),'arPLS');
});

test('compatibility policy still supports four modes, custom ratios and full search',()=>{
  const h=harness();
  for(const [mode,strategy,ratios] of [['holdout','stratified_holdout',[6,2,2]],['external','external_test_holdout',[7,3,0]],['loo','leave_one_sample_id_cv',[9,1,0]],['external_loo','leave_one_sample_id_cv_with_external_test',[8,2,0]]]){
    const p=h.run(`window.SpecAutoAITrainingPolicy.payload(trainingScheme,'${mode}',${JSON.stringify(ratios)},'minmax','svm',{training_profile:'full',feature_scheme:'full'})`);
    assert.equal(p.split_mode,strategy);assert.equal(p.training_profile,'full');assert.equal(p.split_train,ratios[0]);
  }
  assert.throws(()=>h.run("window.SpecAutoAITrainingPolicy.splitPayload('holdout',[8,0,2])"),/正整数/);
});

test('configuration rows distinguish final selection from per-fold configurations',()=>{
  const source=readFileSync(new URL('../../static/js/experiment-details.js',import.meta.url),'utf8').replaceAll('export function ','function ');
  const context=vm.createContext({});vm.runInContext(source,context);
  assert.equal(context.configurationRows({selected_configuration:null,fold_configurations:[{fold_index:1},{fold_index:2}]}).length,2);
  assert.equal(context.configurationRows({selected_configuration:{scheme_id:'full'},fold_configurations:[{}]})[0].label,'最终配置');
  assert.match(context.configurationScore({selection_metric:'validation_loss',selection_score:0}),/0.000000/);
});

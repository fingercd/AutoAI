import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {test} from 'node:test';

const html=readFileSync(new URL('../../static/index.html',import.meta.url),'utf8');
const policySource=readFileSync(new URL('../../static/js/classic-training-policy.js',import.meta.url),'utf8');
function harness(memory=new Map()) {
  const nodes=new Map();
  const get=id=>{if(!nodes.has(id))nodes.set(id,{value:'',checked:false,classList:{toggle(){},remove(){}},setAttribute(){}});return nodes.get(id);};
  const storage={getItem:key=>memory.get(key),setItem:(key,value)=>memory.set(key,value)};
  const context=vm.createContext({console,window:{sessionStorage:storage},$:get});
  vm.runInContext(policySource,context);
  const source=['readSplitNumber','rememberSplitPreferences','applySplitPreset','updateSplitOptionVisibility','configPayload'].map(name=>{
    const start=html.indexOf(`      function ${name}(`),end=html.indexOf('\n      }',start);assert.ok(start>=0);return html.slice(start,end+8);
  }).join('\n');
  vm.runInContext(`let testDatasetId=null,testDatasetPath=null,displayedSplitMode=null;
    const splitPreferences=window.SpecAutoAITrainingPolicy.createPreferences(window.sessionStorage);
    $('cvEnabled').checked=splitPreferences.getCv();
    let trainingScheme={enabled:true,version:'word-0904',defaults:{epochs:200,batch_size:8,learning_rate:.001,weight_decay:.0001,scheduler_factor:.5,scheduler_patience:10,min_learning_rate:.000001,early_stopping_patience:20,seed:42}};
    ${source}`,context);
  get('normalization').value='zscore';get('modelType').value='svm';
  return {run:s=>vm.runInContext(s,context),get,memory};
}
test('all four modes support custom ratios and external data never clears LOO',()=>{
  const h=harness();h.run('updateSplitOptionVisibility()');
  h.get('splitTrain').value='6';h.get('splitValid').value='2';h.get('splitTest').value='2';
  assert.equal(h.run('configPayload().split_train'),6);
  h.run("testDatasetId='external';updateSplitOptionVisibility()");
  assert.equal(h.get('splitValid').value,'2');
  h.get('splitTrain').value='7';h.get('splitValid').value='3';
  h.run('updateSplitOptionVisibility()');assert.equal(h.run('configPayload().split_train'),7);
  assert.equal(h.run("Object.hasOwn(configPayload(),'split_test')"),false);
  h.get('cvEnabled').checked=true;h.run('updateSplitOptionVisibility()');
  assert.equal(h.get('cvEnabled').checked,true);
  assert.equal(h.run('configPayload().split_mode'),'leave_one_sample_id_cv_with_external_test');
  h.get('splitTrain').value='9';h.get('splitValid').value='1';h.run('rememberSplitPreferences()');
  h.run('testDatasetId=null;updateSplitOptionVisibility()');
  assert.equal(h.run('configPayload().split_mode'),'leave_one_sample_id_cv');
  h.get('cvEnabled').checked=false;h.run('updateSplitOptionVisibility()');
  assert.equal(h.run('configPayload().split_train'),6);
  h.run("testDatasetId='external';updateSplitOptionVisibility()");assert.equal(h.run('configPayload().split_train'),7);
});
test('refresh restores CV and per-mode ratios without resetting values on repeated renders',()=>{
  const h=harness();h.get('cvEnabled').checked=true;h.run('updateSplitOptionVisibility()');
  h.get('splitTrain').value='7';h.get('splitValid').value='3';h.run('rememberSplitPreferences()');
  const fresh=harness(h.memory);fresh.run('updateSplitOptionVisibility();updateSplitOptionVisibility()');
  assert.equal(fresh.get('cvEnabled').checked,true);assert.equal(fresh.run('configPayload().split_train'),7);
});
test('versioned payload is fixed, invalid ratios and unavailable capability fail closed',()=>{
  const h=harness();h.run('updateSplitOptionVisibility()');const p=h.run('configPayload()');
  assert.equal(p.experiment_version,'word-0904');assert.equal(p.epochs,200);assert.equal(p.seed,42);
  for(const key of ['dropout','svm_c','random_forest_search_iterations','hidden_size'])assert.equal(key in p,false);
  h.get('splitValid').value='0';assert.throws(()=>h.run('configPayload()'),/正整数/);
  h.get('splitValid').value='1';h.run('trainingScheme=null');assert.throws(()=>h.run('configPayload()'),/尚未就绪/);
});
test('configuration rows distinguish final selection from per-fold configurations',()=>{
  const source=readFileSync(new URL('../../static/js/experiment-details.js',import.meta.url),'utf8').replaceAll('export function ','function ');
  const context=vm.createContext({});vm.runInContext(source,context);
  assert.equal(context.configurationRows({selected_configuration:null,fold_configurations:[{fold_index:1},{fold_index:2}]}).length,2);
  assert.equal(context.configurationRows({selected_configuration:{scheme_id:'full'},fold_configurations:[{}]})[0].label,'最终配置');
  assert.match(context.configurationScore({selection_metric:'validation_loss',selection_score:0}),/0.000000/);
});

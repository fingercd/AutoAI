import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {test} from 'node:test';

const source=readFileSync(new URL('../../static/js/comparison-page.js',import.meta.url),'utf8').replace(/^import .*;$/gm,'').replaceAll('export function ','function ');
const context=vm.createContext({window:{dispatchEvent(){}},Event,Map,Set,console,naturalCompare:(a,b)=>String(a).localeCompare(String(b),'zh-CN',{numeric:true,sensitivity:'base'})});
vm.runInContext(source,context);

test('model ordering does not mutate input and keeps missing values last',()=>{
  const data={models:[{model_type:'b',metrics:{balanced_accuracy:{mean:null}}},{model_type:'c',metrics:{balanced_accuracy:{mean:.8}}},{model_type:'a',metrics:{balanced_accuracy:{mean:.8}}}]};
  assert.equal(context.sortedModels(data).map(row=>row.model_type).join(','),'a,c,b');
  assert.equal(data.models.map(row=>row.model_type).join(','),'b,c,a');
});
test('error filter does not misclassify null and combines with ID search',()=>{
  const data={sample_correctness:{sample_ids:['1','12','14','20'],values:[{values:[null,0,1,0]}]}};
  assert.equal(context.filteredSamples(data,'',true).map(item=>item.id).join(','),'12,20');
  assert.equal(context.filteredSamples(data,'1',true).map(item=>item.id).join(','),'12');
});

test('sample details sort 1 2 3 10 naturally while keeping source value indices',()=>{
  const data={sample_correctness:{sample_ids:['1','10','2','3','S10','S2','S1'],values:[{values:[1,0,0,1,0,1,1]}]}};
  const rows=context.filteredSamples(data);
  assert.equal(rows.map(r=>r.id).join(','),'1,2,3,10,S1,S2,S10');
  assert.equal(rows.map(r=>data.sample_correctness.values[0].values[r.index]).join(','),'1,0,1,0,1,1,0');
  assert.equal(context.filteredSamples(data,'',true).map(r=>r.id).join(','),'2,10,S10');
});

test('feature details are scoped and never fall back to merged model averages',()=>{
  const skipped={feature_scheme:'pca_components_5',comparable:false,models:[]};
  const bins={feature_scheme:'bin_3',comparable:true,models:[{model_type:'pls_da',run_ids:['bin-run']}]};
  const pca={feature_scheme:'pca_components_10',comparable:true,models:[{model_type:'pls_da',run_ids:['pca-run']}]};
  const data={models:[...bins.models,...pca.models],scheme_comparisons:[skipped,bins,pca]};
  assert.equal(context.comparisonScope(data,'').feature_scheme,'bin_3');
  assert.equal(context.comparisonScope(data,'pca_components_10').models[0].run_ids[0],'pca-run');
  assert.equal(context.comparisonScope(data,'pca_components_5').models.length,0);
  const legacy={models:[{model_type:'svm'}]};assert.equal(context.comparisonScope(legacy,''),legacy);
});

test('matrix enlargement stays bounded by both screen dimensions',()=>{
  for(const [width,height] of [[2549,1401],[1920,1080],[1440,900],[540,900],[900,600]]){
    const size=context.matrixZoomWidth(width,height);
    assert.ok(size<=640);
    assert.ok(size+56<=width-24);
    assert.ok(size+100<=height*.85);
  }
});

test('per-class tables keep recall and precision apart',()=>{
  const data={class_metrics:{status:'ready',labels:['1','2'],rows:[
    {model_type:'pls_da',values:[{precision:.9,recall:.8,f1:.85,support:4},{precision:.7,recall:.6,f1:.65,support:4}]},
    {model_type:'svm',values:[{precision:null,recall:.5,f1:null,support:null},{precision:.5,recall:null,f1:.5,support:2}]},
  ]}};
  const models=[{model_type:'pls_da'},{model_type:'svm'}];
  const text=rows=>JSON.stringify(rows);
  assert.equal(text(context.classMetricRows(data,models,'recall')),'[["pls_da","80.00%","60.00%"],["svm","50.00%","—"]]');
  assert.equal(text(context.classMetricRows(data,models,'precision')),'[["pls_da","90.00%","70.00%"],["svm","—","50.00%"]]');
  assert.equal(text(context.classSupportRows(data,models)),'[["pls_da","4","4"],["svm","—","2"]]');
  // 同一个数字不会同时被当成两个指标：recall 与 precision 必须来自不同字段。
  assert.notEqual(context.classMetricRows(data,models,'recall')[0][1],context.classMetricRows(data,models,'precision')[0][1]);
});

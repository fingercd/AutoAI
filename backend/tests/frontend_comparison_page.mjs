import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {test} from 'node:test';

const source=readFileSync(new URL('../../static/js/comparison-page.js',import.meta.url),'utf8').replace(/^import .*;$/gm,'').replaceAll('export function ','function ');
const context=vm.createContext({window:{dispatchEvent(){}},Event,Map,Set,console});
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

test('matrix enlargement stays bounded by both screen dimensions',()=>{
  for(const [width,height] of [[2549,1401],[1920,1080],[1440,900],[540,900],[900,600]]){
    const size=context.matrixZoomWidth(width,height);
    assert.ok(size<=640);
    assert.ok(size+56<=width-24);
    assert.ok(size+100<=height*.85);
  }
});

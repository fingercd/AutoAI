import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { test } from 'node:test';

const html = readFileSync(new URL('../../static/index.html', import.meta.url), 'utf8');
const functions = ['stopBatchWatch', 'rememberCurrentBatch', 'rememberCurrentRun',
  'clearRememberedCurrentRun', 'renderBatchSnapshot', 'renderBatchPollingError',
  'watchBatchComparison', 'restoreCurrentBatch', 'restoreCurrentRun', 'restoreCurrentTraining',
  'batchRunProgressText', 'requestTrainingSnapshot', 'renderRun', 'watchCurrentRun',
  'resetStoppedTrainingPanel', 'resetTrainingPanel', 'clearCurrentRunDetails', 'currentRunNotice'];
const source = functions.map(name => {
  const start = html.search(new RegExp(`      (?:async )?function ${name}\\(`));
  assert.ok(start >= 0, name);
  const end = html.indexOf('\n      }', start);
  return html.slice(start, end + '\n      }'.length);
}).join('\n');

function harness() {
  const nodes = new Map();
  class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.dataset = {}; this.style = {}; this.textContent = ''; this.classList = { add(){}, toggle(){} }; }
    set id(value) { this._id = value; nodes.set(value, this); }
    get id() { return this._id; }
    get childElementCount() { return this.children.length; }
    append(...items) { for (const item of items) { item.parent = this; this.children.push(item); } }
    replaceChildren(...items) { this.children = []; this.append(...items); }
    remove() { this.parent.children = this.parent.children.filter(item => item !== this); }
    addEventListener() {}
    setAttribute() {}
    querySelector(selector) {
      const key = selector.startsWith('[data-') ? selector.slice(6,-1).replace(/-([a-z])/g, (_, c) => c.toUpperCase()) : null;
      for (const child of this.children) {
        if (key ? Object.hasOwn(child.dataset, key) : child.tag === selector) return child;
        const found = child.querySelector(selector); if (found) return found;
      }
      return null;
    }
  }
  for (const id of ['runStatus','trainingProgress','stopTrain','batchComparison','batchComparisonPage','batchComparisonError','metrics','trainingAudit','trainingCharts','confusionMatrix','downloads']) {
    const node = new Element('div'); node.id = id;
  }
  const timers = new Map(); let nextTimer = 0;
  const memory = new Map(); const calls = [];
  const context = vm.createContext({ AbortController, console, Set, Map,
    document: { createElement: tag => new Element(tag) },
    $: id => nodes.get(id),
    window: {
      sessionStorage: { getItem: k => memory.get(k), setItem: (k,v) => memory.set(k,v), removeItem: k => memory.delete(k) },
      setTimeout(fn, delay) { const id = ++nextTimer; timers.set(id, {fn, delay}); return id; },
      clearTimeout: id => timers.delete(id), location: { hash: '#/modeling' },
      SpecAutoAIResults: { parseHash: hash => ({view:hash.includes('comparison')?'comparison':'modeling'}), stopTrainingWatch(){calls.push('stop-single');}, cancelAutoRedirect(){} },
    },
    api: async url => { calls.push(url); return {state:'running', counts:{running:1}, runs:[]}; },
    fetchRun: async id => ({run_id:id, state:'running', config:{batch_id:'batch'}}),
    renderBatchComparison: () => calls.push('render-comparison'),
    loadRuns: async () => {},
    isTraditionalModel: () => true, runTotalTargetEpochs: () => 200,
    runFoldProgressText: () => '1/1', isExternalTestHoldout: () => false,
    formatTrainingTime: () => '-', renderCurrentRunMetrics() {},
  });
  vm.runInContext(`let currentRunId=null, currentBatchId=null, trainingRequestInFlight=false;
    const CURRENT_BATCH_STORAGE_KEY='batch', CURRENT_RUN_STORAGE_KEY='run';
    let batchPollTimer=null, batchPollController=null, batchComparisonTimer=null, batchComparisonController=null;
    let batchPollGeneration=0, batchPollRetryCount=0, batchLastComparisonSignature='';` + source, context);
  return { context, nodes, memory, timers, calls, run: code => vm.runInContext(code, context) };
}
const flush = () => new Promise(resolve => setImmediate(resolve));
const fixture = () => ({state:'running', counts:{queued:2,running:1,succeeded:1}, runs:[
  {run_id:'pls',model_type:'pls_da',state:'queued',progress:{}},
  {run_id:'lr',model_type:'logistic_regression',state:'running',progress:{feature_scheme:'Binning 10',current_fold:1,search_completed:9,search_total:15}},
  {run_id:'svm',model_type:'svm',state:'queued',progress:{}},
  {run_id:'rf',model_type:'random_forest',state:'succeeded',progress:{}},
]});

test('batch renders every model, updates rows in place, and retains them on network error', () => {
  const h = harness(); h.run("currentBatchId='batch'");
  const snapshot = fixture(); h.context.renderBatchSnapshot('batch', snapshot);
  const rows = h.nodes.get('batchTrainingRows').children;
  assert.equal(rows.length, 4);
  assert.match(rows[1].children[2].textContent, /Binning 10.*9\/15/);
  snapshot.runs[0].state='running'; snapshot.runs[0].progress={current_epoch:3};
  h.context.renderBatchSnapshot('batch', snapshot);
  assert.equal(h.nodes.get('batchTrainingRows').children[0], rows[0]);
  assert.equal(rows[0].children[1].textContent, '训练中');
  assert.match(rows[0].children[2].textContent, /Epoch 3/);
  h.context.renderBatchPollingError('batch', Error('offline'));
  assert.equal(h.nodes.get('batchTrainingRows').children.length, 4);
  assert.match(h.nodes.get('batchTrainingWarning').textContent, /offline/);
  h.context.renderBatchSnapshot('batch',snapshot);
  assert.equal(h.nodes.get('batchTrainingWarning').textContent,'');
});

test('refresh prioritizes saved batch even when a single run was also remembered', async () => {
  const h = harness(); h.memory.set('batch','batch'); h.memory.set('run','lr');
  h.context.fetchRun=async () => { throw Error('single recovery must not execute'); };
  await h.context.restoreCurrentTraining(); await flush();
  assert.equal(h.run('currentBatchId'),'batch');
  assert.equal(h.run('currentRunId'),null);
  assert.equal(h.memory.has('run'),false);
  assert.ok(h.calls.includes('/api/training/batches/batch'));
});

test('recovered active child run restores its entire batch', async () => {
  const h=harness();
  h.context.api=async url => url.includes('projection=summary') ? [{run_id:'lr',state:'running'}] : fixture();
  await h.context.restoreCurrentTraining(); await flush();
  assert.equal(h.run('currentBatchId'),'batch');
  assert.equal(h.run('currentRunId'),null);
  assert.equal(h.nodes.get('batchTrainingRows').children.length,4);
});

test('late restore cannot overwrite a new batch or undo STOP', async () => {
  for (const action of ["watchBatchComparison('new')", 'resetStoppedTrainingPanel()']) {
    const h=harness(); let resolve;
    h.context.api=url => url.includes('projection=summary') ? new Promise(r=>{resolve=r;}) : Promise.resolve(fixture());
    const restoring=h.context.restoreCurrentRun();
    h.run(action); resolve([{run_id:'old',state:'running'}]);
    await restoring; await flush();
    assert.equal(h.run('currentRunId'),null);
    assert.equal(h.run('currentBatchId'),action.startsWith('watch')?'new':null);
  }
});

test('slow result comparison cannot block status polling; late responses ignored after STOP', async () => {
  const h=harness(); h.context.window.location.hash='#/comparison';
  let resolveComparison;
  h.context.api=async url => url.endsWith('/comparison') ? new Promise(r=>{resolveComparison=r;}) : fixture();
  h.context.watchBatchComparison('batch'); await flush();
  assert.equal(h.nodes.get('batchTrainingRows').children.length,4);
  assert.ok([...h.timers.values()].some(timer=>timer.delay===3000));
  h.context.resetStoppedTrainingPanel();
  resolveComparison({state:'succeeded'}); await flush();
  assert.equal(h.nodes.get('runStatus').textContent,'尚未开始');
  assert.equal(h.nodes.get('trainingProgress').children.length,0);
  assert.equal(h.calls.includes('render-comparison'),false);
  assert.equal(h.timers.size,0);
});

test('modeling status never requests comparison artifacts; returning to modeling restarts polling', async () => {
  const h=harness(); h.context.watchBatchComparison('batch'); await flush();
  assert.ok(h.calls.includes('/api/training/batches/batch'));
  assert.ok(!h.calls.some(call=>call.endsWith('/comparison')));
  h.context.stopBatchWatch(); assert.equal(h.timers.size,0);
  h.context.watchBatchComparison(h.run('currentBatchId')); await flush();
  assert.equal(h.nodes.get('batchTrainingRows').children.length,0);
  assert.ok([...h.timers.values()].some(timer=>timer.delay===3000));
});

test('timeout produces recoverable error and releases timer', async () => {
  const h=harness();
  h.context.api=(_url,{signal})=>new Promise((_resolve,reject)=>signal.addEventListener('abort',()=>reject(Object.assign(Error('aborted'),{name:'AbortError'}))));
  const pending=h.context.requestTrainingSnapshot('/status',new AbortController().signal);
  const check=assert.rejects(pending,/15 秒/);
  [...h.timers.values()][0].fn(); await check;
  assert.equal(h.timers.size,0);
});

test('single-run queued title changes to running and cannot overwrite batch', () => {
  const h=harness(); h.run("currentRunId='single'");
  h.context.renderRun({run_id:'single',state:'queued',model_type:'svm'});
  h.context.renderRun({run_id:'single',state:'running',model_type:'svm'});
  assert.equal(h.nodes.get('trainingProgress').querySelector('strong').textContent,'模型正在训练');
  h.run("currentBatchId='batch'");
  h.context.renderBatchSnapshot('batch',fixture());
  h.context.renderRun({run_id:'single',state:'running',model_type:'svm'});
  assert.equal(h.nodes.get('batchTrainingRows').children.length,4);
});

test('legacy training explicitly marks missing fine-grained progress', () => {
  const h=harness();
  const text=h.context.batchRunProgressText({state:'running',model_type:'svm',progress:{current_fold:1}});
  assert.match(text,/未提供细分进度.*当前折 1/);
  assert.ok(!text.includes('%'));
});

test('STOP persistence suppresses both recovery paths on refresh', async () => {
  const h=harness(); h.memory.set('specautoai.training.idleAfterStop','1');
  h.memory.set('batch','old'); h.memory.set('run','old');
  h.context.api=async()=>{throw Error('STOP must not query old tasks');};
  await h.context.restoreCurrentTraining();
  assert.equal(h.run('currentBatchId'),null); assert.equal(h.run('currentRunId'),null);
});

test('status request failure retains rows, schedules retry, and resumes updates', async () => {
  const h=harness(); let failing=false;
  h.context.api=async()=>{if(failing)throw Error('offline');return fixture();};
  h.context.watchBatchComparison('batch'); await flush();
  failing=true;
  let entry=[...h.timers].find(([,timer])=>timer.delay===3000);
  h.timers.delete(entry[0]); await entry[1].fn();
  assert.equal(h.nodes.get('batchTrainingRows').children.length,4);
  assert.match(h.nodes.get('batchTrainingWarning').textContent,/offline/);
  failing=false;
  entry=[...h.timers].find(([,timer])=>timer.delay===3000);
  h.timers.delete(entry[0]); await entry[1].fn();
  assert.equal(h.nodes.get('batchTrainingWarning').textContent,'');
});

/** Classic comparison only. No training/STOP/store mutation and no v2 imports. */
import { request, downloadFile } from './api-client.js';

const metrics = [['accuracy','Accuracy'],['balanced_accuracy','Balanced Accuracy'],['macro_f1','Macro-F1'],['weighted_f1','Weighted-F1']];
const names = {pls_da:'PLS-DA',logistic_regression:'Elastic Net',svm:'SVM',random_forest:'Random Forest',xgboost:'XGBoost',cnn1d:'1D-CNN'};
// 暂停特征工程展示，保留下面的实现以便恢复；与后端 feature_policy 对应。
const FEATURE_ENGINEERING_ENABLED = false;
const modelName = id => names[id] || id;
const element = (tag, text='', cls='') => { const node=document.createElement(tag);node.textContent=text;if(cls)node.className=cls;return node; };
const button = (text, action) => {const node=element('button',text,'secondary');node.type='button';node.addEventListener('click',action);return node;};
const select = (items, value, action) => {const node=element('select');items.forEach(([key,label])=>{const option=element('option',label);option.value=key;node.append(option);});node.value=value;node.addEventListener('change',()=>action(node.value));return node;};
const labeled = (text, control) => {const node=element('label',text);node.append(control);return node;};
const details = (title, content) => {const node=element('details');node.append(element('summary',title),content);return node;};
const table = (headers, rows) => {const wrap=element('div','','cmp-scroll'),node=element('table');const head=element('thead'),hr=element('tr');headers.forEach(text=>hr.append(element('th',text)));head.append(hr);const body=element('tbody');rows.forEach(row=>{const tr=element('tr');row.forEach(value=>tr.append(element('td',value==null?'—':String(value))));body.append(tr);});node.append(head,body);wrap.append(node);return wrap;};

export function sortedModels(data, sort='balanced_accuracy') {
  const value = row => typeof row.metrics?.[sort]?.mean==='number' ? row.metrics[sort].mean : -Infinity;
  return [...(data.models||[])].sort((a,b)=>value(b)-value(a)||(a.model_type<b.model_type?-1:a.model_type>b.model_type?1:0));
}
export function filteredSamples(data, search='', errors=false) {
  const sample=data.sample_correctness||{};
  return (sample.sample_ids||[]).map((id,index)=>({id,index})).filter(({id,index})=>String(id).toLowerCase().includes(search.toLowerCase())&&(!errors||(sample.values||[]).some(row=>row.values?.[index]===0)));
}

let state=null, root=null, observer=null, controller=null, renderToken=0, resizeTimer=null;
let urls=[];
const imageCache=new Map();
function release() {controller?.abort();urls.forEach(url=>URL.revokeObjectURL(url));urls=[];renderToken+=1;}
function blobUrl(blob) {const url=URL.createObjectURL(blob);urls.push(url);return url;}
function saveBlob(blob,name) {const url=URL.createObjectURL(blob),link=element('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function filename(options, extension) {return [options.kind, options.metric, options.model, options.kind==='samples'?`page-${options.page+1}`:'', options.kind==='matrix'?options.matrix_mode:''].filter(Boolean).join('_')+'.'+extension;}
function section(title, controls=[]) {const node=element('section','','cmp-section'),head=element('div','','cmp-toolbar');head.append(element('h3',title));if(controls.length){const actions=element('div','','cmp-actions');actions.append(...controls);head.append(actions);}node.append(head);return node;}

export function matrixZoomWidth(viewportWidth, viewportHeight) {
  // The square plot, controls and dialog padding should fit on one screen.
  return Math.max(260, Math.floor(Math.min(640, viewportWidth - 88, viewportHeight * .85 - 110)));
}

function chart(title, options, tasks, {sample=false, zoom=true}={}) {
  const card=element('figure','','cmp-figure'+(['recall','samples','features'].includes(options.kind)?' cmp-compact':'')),head=element('div','','cmp-figure-head'),actions=element('div','','cmp-actions');
  const host=element('div','','cmp-scroll'),message=element('p','正在绘图…','cmp-note');host.append(message);
  let rendered=null, requested=null;
  async function download(format) {
    if(!rendered)return;
    try {
      if(format==='svg')saveBlob(new Blob([rendered.svg],{type:'image/svg+xml'}),filename(requested,'svg'));
      else {const result=await downloadFile(`/api/training/batches/${encodeURIComponent(state.id)}/figure?format=png`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(requested)});saveBlob(result.blob,filename(requested,'png'));}
    } catch(error) {message.textContent=error.message;message.className='cmp-error';host.append(message);}
  }
  const svg=button('SVG',()=>download('svg')),png=button('PNG',()=>download('png'));svg.disabled=png.disabled=true;
  actions.append(svg,png);
  if(zoom)actions.append(button('放大',()=>{
    const matrix = options.kind === 'matrix';
    const dialog=element('dialog','','cmp'+(matrix?' cmp-matrix-dialog':''));
    const matrixWidth=matrixZoomWidth(window.innerWidth,window.innerHeight);
    if(matrix)dialog.style.width=`${matrixWidth+56}px`;
    const close=button('关闭',()=>{dialog.close();dialog.remove();});dialog.append(close);document.getElementById('view-comparison').append(dialog);dialog.showModal();
    const width=matrix?matrixWidth:Math.min(1400,Math.floor(dialog.clientWidth-60));
    const queue=[];dialog.append(chart(title,{...options,width},queue,{zoom:false}));queue.forEach(task=>task());
    dialog.addEventListener('close',()=>dialog.remove(),{once:true});
  }));
  head.append(element('h4',title),actions);card.append(head,host);
  tasks.push(async()=>{
    const token=renderToken;
    requested={sort:state.sort,width:Math.min(1800,Math.max(260,Math.floor(card.clientWidth-22))),...options};
    try {
      const cacheKey=JSON.stringify({batch:state.id,...requested});
      const result=imageCache.get(cacheKey)||await request(`/api/training/batches/${encodeURIComponent(state.id)}/figure`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(requested),signal:controller.signal});
      if(token!==renderToken||!card.isConnected)return;
      imageCache.set(cacheKey,result);if(imageCache.size>96)imageCache.delete(imageCache.keys().next().value);
      rendered=result;const url=blobUrl(new Blob([result.svg],{type:'image/svg+xml'}));
      const makeImage=()=>{const img=element('img');img.src=url;img.alt=title;img.width=result.width;img.height=result.height;return img;};
      if(sample){const wrap=element('div','','cmp-sample-wrap'),labels=element('div','','cmp-sample-labels'),body=element('div','','cmp-sample-body');labels.setAttribute('aria-hidden','true');labels.append(makeImage());body.append(makeImage());wrap.append(labels,body);host.replaceChildren(wrap);}
      else host.replaceChildren(makeImage());
      svg.disabled=png.disabled=false;
    }catch(error){if(token===renderToken&&error.name!=='AbortError'){message.textContent=error.message;message.className='cmp-error';}}
  });
  return card;
}

async function historyControls(host) {
  const selected=select([['','选择历史对比']], '', id=>{if(id)window.location.hash=`#/comparison?batch_id=${encodeURIComponent(id)}`;});
  selected.setAttribute('aria-label','选择历史对比');
  const more=button('更多',()=>load()),note=element('span','','cmp-note');let cursor=null;
  async function load(){more.disabled=true;try{const data=await request(`/api/training/batches?limit=20${cursor?'&cursor='+encodeURIComponent(cursor):''}`);if(!host.isConnected)return;data.items.forEach(item=>{const time=new Date(item.created_at).toLocaleString('zh-CN',{hour12:false});const label={succeeded:'已完成',partial:'部分完成',failed:'失败',cancelled:'已停止',running:'训练中',queued:'排队中'}[item.state]||item.state;const option=element('option',`${time} · ${item.dataset_name||'数据集'} · ${item.model_types.length} 模型 · ${label}`);option.value=item.batch_id;selected.append(option);});cursor=data.next_cursor;more.hidden=!cursor;note.textContent='';}catch(error){note.textContent=error.message;}finally{more.disabled=false;}}
  host.append(selected,more,note);await load();
}

async function ensureSaved(force=false) {
  if(!state||['queued','running','cancelled'].includes(state.data.state))return;
  const current=state;
  if(current.saving)return;
  current.saving=true;current.saveText='正在保存对比与图像…';updateSaveText();
  try {
    let result=await request(`/api/training/batches/${encodeURIComponent(current.id)}/archive`);
    if(force||result.state!=='ready'){await request(`/api/training/batches/${encodeURIComponent(current.id)}/archive${force?'?force=true':''}`,{method:'POST'});result=await request(`/api/training/batches/${encodeURIComponent(current.id)}/archive`);}
    if(state!==current)return;
    current.saved=result.state==='ready';current.saveText=current.saved?'对比与图像已保存':result.message;
    if(current.saved&&result.comparison){current.archiveData=result.comparison;paint();}
  }catch(error){if(state===current){current.saveText=`保存未完成：${error.message}`;}}
  finally{current.saving=false;if(state===current)updateSaveText();}
}
function updateSaveText(){if(!root||!state)return;const note=root.querySelector('[data-save-note]');if(note)note.textContent=state.saveText||'训练结束后自动保存';root.querySelectorAll('[data-download-all]').forEach(node=>node.disabled=!state.saved);root.querySelectorAll('[data-retry-save]').forEach(node=>{node.hidden=state.saved;node.disabled=state.saving;});}

function paint() {
  if(!root||!state)return;
  const openDetails=new Set([...root.querySelectorAll('details[open]')].map(node=>node.querySelector('summary')?.textContent));
  release();controller=new AbortController();
  const data=state.archiveData||state.data,models=sortedModels(data,state.sort),tasks=[];
  const top=section('对比结果');
  const meta=data.archive||{},evaluation=data.evaluation||{};
  const agg=evaluation.primary_aggregation==='pooled_oof'?'合并留一法预测（OOF）':evaluation.primary_aggregation==='direct_external_test'?'独立测试集':'Test';
  top.append(element('p',`${meta.dataset_name||state.datasetName||'当前数据集'} · ${agg} · ${models.length} 个可比较模型`,'cmp-note'));
  const controls=element('div','','cmp-toolbar'),actions=element('div','','cmp-actions');
  actions.append(labeled('排序',select(metrics,state.sort,value=>{state.sort=value;paint();})));
  const all=button('下载全部',async()=>{try{all.disabled=true;const file=await downloadFile(`/api/training/batches/${encodeURIComponent(state.id)}/archive/files/all.zip`);saveBlob(file.blob,`comparison-${state.id.slice(0,8)}.zip`);}catch(error){state.saveText=error.message;state.saved=false;}finally{updateSaveText();}});all.dataset.downloadAll='';all.disabled=!state.saved;
  const retry=button('重试保存',()=>ensureSaved(true));retry.hidden=state.saved;retry.dataset.retrySave='';
  const note=element('span',state.saveText||'训练结束后自动保存','cmp-note');note.dataset.saveNote='';
  actions.append(all,retry,note);const history=element('div','','cmp-actions');controls.append(actions,history);top.append(controls);
  root.replaceChildren(top);historyControls(history);
  if(!data.comparable){root.append(element('p',data.reason||'暂无完整可比较结果','cmp-note'));return;}
  const overall=section('总体性能'),grid=element('div','','cmp-grid');
  metrics.forEach(([metric,title])=>grid.append(chart(title,{kind:'overall',metric},tasks)));overall.append(grid);
  const percent=value=>typeof value==='number'?`${(value*100).toFixed(2)}%`:'—';
  overall.append(details('精确数值',table(['模型',...metrics.map(([,label])=>label)],models.map(row=>[modelName(row.model_type),...metrics.map(([key])=>percent(row.metrics?.[key]?.mean))]))));root.append(overall);
  const matrices=section('混淆矩阵',[labeled('显示',select([['percent','真实类别归一化'],['count','原始计数']],state.matrixMode,value=>{state.matrixMode=value;paint();}))]);
  matrices.append(element('p','纵轴：真实类别；横轴：预测类别。百分比固定 0–100%，计数视图共享色标。','cmp-note'));
  const matrixGrid=element('div','','cmp-grid'+((data.confusion_matrices||[]).some(row=>row.labels.length>6)?' cmp-large-classes':''));
  models.forEach(model=>{const row=(data.confusion_matrices||[]).find(item=>item.model_type===model.model_type);if(row?.confusion_matrix)matrixGrid.append(chart(modelName(model.model_type),{kind:'matrix',model:model.model_type,matrix_mode:state.matrixMode},tasks));});matrices.append(matrixGrid);root.append(matrices);
  if(data.class_recall?.status==='ready'){
    const recall=section('各类别 Recall / Sensitivity');recall.append(chart('类别召回率',{kind:'recall'},tasks));
    recall.append(details('各类别记录数',table(['模型',...data.class_recall.labels],models.map(row=>{const entry=data.class_recall.rows.find(item=>item.model_type===row.model_type);return [modelName(row.model_type),...(entry?.values||[]).map(item=>item?.support)];}))));root.append(recall);
  }
  if(data.sample_correctness?.status==='ready'){
    const search=element('input');search.type='search';search.placeholder='搜索 Sample_ID';search.value=state.search;search.setAttribute('aria-label','搜索 Sample_ID');
    const check=element('input');check.type='checkbox';check.checked=state.errors;check.addEventListener('change',()=>{state.errors=check.checked;state.page=0;paint();});
    const apply=()=>{state.search=search.value;state.page=0;paint();};search.addEventListener('keydown',event=>{if(event.key==='Enter')apply();});
    const sample=section('Sample_ID × 模型预测正误',[search,button('搜索',apply),labeled('只看错误',check)]);
    sample.append(element('p','✓ 正确　× 错误　— 缺失；每列是一个 Sample_ID，不是类别编号。','cmp-note'));
    const filtered=filteredSamples(data,state.search,state.errors),pages=Math.max(1,Math.ceil(filtered.length/50));state.page=Math.min(state.page,pages-1);
    if(filtered.length)sample.append(chart('样品预测',{kind:'samples',page:state.page,search:state.search,errors:state.errors},tasks,{sample:true}));
    else sample.append(element('p','没有符合条件的样品。','cmp-note'));
    const pager=element('div','','cmp-actions'),prev=button('上一页',()=>{state.page--;paint();}),next=button('下一页',()=>{state.page++;paint();});prev.disabled=state.page===0;next.disabled=state.page>=pages-1;pager.append(prev,element('span',`${state.page+1}/${pages} 页 · ${filtered.length} 个样品`),next);sample.append(pager);
    sample.append(details('当前页样品详情',table(['模型','Sample_ID','真实类别','预测类别','测量条数','汇总方法'],models.flatMap(row=>{const entry=data.sample_correctness.values.find(item=>item.model_type===row.model_type);return filtered.slice(state.page*50,(state.page+1)*50).map(({id,index})=>{const detail=entry?.details?.[index]||{};return[modelName(row.model_type),id,detail.true_label,detail.pred_label,detail.measurement_count,detail.aggregation];});}))));root.append(sample);
  }
  if(FEATURE_ENGINEERING_ENABLED){
    const feature=section('特征工程 × 模型',[labeled('指标',select(metrics,state.featureMetric,value=>{state.featureMetric=value;paint();}))]);
    if(models.some(row=>row.experiment)){feature.append(chart('特征方案比较',{kind:'features',metric:state.featureMetric},tasks));feature.append(element('p','* 表示验证/CV 选定的方案，不按测试分数选优；留一法逐折配置见详情。','cmp-note'));}
    else feature.append(element('p','这批历史结果未记录特征工程比较数据。','cmp-note'));root.append(feature);
  }
  const audit=element('pre');audit.textContent=JSON.stringify({batch_id:state.id,evaluation,archive:meta,models:models.map(row=>({model:modelName(row.model_type),configuration:FEATURE_ENGINEERING_ENABLED?(row.experiment||null):undefined}))},null,2);
  root.append(details('配置与审计详情',audit));
  const excluded=(meta.runs||[]).filter(run=>!models.some(model=>model.run_ids?.includes(run.run_id)));
  if(excluded.length)root.append(element('p',`未参与比较：${excluded.map(run=>`${modelName(run.model_type)}（${run.state}）`).join('、')}`,'cmp-note'));
  root.querySelectorAll('details').forEach(node=>{node.open=openDetails.has(node.querySelector('summary')?.textContent);});
  // Bound parallel figure rendering; old responses cannot touch a new batch/filter.
  const token=renderToken;let next=0;
  const consume=async()=>{while(next<tasks.length&&token===renderToken){await tasks[next++]();}};
  consume();consume();
}

function mount(){const host=document.getElementById('batchComparisonPage');if(root?.isConnected)return;root=element('div','','cmp');host.replaceChildren(root);let width=0;observer?.disconnect();observer=new ResizeObserver(entries=>{const next=Math.floor(entries[0].contentRect.width);if(next===width)return;width=next;clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(state&&root.isConnected)paint();},200);});observer.observe(root);}
export function render(batchId,data){mount();const signature=JSON.stringify(data);if(!state||state.id!==batchId){imageCache.clear();state={id:batchId,data,signature,sort:'balanced_accuracy',matrixMode:'percent',featureMetric:'balanced_accuracy',page:0,search:'',errors:false,saved:false,saving:false,attempted:false};}else{if(state.signature!==signature)imageCache.clear();state.signature=signature;state.data=data;}paint();if(!state.attempted&&!['queued','running','cancelled'].includes(data.state)){state.attempted=true;ensureSaved();}}
export function dispose(){release();observer?.disconnect();observer=null;clearTimeout(resizeTimer);root=null;document.querySelectorAll('#view-comparison dialog').forEach(dialog=>dialog.remove());}
export function history(){dispose();state=null;mount();root.append(element('p','选择已保存的多模型比较，或从训练记录进入。'));const controls=element('div','','cmp-actions');root.append(controls);historyControls(controls);}
window.SpecAutoAIComparison={render,dispose,history};
window.dispatchEvent(new Event('specautoai:comparison-ready'));

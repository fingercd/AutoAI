/** Classic comparison only. No training/STOP/store mutation and no v2 imports. */
import { request, downloadFile } from './api-client.js';
import { experimentDetails } from './experiment-details.js';

const metrics = [['accuracy','Accuracy'],['balanced_accuracy','Balanced Accuracy'],['macro_f1','Macro-F1'],['weighted_f1','Weighted-F1']];
const featureMetrics = [['accuracy','准确率'],['balanced_accuracy','平衡准确率'],['macro_f1','宏平均 F1'],['weighted_f1','加权 F1']];
const names = {pls_da:'PLS-DA',logistic_regression:'Elastic Net',svm:'SVM',random_forest:'Random Forest',xgboost:'XGBoost',cnn1d:'1D-CNN'};
// 新版能力只从后端目录获取，不设置另一个独立的前端开关。
let trainingScheme = null;
let capabilityRequest = null;
const modelName = id => names[id] || id;
const element = (tag, text='', cls='') => { const node=document.createElement(tag);node.textContent=text;if(cls)node.className=cls;return node; };
const iconPaths = {
  download:'M12 3v12m-5-5 5 5 5-5M4 16v4h16v-4',
  table:'M4 4h16v16H4zM4 9h16M9 9v11',
  chart:'M4 3v17h17M8 16v-5m5 5V6m5 10V9',
  grid:'M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z',
  samples:'M4 6h16M4 12h16M4 18h16M8 4v16M16 4v16',
  layers:'m12 3 9 5-9 5-9-5 9-5zm-9 9 9 5 9-5M3 16l9 5 9-5',
  settings:'M4 7h16M4 17h16M9 4v6m6 4v6',
  expand:'M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5',
  check:'m5 12 4 4L19 6',
};
function icon(name) {
  const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
  svg.setAttribute('viewBox','0 0 24 24');svg.setAttribute('class','cmp-icon');svg.setAttribute('aria-hidden','true');
  const path=document.createElementNS(svg.namespaceURI,'path');path.setAttribute('d',iconPaths[name]);svg.append(path);return svg;
}
const button = (text, action, symbol) => {const node=element('button','','secondary');if(symbol)node.append(icon(symbol));node.append(element('span',text));node.type='button';node.addEventListener('click',action);return node;};
const select = (items, value, action) => {const node=element('select');items.forEach(([key,label])=>{const option=element('option',label);option.value=key;node.append(option);});node.value=value;node.addEventListener('change',()=>action(node.value));return node;};
const labeled = (text, control) => {const node=element('label',text);node.append(control);return node;};
const details = (title, content) => {const node=element('details');node.append(element('summary',title),content);return node;};
const focusKey = (node, key) => {node.dataset.focusKey=key;return node;};
const percent = value => typeof value==='number'?`${(value*100).toFixed(2)}%`:'—';
const table = (headers, rows) => {const wrap=element('div','','cmp-scroll'),node=element('table');const head=element('thead'),hr=element('tr');headers.forEach(text=>hr.append(element('th',text)));head.append(hr);const body=element('tbody');rows.forEach(row=>{const tr=element('tr');row.forEach(value=>tr.append(element('td',value==null?'—':String(value))));body.append(tr);});node.append(head,body);wrap.append(node);return wrap;};

export function sortedModels(data, sort='balanced_accuracy') {
  const value = row => typeof row.metrics?.[sort]?.mean==='number' ? row.metrics[sort].mean : -Infinity;
  return [...(data.models||[])].sort((a,b)=>value(b)-value(a)||(a.model_type<b.model_type?-1:a.model_type>b.model_type?1:0));
}
export function filteredSamples(data, search='', errors=false) {
  const sample=data.sample_correctness||{};
  return (sample.sample_ids||[]).map((id,index)=>({id,index})).filter(({id,index})=>String(id).toLowerCase().includes(search.toLowerCase())&&(!errors||(sample.values||[]).some(row=>row.values?.[index]===0)));
}
// 分类预测指标表：Recall 与 Precision 是两个不同指标，各占一张表；缺失值保持 —。
export function classMetricValue(item,key){return typeof item?.[key]==='number'?`${(item[key]*100).toFixed(2)}%`:'—';}
export function classMetricRows(data,models,key){
  const block=data.class_metrics||{};
  return models.map(row=>{const entry=(block.rows||[]).find(item=>item.model_type===row.model_type);return [row.model_type,...(entry?.values||[]).map(item=>classMetricValue(item,key))];});
}
export function classSupportRows(data,models){
  const block=data.class_metrics||{};
  return models.map(row=>{const entry=(block.rows||[]).find(item=>item.model_type===row.model_type);return [row.model_type,...(entry?.values||[]).map(item=>typeof item?.support==='number'?String(item.support):'—')];});
}

let state=null, root=null, observer=null, controller=null, renderToken=0, resizeTimer=null;
let urls=[];
const imageCache=new Map();
function release() {controller?.abort();document.querySelectorAll('#view-comparison dialog').forEach(dialog=>dialog.remove());urls.forEach(url=>URL.revokeObjectURL(url));urls=[];renderToken+=1;}
function blobUrl(blob) {const url=URL.createObjectURL(blob);urls.push(url);return url;}
function saveBlob(blob,name) {const url=URL.createObjectURL(blob),link=element('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function filename(options, extension) {return [options.kind, options.metric, options.model, options.kind==='samples'?`page-${options.page+1}`:'', options.kind==='matrix'?options.matrix_mode:''].filter(Boolean).join('_')+'.'+extension;}
function section(title, controls=[]) {const node=element('section','','cmp-section'),head=element('div','','cmp-toolbar');head.append(element('h3',title));if(controls.length){const actions=element('div','','cmp-actions');actions.append(...controls);head.append(actions);}node.append(head);return node;}

export function matrixZoomWidth(viewportWidth, viewportHeight) {
  // The square plot, controls and dialog padding should fit on one screen.
  return Math.max(260, Math.floor(Math.min(640, viewportWidth - 88, viewportHeight * .85 - 110)));
}

function chart(title, options, tasks, {sample=false, zoom=true}={}) {
  const card=element('figure','','cmp-figure cmp-figure-'+options.kind),head=element('div','','cmp-figure-head'),actions=element('div','','cmp-actions');
  const host=element('div','','cmp-scroll'),message=element('p','正在加载图表…','cmp-loading');host.append(message);
  let rendered=null, requested=null;
  async function download(format) {
    if(!rendered)return;
    try {
      if(format==='svg')saveBlob(new Blob([rendered.svg],{type:'image/svg+xml'}),filename(requested,'svg'));
      else {const result=await downloadFile(`/api/training/batches/${encodeURIComponent(state.id)}/figure?format=png`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(requested)});saveBlob(result.blob,filename(requested,'png'));}
    } catch(error) {message.textContent=error.message;message.className='cmp-error';host.append(message);}
  }
  const svg=button('SVG',()=>download('svg')),png=button('PNG',()=>download('png'));svg.disabled=png.disabled=true;
  svg.setAttribute('aria-label',`下载${title} SVG`);png.setAttribute('aria-label',`下载${title} PNG`);
  actions.append(svg,png);
  if(zoom){const zoomButton=button('放大',()=>{
    const matrix = options.kind === 'matrix';
    const dialog=element('dialog','','cmp'+(matrix?' cmp-matrix-dialog':''));
    dialog.setAttribute('aria-label',`${title}放大图`);
    const matrixWidth=matrixZoomWidth(window.innerWidth,window.innerHeight);
    if(matrix)dialog.style.width=`${matrixWidth+56}px`;
    const closeDialog=()=>{dialog.close();dialog.remove();zoomButton.focus({preventScroll:true});};
    const close=button('关闭',closeDialog);dialog.append(close);document.getElementById('view-comparison').append(dialog);dialog.showModal();
    const width=matrix?matrixWidth:Math.min(1400,Math.floor(dialog.clientWidth-60));
    const queue=[];dialog.append(chart(title,{...options,width},queue,{zoom:false}));queue.forEach(task=>task());
    dialog.addEventListener('cancel',event=>{event.preventDefault();closeDialog();});
    dialog.addEventListener('click',event=>{if(event.target===dialog){const rect=dialog.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)closeDialog();}});
  },'expand');zoomButton.setAttribute('aria-label',`放大${title}`);actions.append(zoomButton);}
  head.append(element('h4',title),element('span','导出图表','cmp-export-label'),actions);card.append(head,host);
  tasks.push(async()=>{
    const token=renderToken;
    const cardStyle=window.getComputedStyle(card);
    const innerWidth=card.clientWidth-parseFloat(cardStyle.paddingLeft)-parseFloat(cardStyle.paddingRight);
    const chartWidth=options.kind==='matrix'?Math.min(460,innerWidth):innerWidth;
    requested={sort:state.sort,width:Math.min(1800,Math.max(260,Math.floor(chartWidth))),...options};
    try {
      const cacheKey=JSON.stringify({batch:state.id,...requested});
      const result=imageCache.get(cacheKey)||await request(`/api/training/batches/${encodeURIComponent(state.id)}/figure`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(requested),signal:controller.signal});
      if(token!==renderToken||!card.isConnected)return;
      imageCache.set(cacheKey,result);if(imageCache.size>96)imageCache.delete(imageCache.keys().next().value);
      rendered=result;const url=blobUrl(new Blob([result.svg],{type:'image/svg+xml'}));
      const makeImage=()=>{const img=element('img');img.src=url;img.alt=title;img.width=result.width;img.height=result.height;if(!sample&&result.width<=innerWidth*1.15)img.className='cmp-fit';return img;};
      if(sample){const wrap=element('div','','cmp-sample-wrap'),labels=element('div','','cmp-sample-labels'),body=element('div','','cmp-sample-body');labels.setAttribute('aria-hidden','true');labels.append(makeImage());body.append(makeImage());wrap.append(labels,body);host.replaceChildren(wrap);}
      else host.replaceChildren(makeImage());
      svg.disabled=png.disabled=false;
    }catch(error){if(token===renderToken&&error.name!=='AbortError'){message.textContent=error.message;message.className='cmp-error';}}
  });
  return card;
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

function overview(data,models,tasks) {
  const host=element('div','','cmp-panel-content');
  const order=focusKey(select(metrics,state.sort,value=>{state.sort=value;paint();}),'sort');
  const overall=section('总体性能',[labeled('排序依据',order)]),grid=element('div','','cmp-grid cmp-metric-grid');
  metrics.forEach(([metric,title])=>grid.append(chart(title,{kind:'overall',metric},tasks)));overall.append(grid);
  overall.append(details('查看精确数值',table(['模型',...metrics.map(([,label])=>label)],models.map(row=>[modelName(row.model_type),...metrics.map(([key])=>percent(row.metrics?.[key]?.mean))]))));host.append(overall);
  const mode=focusKey(select([['percent','真实类别归一化'],['count','原始计数']],state.matrixMode,value=>{state.matrixMode=value;paint();}),'matrix-mode');
  const matrices=section('混淆矩阵',[labeled('显示方式',mode)]);
  matrices.append(element('p','纵轴为真实类别，横轴为预测类别。归一化范围为 0–100%，原始计数共享色标。','cmp-note'));
  const matrixGrid=element('div','','cmp-grid cmp-matrix-grid'+((data.confusion_matrices||[]).some(row=>row.labels.length>6)?' cmp-large-classes':''));
  models.forEach(model=>{const row=(data.confusion_matrices||[]).find(item=>item.model_type===model.model_type);if(row?.confusion_matrix)matrixGrid.append(chart(modelName(model.model_type),{kind:'matrix',model:model.model_type,matrix_mode:state.matrixMode},tasks));});
  matrices.append(matrixGrid.childElementCount?matrixGrid:element('p','这批结果没有可用的混淆矩阵。','cmp-empty'));host.append(matrices);
  return host;
}

function classification(data,models,tasks) {
  const cls=section('分类预测指标'),classMetrics=data.class_metrics;
  if(classMetrics?.status!=='ready'){cls.append(element('p',classMetrics?.reason||'这批结果没有可用的分类指标。','cmp-empty'));return cls;}
  const grid=element('div','','cmp-grid cmp-class-grid');
  for(const [kind,title] of [['recall','各类别召回率'],['precision','各类别 Precision（精确率）']]){
    const group=element('div','','cmp-chart-group');
    group.append(element('h4',title,'cmp-subtitle'),chart(title,{kind},tasks));
    group.append(details(`${title} · 精确数值`,table(['模型',...classMetrics.labels],classMetricRows(data,models,kind).map(row=>[modelName(row[0]),...row.slice(1)]))));
    grid.append(group);
  }
  cls.append(grid,details('各类别记录数（Support）',table(['模型',...classMetrics.labels],classSupportRows(data,models).map(row=>[modelName(row[0]),...row.slice(1)]))));
  return cls;
}

function samples(data,models,tasks) {
  const sample=section('Sample_ID × 模型预测正误');
  if(data.sample_correctness?.status!=='ready'){sample.append(element('p',data.sample_correctness?.reason||'这批结果没有可用的样品预测记录。','cmp-empty'));return sample;}
  const search=focusKey(element('input'),'sample-search');search.type='search';search.placeholder='输入 Sample_ID 搜索';search.value=state.search;search.setAttribute('aria-label','搜索 Sample_ID');
  const check=focusKey(element('input'),'sample-errors');check.type='checkbox';check.checked=state.errors;check.addEventListener('change',()=>{state.search=search.value;state.errors=check.checked;state.page=0;paint();});
  const apply=()=>{state.search=search.value;state.page=0;paint();};search.addEventListener('keydown',event=>{if(event.key==='Enter')apply();});
  search.addEventListener('search',apply);
  const controls=element('div','','cmp-sample-controls'),searchGroup=element('div','','cmp-actions');
  searchGroup.append(search,focusKey(button('搜索',apply),'sample-submit'),labeled('只看错误',check));
  const legend=element('div','','cmp-legend');legend.append(element('span','✓ 正确','cmp-correct'),element('span','× 错误','cmp-wrong'),element('span','— 缺失'));
  controls.append(searchGroup,legend);sample.append(controls);
  const filtered=filteredSamples(data,state.search,state.errors),pages=Math.max(1,Math.ceil(filtered.length/50));state.page=Math.min(state.page,pages-1);
  if(filtered.length)sample.append(chart('样品预测',{kind:'samples',page:state.page,search:state.search,errors:state.errors},tasks,{sample:true}));
  else {const empty=element('div','','cmp-empty');empty.append(element('p','没有符合条件的样品。'),button('清除筛选',()=>{state.search='';state.errors=false;state.page=0;paint();}));sample.append(empty);}
  const pager=element('div','','cmp-pager'),paging=element('div','','cmp-actions');
  const prev=focusKey(button('上一页',()=>{state.page--;paint();}),'sample-prev'),next=focusKey(button('下一页',()=>{state.page++;paint();}),'sample-next');prev.disabled=state.page===0;next.disabled=state.page>=pages-1;
  paging.append(prev,element('span',`${state.page+1} / ${pages}`),next);pager.append(element('span',`${filtered.length} 个样品 · 每页最多 50 个`,'cmp-note'),paging);sample.append(pager);
  sample.append(details('当前页样品详情',table(['模型','Sample_ID','真实类别','预测类别','测量条数','汇总方法'],models.flatMap(row=>{const entry=data.sample_correctness.values.find(item=>item.model_type===row.model_type);return filtered.slice(state.page*50,(state.page+1)*50).map(({id,index})=>{const detail=entry?.details?.[index]||{};return[modelName(row.model_type),id,detail.true_label,detail.pred_label,detail.measurement_count,detail.aggregation];});}))));
  return sample;
}

function features(data,models,tasks) {
  const metric=focusKey(select(featureMetrics,state.featureMetric,value=>{state.featureMetric=value;paint();}),'feature-metric');
  const feature=section('特征工程 × 模型',[labeled('评估指标',metric)]);
  if(models.some(row=>row.experiment?.schemes?.some(scheme=>scheme.status==='ready'))){
    feature.append(chart('特征方案比较',{kind:'features',metric:state.featureMetric},tasks));
    feature.append(element('p','* 表示验证/CV 选定的方案，不按测试分数选优；普通留一法无全局最佳标记，逐折配置见“参数与划分”。','cmp-note'));
  }else feature.append(element('p','这批历史结果未记录特征工程比较数据。','cmp-empty'));
  return feature;
}

function configurations(data,models) {
  const host=section('最佳配置与逐折参数');
  host.append(element('p','展开模型，查看所选方案、训练参数和实际数据划分。','cmp-note'));
  models.forEach(row=>host.append(details(modelName(row.model_type),experimentDetails(row.experiment))));
  return host;
}

function summary(data,models) {
  const current=state,meta=data.archive||{},evaluation=data.evaluation||{};
  const agg=evaluation.primary_aggregation==='pooled_oof'?'合并留一法预测（OOF）':evaluation.primary_aggregation==='direct_external_test'||evaluation.strategy?.includes('external_test')?'独立测试集':'留出测试集（Test）';
  const hero=element('section','','cmp-summary'),info=element('div','','cmp-summary-info');
  const statusNames={succeeded:'训练已完成',partial:'部分结果可用',failed:'训练失败',cancelled:'已停止',running:'训练进行中',queued:'等待训练'};
  const status=element('span',statusNames[data.state]||'结果已就绪','cmp-status');
  if(data.state==='succeeded')status.prepend(icon('check'));
  else status.classList.add('cmp-status-pending');
  const heading=element('div','','cmp-summary-title');heading.append(element('h3',meta.dataset_name||state.datasetName||'当前数据集'),status);
  info.append(heading,element('p',`评估范围：${agg}`,'cmp-note'));
  const stats=element('div','','cmp-stats');
  const sampleCount=data.sample_correctness?.status==='ready'?data.sample_correctness.sample_ids.length:null;
  const labels=data.class_metrics?.labels||data.confusion_matrices?.[0]?.labels;
  for(const [value,label] of [[data.comparable?models.length:null,'可比较模型'],[data.comparable?sampleCount:null,'测试样品 · Sample_ID'],[data.comparable?labels?.length:null,'类别']]){
    const item=element('div','','cmp-stat');item.append(element('strong',value==null?'—':String(value)),element('span',label));stats.append(item);
  }
  const top=element('div','','cmp-summary-top');top.append(info,stats);hero.append(top);
  const bottom=element('div','','cmp-summary-bottom'),actions=element('div','','cmp-actions');
  const all=button('下载全部',async()=>{try{all.disabled=true;const file=await downloadFile(`/api/training/batches/${encodeURIComponent(current.id)}/archive/files/all.zip`);saveBlob(file.blob,`comparison-${current.id.slice(0,8)}.zip`);}catch(error){current.saveText=error.message;}finally{if(state===current)updateSaveText();}},'download');
  all.classList.add('cmp-primary');all.dataset.downloadAll='';all.disabled=!current.saved;all.title='下载本批次的完整图表和数据（ZIP）';
  const message=element('p','','cmp-download-message');message.setAttribute('role','status');
  // Excel 依赖预测产物，不依赖图集归档完成。
  const excel=button('预测明细 Excel',async()=>{try{excel.disabled=true;const file=await downloadFile(`/api/training/batches/${encodeURIComponent(current.id)}/predictions.xlsx`);saveBlob(file.blob,`predictions-${current.id.slice(0,8)}.xlsx`);message.textContent='已下载：工作表 1 为预测类别，工作表 2 为逐类概率。';message.className='cmp-download-message';}catch(error){message.textContent=`Excel 下载失败：${error.message}`;message.className='cmp-download-message cmp-error';}finally{excel.disabled=false;}},'table');excel.dataset.downloadExcel='';excel.title='下载每条记录的预测类别与逐类概率';
  const retry=button('重试保存',()=>ensureSaved(true));retry.hidden=current.saved;retry.dataset.retrySave='';
  const note=element('span',current.saveText||'训练结束后自动保存','cmp-save-note');note.dataset.saveNote='';note.setAttribute('role','status');
  const save=element('div','','cmp-save');save.append(note,retry);
  actions.append(all,excel);bottom.append(actions,save);hero.append(bottom,message);
  return hero;
}

function tabs(data) {
  const items=[['overview','总体表现','chart',overview],['classes','分类指标','grid',classification],['samples','样品预测','samples',samples]];
  if(trainingScheme?.enabled)items.push(['features','特征方案','layers',features]);
  items.push(['configuration','参数与划分','settings',configurations]);
  if(!items.some(([key])=>key===state.tab))state.tab='overview';
  const list=element('div','','cmp-tabs');list.setAttribute('role','tablist');list.setAttribute('aria-label','结果分析');
  items.forEach(([key,label,symbol],index)=>{
    const tab=focusKey(button(label,()=>{state.tab=key;paint();},symbol),`tab-${key}`);
    tab.id=`cmp-tab-${key}`;tab.setAttribute('role','tab');tab.setAttribute('aria-selected',String(state.tab===key));tab.setAttribute('aria-controls',`cmp-panel-${key}`);tab.tabIndex=state.tab===key?0:-1;
    tab.addEventListener('keydown',event=>{
      let next=index;
      if(event.key==='ArrowRight')next=(index+1)%items.length;
      else if(event.key==='ArrowLeft')next=(index-1+items.length)%items.length;
      else if(event.key==='Home')next=0;
      else if(event.key==='End')next=items.length-1;
      else return;
      event.preventDefault();state.tab=items[next][0];paint();root.querySelector(`#cmp-tab-${state.tab}`).focus();
    });list.append(tab);
  });
  return {list,items};
}

function paint() {
  if(!root||!state)return;
  const active=document.activeElement,focused=root.contains(active)?active.dataset.focusKey:null;
  // Preserve disclosure state across tab changes, filters, archive completion and resize.
  root.querySelectorAll('details').forEach(node=>{const key=node.querySelector('summary')?.textContent;if(node.open)state.openDetails.add(key);else state.openDetails.delete(key);});
  release();controller=new AbortController();
  const data=state.archiveData||state.data,models=sortedModels(data,state.sort),tasks=[];
  root.replaceChildren(summary(data,models));
  if(!data.comparable){root.append(element('p',data.reason||'暂无完整可比较结果','cmp-empty'));return;}
  const {list,items}=tabs(data),workspace=element('div','','cmp-workspace');workspace.append(list);
  items.forEach(([key,,,renderPanel])=>{
    const panel=element('div','','cmp-tab-panel');panel.id=`cmp-panel-${key}`;panel.setAttribute('role','tabpanel');panel.setAttribute('aria-labelledby',`cmp-tab-${key}`);panel.tabIndex=0;panel.hidden=key!==state.tab;
    if(!panel.hidden)panel.append(renderPanel(data,models,tasks));workspace.append(panel);
  });root.append(workspace);
  const excluded=(data.archive?.runs||[]).filter(run=>!models.some(model=>model.run_ids?.includes(run.run_id)));
  if(excluded.length)root.append(element('p',`未参与比较：${excluded.map(run=>`${modelName(run.model_type)}（${run.state}）`).join('、')}`,'cmp-note cmp-excluded'));
  root.querySelectorAll('details').forEach(node=>{node.open=state.openDetails.has(node.querySelector('summary')?.textContent);});
  if(focused)root.querySelector(`[data-focus-key="${focused}"]`)?.focus({preventScroll:true});
  // Only request visible charts, with at most two concurrent requests.
  const token=renderToken;let next=0;
  const consume=async()=>{while(next<tasks.length&&token===renderToken){await tasks[next++]();}};
  consume();consume();
}

function mount(){const host=document.getElementById('batchComparisonPage');if(root?.isConnected)return;root=element('div','','cmp');host.replaceChildren(root);let width=0;observer?.disconnect();observer=new ResizeObserver(entries=>{const next=Math.floor(entries[0].contentRect.width);if(next===width)return;width=next;clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(state&&root.isConnected)paint();},200);});observer.observe(root);}
export function render(batchId,data){mount();const signature=JSON.stringify(data);if(!state||state.id!==batchId){imageCache.clear();state={id:batchId,data,signature,tab:'overview',openDetails:new Set(),sort:'balanced_accuracy',matrixMode:'percent',featureMetric:'balanced_accuracy',page:0,search:'',errors:false,saved:false,saving:false,attempted:false};}else{if(state.signature!==signature)imageCache.clear();state.signature=signature;state.data=data;}paint();if(!capabilityRequest){capabilityRequest=request('/api/models').then(result=>{trainingScheme=result.training_scheme;if(root?.isConnected)paint();}).catch(()=>{capabilityRequest=null;});}if(!state.attempted&&!['queued','running','cancelled'].includes(data.state)){state.attempted=true;ensureSaved();}}
export function dispose(){release();observer?.disconnect();observer=null;clearTimeout(resizeTimer);root=null;document.querySelectorAll('#view-comparison dialog').forEach(dialog=>dialog.remove());}
export function history(){dispose();state=null;mount();const link=element('a','前往训练记录','secondary');link.href='#/runs';root.append(element('p','请从训练记录中点击“查看所属对比”打开模型对比结果。'),link);}
window.SpecAutoAIComparison={render,dispose,history};
window.dispatchEvent(new Event('specautoai:comparison-ready'));

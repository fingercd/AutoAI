/** Read-only, reusable configuration audit for classic comparison and result pages. */
const textValue = value => value == null ? '—' : typeof value==='object' ? JSON.stringify(value) : String(value);
const pct = value => typeof value==='number' && Number.isFinite(value) ? `${(100*value).toFixed(2)}%` : '—';
export function configurationRows(experiment) {
  const rows=[];
  if (experiment?.selected_configuration) rows.push({label:'最终配置',...experiment.selected_configuration});
  const folds=experiment?.fold_configurations||[];
  if(folds.length>1) folds.forEach(fold=>rows.push({label:`第 ${fold.fold_index} 折`,...fold}));
  return rows;
}
export function configurationScore(row) {
  if(row.selection_metric==='validation_loss') return `验证损失 ${typeof row.selection_score==='number'?row.selection_score.toFixed(6):'—'}`;
  return row.selection_metric==='validation_balanced_accuracy'
    ? `验证集 Balanced Accuracy ${pct(row.selection_score)}`
    : `五折平均 Balanced Accuracy ${pct(row.selection_score)}`;
}
function node(tag,text) {const value=document.createElement(tag);if(text!=null)value.textContent=text;return value;}
function auditTable(headers,rows) {
  const wrapper=node('div');wrapper.style.overflowX='auto';const table=node('table');
  const head=node('thead'),tr=node('tr');headers.forEach(value=>tr.append(node('th',value)));head.append(tr);table.append(head);
  const body=node('tbody');rows.forEach(values=>{const row=node('tr');values.forEach(value=>row.append(node('td',textValue(value))));body.append(row);});table.append(body);wrapper.append(table);return wrapper;
}
export function experimentDetails(experiment) {
  const host=node('div');
  if(!experiment){host.append(node('p','历史结果没有记录特征方案选择信息。'));return host;}
  const rows=configurationRows(experiment);
  if(!experiment.selected_configuration) host.append(node('p','留一法每折独立选优；总体成绩来自各折选定配置的合并预测，不存在一个全局最佳参数。'));
  rows.forEach(row=>{
    const fold=row.label!=='最终配置';const section=node(fold?'details':'div');
    section.append(node(fold?'summary':'h4',`${row.label} · ${row.scheme_name||row.scheme_id} · ${configurationScore(row)}`));
    if(fold)section.append(node('p',`留出 Sample_ID：${(row.test_sample_ids||[]).join('、')||'历史未记录'}`));
    section.append(auditTable(['参数','实际值'],Object.entries(row.params||{})));
    if(row.transform)section.append(node('p',`特征数：${row.transform.input_features} → ${row.transform.output_features}；标准化：${row.transform.normalization}${row.transform.pca_components!=null?`；PCA 实际主成分数：${row.transform.pca_components}`:''}`));
    const splits=row.split_summary;
    if(splits){
      const total=Object.values(splits).reduce((sum,item)=>sum+item.sample_count,0);
      section.append(auditTable(['分区','Sample_ID 数','测量条数','样品占比'],Object.entries(splits).map(([key,item])=>[key,item.sample_count,item.measurement_count,total?pct(item.sample_count/total):'—'])));
      if(row.requested_ratio)section.append(node('p',`目标 Train:Valid${row.requested_ratio.test?':Test':''} = ${row.requested_ratio.train}:${row.requested_ratio.valid}${row.requested_ratio.test?':'+row.requested_ratio.test:''}。实际按 Sample_ID 整组分层，小样本下比例可能有取整偏差；独立 Test 不计入主数据划分目标。`));
    }
    host.append(section);
  });
  host.append(node('p','传统模型 Test 来自 Train+Valid 重训；训练期 Train/Valid 为选参后的审计结果，不作为独立泛化成绩。CNN 方案依据 validation loss 选择。'));
  return host;
}

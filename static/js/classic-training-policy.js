/* Classic entry only: versioned request policy and student-facing defaults. */
(function (global) {
  function splitPayload(selectedMode, values) {
    const [train,valid,test] = values.map(Number);
    const three = selectedMode==='holdout';
    if (![train,valid,...(three?[test]:[])].every(v=>Number.isInteger(v)&&v>0) || train+valid+(three?test:0)!==10)
      throw new Error(three?'训练、验证、测试比例必须为正整数且相加为 10':'训练、验证比例必须为正整数且相加为 10');
    const modes={holdout:'stratified_holdout',external:'external_test_holdout',loo:'leave_one_sample_id_cv',external_loo:'leave_one_sample_id_cv_with_external_test'};
    return {split_mode:modes[selectedMode],split_train:train,split_valid:valid,...(three?{split_test:test}:{})};
  }
  function payload(scheme, selectedMode, ratios, normalization, model, options={}) {
    if (!scheme?.enabled || scheme.version!=='word-0904') throw new Error('新版建模方案尚未就绪，请确认后端已更新后刷新页面');
    if (!['zscore','minmax'].includes(normalization)) throw new Error('新版方案请选择 Z-score 或 Min-Max 标准化');
    const d=scheme.defaults;
    const profile=options.training_profile || scheme.default_profile;
    const feature=options.feature_scheme || scheme.default_feature_scheme;
    if (!scheme.profiles?.some(item=>item.id===profile) || !scheme.feature_schemes?.some(item=>item.id===feature))
      throw new Error('训练配置尚未就绪，请更新并重启后端与 Worker 后刷新页面');
    if (profile==='quick' && model==='cnn1d' && feature.startsWith('pca_')) throw new Error('CNN 不支持 PCA，请选择全特征或相邻点合并');
    return {...splitPayload(selectedMode,ratios),experiment_version:scheme.version,model_type:model,normalization,
      training_profile:profile,feature_scheme:feature,
      epochs:d.epochs,batch_size:d.batch_size,learning_rate:d.learning_rate,weight_decay:d.weight_decay,
      scheduler_factor:d.scheduler_factor,scheduler_patience:d.scheduler_patience,min_learning_rate:d.min_learning_rate,
      early_stopping_patience:d.early_stopping_patience,seed:d.seed,class_balance:'none'};
  }
  // 学生界面只减少选择，不改变后端 quick/full 与四种评估模式的能力。
  function studentFeatureSchemes(scheme) {
    const labels = {full:'全特征', bin_10:'Binning（每 10 个相邻点合并）', pca_95:'PCA（保留 95% 方差）'};
    return (scheme?.feature_schemes || []).filter(item => Object.hasOwn(labels,item.id))
      .map(item => ({...item,name:labels[item.id]}));
  }
  function studentPayload(scheme, external, model, feature='full') {
    if (!studentFeatureSchemes(scheme).some(item=>item.id===feature))
      throw new Error('所选特征处理不可用，请刷新模型目录后重新选择');
    return payload(scheme,external?'external':'holdout',external?[8,2,0]:[8,1,1],'zscore',model,
      {training_profile:'quick',feature_scheme:feature});
  }
  global.SpecAutoAITrainingPolicy = {splitPayload,payload,studentFeatureSchemes,studentPayload};
})(typeof window==='undefined'?globalThis:window);

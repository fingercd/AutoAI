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
    const feature=scheme.forced_feature_schemes?.[model] || options.feature_scheme || scheme.default_feature_scheme;
    if (!scheme.profiles?.some(item=>item.id===profile) || !scheme.feature_schemes?.some(item=>item.id===feature))
      throw new Error('训练配置尚未就绪，请更新并重启后端与 Worker 后刷新页面');
    if (profile==='quick' && scheme.feature_schemes.find(item=>item.id===feature)?.unsupported_models?.includes(model)) throw new Error('该模型不支持所选外层特征处理，请选择相邻点合并');
    return {...splitPayload(selectedMode,ratios),experiment_version:scheme.version,model_type:model,normalization,
      training_profile:profile,feature_scheme:feature,
      epochs:d.epochs,batch_size:d.batch_size,learning_rate:d.learning_rate,weight_decay:d.weight_decay,
      scheduler_factor:d.scheduler_factor,scheduler_patience:d.scheduler_patience,min_learning_rate:d.min_learning_rate,
      early_stopping_patience:d.early_stopping_patience,seed:d.seed,class_balance:'none'};
  }
  // 学生界面只减少选择，不改变后端 quick/full 与四种评估模式的能力。
  function studentFeatureSchemes(scheme) {
    return (scheme?.feature_schemes || []).filter(item => item.student_visible === true);
  }
  function studentPayload(scheme, external, model, feature=scheme?.student_default_feature_scheme, cv=false) {
    if (!scheme?.forced_feature_schemes?.[model] && !studentFeatureSchemes(scheme).some(item=>item.id===feature))
      throw new Error('所选特征处理不可用，请刷新模型目录后重新选择');
    const mode=external?(cv?'external_loo':'external'):(cv?'loo':'holdout');
    return payload(scheme,mode,mode==='holdout'?[8,1,1]:[8,2,0],'zscore',model,
      {training_profile:'quick',feature_scheme:feature});
  }
  function studentPlan(scheme, models, features, external=false, cv=false) {
    if (!models.length) throw new Error('至少选择一个可用模型');
    const forced=scheme?.forced_feature_schemes || {};
    const deepCount=models.filter(model=>forced[model]).length, traditionalCount=models.length-deepCount;
    if (traditionalCount && !features.length) throw new Error('至少选择一种特征处理');
    const available=studentFeatureSchemes(scheme), configs=[], inapplicable=[];
    for (const feature of features) {
      const entry=available.find(item=>item.id===feature);
      if (!entry) throw new Error('所选特征处理不可用，请刷新模型目录后重新选择');
      for (const model of models.filter(model=>!forced[model])) {
        if (entry.unsupported_models?.includes(model)) {
          inapplicable.push({model_type:model,feature_scheme:feature,reason:entry.inapplicable_reasons?.[model] || '模型内置 PCA，不重复叠加外层 PCA'});
        } else configs.push(studentPayload(scheme,external,model,feature,cv));
      }
    }
    for(const model of models.filter(model=>forced[model]))configs.push(studentPayload(scheme,external,model,forced[model],cv));
    if (!configs.length) throw new Error('没有可运行的模型 × 特征处理组合：内置 PCA 模型不重复叠加外层 PCA');
    return {configs,inapplicable,traditionalCount,deepCount};
  }
  global.SpecAutoAITrainingPolicy = {splitPayload,payload,studentFeatureSchemes,studentPayload,studentPlan};
})(typeof window==='undefined'?globalThis:window);

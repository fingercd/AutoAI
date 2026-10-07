/* Classic entry only: mode-local split preferences and versioned request policy. */
(function (global) {
  const defaults = {holdout:[8,1,1], external:[8,2,0], loo:[8,2,0], external_loo:[8,2,0]};
  const key = 'specautoai.split-preferences.v1';
  function createPreferences(storage) {
    let saved = {};
    try { saved = JSON.parse(storage.getItem(key) || '{}'); } catch (_) {}
    if (!saved || typeof saved !== 'object' || Array.isArray(saved)) saved = {};
    const ratios = {};
    for (const mode of Object.keys(defaults)) {
      const value = saved.ratios?.[mode];
      ratios[mode] = Array.isArray(value) && value.length===3 && value.every(Number.isFinite) ? value : [...defaults[mode]];
    }
    let cv = saved.cv === true;
    const persist = () => {try {storage.setItem(key, JSON.stringify({ratios,cv}));} catch (_) {}};
    return {get: mode => [...ratios[mode]], set(mode, value) {ratios[mode]=[...value];persist();},
      getCv:()=>cv, setCv(value) {cv=Boolean(value);persist();}};
  }
  function mode(external, cv) {return external ? (cv?'external_loo':'external') : (cv?'loo':'holdout');}
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
  global.SpecAutoAITrainingPolicy = {createPreferences,mode,splitPayload,payload};
})(typeof window==='undefined'?globalThis:window);

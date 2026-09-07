"""0904 feature/model selection, isolated from the legacy model implementations.

All selection scores use inner validation only. Evaluation predictions are retained
per scheme, so heatmaps never select winners using their displayed test metrics.
"""
from __future__ import annotations
import copy
import math
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from sklearn.cross_decomposition import PLSRegression
from .feature_engineering import SCHEMES, FeatureTransform, FeatureClassifier


class PLS0904:
    """PLS-DA with the requested latent-component grid, without C-1 clipping."""
    def __init__(self, components):
        self.components = components

    def fit(self, x, y):
        self.classes_ = np.unique(y)
        if self.components > min(len(x) - 1, x.shape[1]):
            raise ValueError('PLS 潜变量数超过当前训练折的样本／特征维度')
        self.model = PLSRegression(n_components=self.components, scale=False, max_iter=500, tol=1e-6)
        self.model.fit(x, (np.asarray(y)[:, None] == self.classes_[None, :]).astype(float))
        if not np.isfinite(self.model.coef_).all():
            raise ValueError('PLS 拟合得到非有限系数')
        return self

    def predict_proba(self, x):
        response = np.asarray(self.model.predict(x))
        response -= response.max(axis=1, keepdims=True)
        values = np.exp(response)
        return values / values.sum(axis=1, keepdims=True)

    def predict(self, x):
        return self.classes_[self.predict_proba(x).argmax(axis=1)]


def build_estimator(config, y, label_count):
    from . import training as t
    return PLS0904(config.pls_components) if config.model_type == 'pls_da' else t.build_traditional_model(config, y, label_count)


def cnn_profile(sample_count: int, length: int) -> dict:
    band = 0 if sample_count <= 100 else 1 if sample_count <= 300 else 2
    feature_band = 0 if length <= 1000 else 1 if length <= 3000 else 2
    return {'N': sample_count, 'L': length, 'channels': [[8, 16, 32], [16, 32, 64], [32, 64, 128]][band], 'dropout': [.5, .4, .3][band], 'kernels': [[7, 5, 3], [9, 5, 3], [9, 7, 5]][feature_band], 'pools': [[2, 2, 2], [4, 2, 2], [4, 2, 2]][feature_band]}


class CNN0904(nn.Module):
    def __init__(self, profile: dict, class_count: int):
        super().__init__()
        blocks, previous = [], 1
        for channels, kernel, pool in zip(profile['channels'], profile['kernels'], profile['pools']):
            blocks.extend([nn.Conv1d(previous, channels, kernel, padding=kernel // 2), nn.BatchNorm1d(channels), nn.ReLU(), nn.MaxPool1d(pool)])
            previous = channels
        self.network = nn.Sequential(*blocks, nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(profile['dropout']), nn.Linear(previous, class_count))

    def forward(self, x):
        return self.network(x.unsqueeze(1) if x.ndim == 2 else x)


def candidate_configs(config, model_type, n_features, min_train):
    from . import training as t
    if model_type == 'pls_da':
        return [t._clone_config(config, pls_components=n) for n in (1,2,3,4,5,6,8,10,12,15)]
    if model_type == 'svm':
        return [t._clone_config(config, svm_kernel=kernel, svm_c=c, svm_gamma=gamma) for kernel in ('linear', 'rbf') for c in (.01, .1, 1., 10., 100.) for gamma in (('scale',) if kernel == 'linear' else ('scale', .001, .01))]
    return t._traditional_candidate_configs(config, model_type, n_features, min_train)


def eval_probabilities(prob, y, indices, labels, predictions=None):
    from . import training as t
    true = np.asarray(y)[indices]
    pred = np.argmax(prob, axis=1) if predictions is None else np.asarray(predictions)
    return {**t._classification_metrics_payload(true, pred, labels), 'true': true.tolist(), 'pred': pred.tolist(), 'probabilities': np.asarray(prob).tolist()}


def grouped_experiment_folds(y, groups, pool, seed, labels):
    """Stratify independent Sample_IDs, not their uneven measurement counts."""
    from sklearn.model_selection import StratifiedKFold
    from . import training as t
    pool = np.asarray(pool, dtype=int)
    mapping = t._group_label_map(y[pool], groups[pool])
    ids = np.asarray(list(mapping))
    targets = np.asarray([mapping[sid] for sid in ids])
    insufficient = [f'{name}（{int(sum(targets == i))} 个 Sample_ID）' for i, name in enumerate(labels) if sum(targets == i) < 5]
    if insufficient:
        raise ValueError('传统模型 5 折分组搜索要求每类至少 5 个 Sample_ID；当前不足：'+'、'.join(insufficient))
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return [(pool[np.isin(groups[pool], ids[train])], pool[np.isin(groups[pool], ids[valid])])
            for train, valid in splitter.split(ids, targets)]


def train_cnn(config, x, y, splits, groups, labels, cancel, progress, *, fixed_epochs=None):
    profile = cnn_profile(len(set(groups[splits['train']])), x.shape[1])
    if x.shape[1] < math.prod(profile['pools']):
        raise ValueError('变换后特征数不足以执行 CNN 三层池化')
    torch.manual_seed(config.model_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN0904(profile, len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.scheduler_factor, patience=config.scheduler_patience, min_lr=config.min_learning_rate)
    criterion = nn.CrossEntropyLoss()
    values = torch.as_tensor(x, dtype=torch.float32)
    targets = torch.as_tensor(y, dtype=torch.long)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(values[splits['train']], targets[splits['train']]), batch_size=config.batch_size, shuffle=True, generator=torch.Generator().manual_seed(config.model_seed))
    best_loss, best_state, best_epoch, stale, history = math.inf, None, 0, 0, []
    for epoch in range(1, (fixed_epochs or config.epochs) + 1):
        cancel(); model.train(); total_loss = 0.
        for bx, by in loader:
            cancel()
            # A final singleton batch must not be dropped; use accumulated BN
            # statistics when pooling has reduced its spatial dimension to one.
            if len(bx) == 1:
                for layer in model.modules():
                    if isinstance(layer, nn.BatchNorm1d):
                        layer.eval()
            optimizer.zero_grad(); logits = model(bx.to(device)); loss = criterion(logits, by.to(device)); loss.backward(); optimizer.step(); total_loss += float(loss.detach()) * len(by)
            model.train()
        model.eval()
        with torch.no_grad():
            valid_logits = model(values[splits['valid']].to(device))
            valid_loss = float(criterion(valid_logits, targets[splits['valid']].to(device)))
        if not math.isfinite(valid_loss):
            raise ValueError('CNN 验证损失为非有限值')
        history.append({'epoch': epoch, 'train_loss': total_loss / len(splits['train']), 'valid_loss': valid_loss, 'learning_rate': optimizer.param_groups[0]['lr']})
        progress(epoch)
        if valid_loss < best_loss:
            best_loss, best_state, best_epoch, stale = valid_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if not fixed_epochs:
            scheduler.step(valid_loss)
            if stale >= config.early_stopping_patience:
                break
    if not fixed_epochs:
        model.load_state_dict(best_state)
    model.eval()
    def predict(indices):
        chunks=[]
        with torch.no_grad():
            for start in range(0,len(indices),128):
                chunks.append(torch.softmax(model(values[indices[start:start+128]].to(device)), dim=1).cpu().numpy())
        return np.concatenate(chunks)
    return model, predict, history, best_loss, best_epoch, profile


def fit_experiment_fold(config, model_type, x_raw, y, groups, splits, labels, fold_index, cancel, progress, *, external_final=False):
    from . import training as t
    traditional = model_type != 'cnn1d'
    pool = sorted(set(splits['train']) | set(splits['valid']))
    inner = grouped_experiment_folds(y, groups, pool, config.split_seed, labels) if traditional else []
    results, search, winner = [], [], None
    for scheme_index, (scheme, name) in enumerate(SCHEMES):
        cancel()
        item = {'scheme_id': scheme, 'scheme_name': name, 'fold_index': fold_index, 'status': 'ready', 'reason': None}
        if not traditional and scheme.startswith('pca_'):
            results.append({**item, 'status': 'not_applicable', 'reason': 'CNN 保留光谱顺序，不使用 PCA 输入'}); continue
        def update(**extra):
            progress({'feature_scheme': name, 'feature_scheme_index': scheme_index + 1, 'feature_scheme_count': 7 if traditional else 4, 'training_stage': 'feature_search', 'training_stage_label': f'{name} · 特征方案比较', **extra})
        update()
        try:
            if traditional:
                # Transform once per inner fold/feature scheme, never using its validation rows in fit.
                prepared=[]
                for train, valid in inner:
                    cancel(); transform=FeatureTransform(scheme, config.normalization).fit(x_raw[train]); prepared.append((train, valid, transform.transform(x_raw[train]), transform.transform(x_raw[valid])))
                candidates=candidate_configs(config, model_type, min(p[2].shape[1] for p in prepared), min(len(p[0]) for p in prepared))
                best_score, best_candidate, selected_search = -math.inf, None, None
                for index, candidate in enumerate(candidates):
                    cancel(); scores=[]; reason=None
                    try:
                        for train, valid, train_x, valid_x in prepared:
                            cancel(); estimator=build_estimator(candidate,y[train],len(labels)); estimator.fit(train_x,y[train]); scores.append(float(t._evaluate_traditional_model(estimator,valid_x,y[valid],list(range(len(valid))),labels)['balanced_accuracy']))
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        reason=str(exc)
                    score=float(np.mean(scores)) if reason is None and len(scores)==5 else None
                    record={**item,'model_type':model_type,'candidate_index':index,'is_selected':False,'selection_metric':'mean_balanced_accuracy_grouped_5fold','selection_score':score,'params':t._traditional_params(candidate,model_type),'inner_fold_scores':scores,'status':'skipped' if reason else 'ready','reason':reason}
                    search.append(record)
                    if score is not None and math.isfinite(score) and score>best_score+1e-12:
                        best_score,best_candidate,selected_search=score,candidate,record
                    update(search_completed=index+1,search_total=len(candidates))
                if best_candidate is None:
                    raise ValueError('当前特征维度下没有完成全部五折的有效参数组合')
                selected_search['scheme_selected']=True
                transform=FeatureTransform(scheme,config.normalization).fit(x_raw[pool]); estimator=build_estimator(best_candidate,y[pool],len(labels)); estimator.fit(transform.transform(x_raw[pool]),y[pool]); model=FeatureClassifier(transform,estimator)
                evaluations={key:eval_probabilities(t._traditional_probabilities(estimator,transform.transform(x_raw[indices])),y,indices,labels,estimator.predict(transform.transform(x_raw[indices]))) for key,indices in splits.items() if indices}
                # Validation/train audit comes from a train-only fit, not the train+valid refit.
                if not external_final:
                    audit_transform=FeatureTransform(scheme,config.normalization).fit(x_raw[splits['train']]); audit=build_estimator(best_candidate,y[splits['train']],len(labels)); audit.fit(audit_transform.transform(x_raw[splits['train']]),y[splits['train']])
                    for key in ('train','valid'):
                        evaluations[key]=eval_probabilities(t._traditional_probabilities(audit,audit_transform.transform(x_raw[splits[key]])),y,splits[key],labels,audit.predict(audit_transform.transform(x_raw[splits[key]])))
                score,metric,params,history,profile=best_score,'mean_balanced_accuracy_grouped_5fold',t._traditional_params(best_candidate,model_type),[],None
                if model_type == 'logistic_regression':
                    params.update(penalty='elasticnet', solver='saga')
            else:
                transform=FeatureTransform(scheme,config.normalization).fit(x_raw[splits['train']]); x=transform.transform(x_raw)
                model,predict,history,loss,best_epoch,profile=train_cnn(config,x,y,splits,groups,labels,cancel,lambda epoch:update(current_epoch=epoch))
                score,metric,params=-loss,'validation_loss',{'best_epoch':best_epoch,**profile,
                    'optimizer':'AdamW','loss':'CrossEntropyLoss','batch_size':config.batch_size,
                    'learning_rate':config.learning_rate,'weight_decay':config.weight_decay,
                    'max_epochs':config.epochs,'scheduler':'ReduceLROnPlateau',
                    'scheduler_factor':config.scheduler_factor,'scheduler_patience':config.scheduler_patience,
                    'min_learning_rate':config.min_learning_rate,'early_stopping_patience':config.early_stopping_patience,
                    'seed':config.model_seed}
                if external_final:
                    # Fix both scheme score and epoch before refitting on all primary records.
                    transform=FeatureTransform(scheme,config.normalization).fit(x_raw[pool]); x=transform.transform(x_raw)
                    final_splits={**splits,'train':pool}
                    model,predict,_,_,_,profile=train_cnn(config,x,y,final_splits,groups,labels,cancel,lambda epoch:update(current_epoch=epoch),fixed_epochs=best_epoch)
                    params['selection_sample_count'] = params['N']
                    params.update(profile)  # Audit the actual all-primary refit architecture.
                evaluations={key:eval_probabilities(predict(indices),y,indices,labels) for key,indices in splits.items() if indices}
            item.update({'selection_score':-score if metric=='validation_loss' else score,'selection_metric':metric,'params':params,'transform':transform.metadata(),'metrics':t._metrics_from_eval(evaluations['test'],labels)})
            result={**item,'evals':evaluations,'model':model,'transformer':transform,'history':history,'profile':profile}
            results.append({**item, 'evals': {'test': evaluations['test']}})
            if winner is None or score>winner['_score']+1e-12:
                winner={**result,'_score':score}
        except (ValueError,np.linalg.LinAlgError) as exc:
            results.append({**item,'status':'failed','reason':str(exc)})
    if winner is None:
        raise ValueError('全部特征方案失败：'+'；'.join(f"{r['scheme_name']}: {r['reason']}" for r in results))
    for row in search:
        row['is_selected']=bool(row.get('scheme_selected') and row['scheme_id']==winner['scheme_id'])
    winner['results']=results;winner['search_rows']=search
    return winner


def summarize_experiments(folds, labels, *, external_final=None):
    from . import training as t
    def config(r):
        return {k:r.get(k) for k in ('fold_index','test_sample_ids','split_summary','requested_ratio','scheme_id','scheme_name','selection_metric','selection_score','params','transform')}
    source=[external_final] if external_final else folds
    schemes=[]
    for scheme,name in SCHEMES:
        entries=[next((r for r in f['results'] if r['scheme_id']==scheme),{}) for f in source]
        ok=[r for r in entries if r.get('status')=='ready']
        status='ready' if len(ok)==len(source) else 'not_applicable' if all(r.get('status')=='not_applicable' for r in entries) else 'partial' if ok else 'failed'
        metrics=None
        if status=='ready':
            metrics=t._classification_metrics_payload([v for r in ok for v in r['evals']['test']['true']],[v for r in ok for v in r['evals']['test']['pred']],labels)
        schemes.append({'scheme_id':scheme,'scheme_name':name,'status':status,'reason':'; '.join(sorted({r.get('reason') for r in entries if r.get('reason')})) or None,'metrics':metrics,'completed_folds':len(ok),'expected_folds':len(source),'configurations':[config(r) for r in ok]})
    return {'version':'word-0904','selected_configuration':config(external_final or folds[0]) if external_final or len(folds)==1 else None,'fold_configurations':[config(r) for r in folds],'schemes':schemes}

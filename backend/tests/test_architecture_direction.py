from pathlib import Path


def test_architecture_docs_forbid_worker_backgroundtasks_and_model_changes():
    context = Path('CONTEXT.md').read_text(encoding='utf-8')
    handoff = Path('docs/frontend_backend_handoff.md').read_text(encoding='utf-8')

    assert 'HTTP 请求只创建 queued Run，不直接启动训练' in context
    assert 'BackgroundTasks 不承担训练执行' in context
    assert '模型算法改动必须使用独立模型计划' in context
    assert 'owner_id' not in handoff.split('## 5. 创建训练 Run', 1)[1].split('## 6.', 1)[0]


def test_classification_v2_contract_is_documented_consistently():
    documents = [
        Path('README.md').read_text(encoding='utf-8'),
        Path('CONTEXT.md').read_text(encoding='utf-8'),
        Path('docs/frontend_backend_handoff.md').read_text(encoding='utf-8'),
    ]
    required_phrases = (
        '仅支持分类',
        'Sample_ID',
        'external_test_holdout',
        '8:2',
        'stratified_holdout',
        '8:1:1',
        'leave_one_sample_id_cv',
        'AdamW',
        'docx-classification-v3',
        'cnn_mamba1d',
        '依赖不可用',
        'TEMPORARILY_HIDDEN',
    )
    for document in documents:
        for phrase in required_phrases:
            assert phrase in document


def test_result_page_server_security_and_artifact_contracts_are_documented_consistently():
    readme = Path('README.md').read_text(encoding='utf-8')
    context = Path('CONTEXT.md').read_text(encoding='utf-8')
    handoff = Path('docs/frontend_backend_handoff.md').read_text(encoding='utf-8')
    result_contract = Path('docs/run_result_contract.md').read_text(encoding='utf-8')
    deployment = Path('deploy/server_deploy.md').read_text(encoding='utf-8')

    for document in (readme, context, handoff, deployment):
        assert 'AUTOAI_DEPLOYMENT_MODE' in document
        assert 'AUTOAI_API_TOKEN' in document
    for document in (readme, context, handoff, result_contract):
        assert '/api/training/runs/{run_id}/result' in document
        assert 'run-result-v1' in document
        assert 'pooled OOF' in document
    assert '#/results?run_id=' in readme
    assert '#/results?run_id=' in handoff
    assert 'allow_origins=["*"]' not in handoff
    assert '禁止 `*`' in handoff
    assert 'model.pkl' in handoff and '不在新结果页下载白名单' in handoff
    assert '当前没有正式计算 ROC-AUC' in handoff
    assert '--rebind-unowned' in readme
    assert '--rebind-unowned' in deployment


def test_wide_feature_v2_csv_contract_is_documented_consistently():
    documents = [
        Path('README.md').read_text(encoding='utf-8'),
        Path('CONTEXT.md').read_text(encoding='utf-8'),
        Path('docs/frontend_backend_handoff.md').read_text(encoding='utf-8'),
    ]

    for document in documents:
        assert 'wide-feature-v2' in document
        assert 'wide-feature-v1' in document
        assert 'Index,Label,Sample_ID,Name' in document.replace(', ', ',')
        assert '16,384' in document
        assert '16,380' in document
        assert '公共轴' in document
    assert 'xxx_encoding=column_headers' in documents[0]
    assert 'xxx_precision=float64-roundtrip' in documents[1]
    assert '旧六列数组/JSON' in documents[1]

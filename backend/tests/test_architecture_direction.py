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
        '15 个目标分类模型',
        '仅支持分类',
        'Repeat_index',
        'external_test_holdout',
        '8:2',
        'stratified_holdout',
        '8:1:1',
        'leave_one_repeat_index_cv',
        'balanced accuracy',
        'train+valid',
        'AdamW',
        'batch size 8',
        '最多 200 epochs',
        '最低 validation loss',
        'docx-classification-v2',
        'cnn_mamba1d',
        '依赖不可用',
    )
    for document in documents:
        for phrase in required_phrases:
            assert phrase in document

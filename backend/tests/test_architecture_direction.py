from pathlib import Path


def test_architecture_docs_forbid_worker_backgroundtasks_and_model_changes():
    context = Path('CONTEXT.md').read_text(encoding='utf-8')
    handoff = Path('docs/frontend_backend_handoff.md').read_text(encoding='utf-8')

    assert 'HTTP 请求只创建 queued Run，不直接启动训练' in context
    assert 'BackgroundTasks 不承担训练执行' in context
    assert '模型算法改动必须使用独立模型计划' in context
    assert 'owner_id' not in handoff.split('## 5. 创建训练 Run', 1)[1].split('## 6.', 1)[0]

from pathlib import Path


def test_index_imports_run_transport_modules():
    html = Path('static/index.html').read_text(encoding='utf-8')
    store = Path('static/js/training-store.js').read_text(encoding='utf-8')
    assert 'type="module" src="/static/js/training-store.js"' in html
    assert 'export async function createRun' in store
    assert 'export async function cancelRun' in store

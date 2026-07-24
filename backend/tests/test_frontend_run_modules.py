from pathlib import Path


def test_index_imports_run_transport_modules():
    html = Path('static/index.html').read_text(encoding='utf-8')
    store = Path('static/js/training-store.js').read_text(encoding='utf-8')
    assert 'type="module" src="/static/js/training-store.js"' in html
    assert 'export async function createRun' in store
    assert 'export async function cancelRun' in store
    assert 'export async function stopRun' in store
    assert '/stop' in store
    assert 'typeof store?.stopRun === "function"' in html
    assert 'typeof store?.cancelRun === "function"' in html
    assert 'id="stopTrain"' in html
    assert 'CURRENT_RUN_STORAGE_KEY' in html
    assert 'restoreCurrentRun()' in html

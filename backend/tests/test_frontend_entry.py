"""正式前端入口与已移除工作台的路由回归。"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


def test_classic_frontend_remains_available():
    response = TestClient(app).get('/')
    assert response.status_code == 200
    assert 'text/html' in response.headers['content-type']
    assert 'classic-training-policy.js' in response.text


@pytest.mark.parametrize('path', ['/v2', '/v2/', '/static/v2/index.html'])
def test_removed_workbench_is_not_served(path):
    assert TestClient(app, follow_redirects=False).get(path).status_code == 404

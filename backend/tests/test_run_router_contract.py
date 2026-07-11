import inspect

from fastapi.routing import APIRoute

from backend.app.main import app


def test_create_run_route_does_not_depend_on_background_tasks():
    route = next(item for item in app.routes if isinstance(item, APIRoute) and item.path == '/api/training/runs' and 'POST' in item.methods)
    assert 'background_tasks' not in inspect.signature(route.endpoint).parameters
    assert route.endpoint.__module__ == 'backend.app.routers.runs'

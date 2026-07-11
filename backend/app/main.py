from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .paths import RUNS_DIR, STATIC_DIR, ensure_storage
from .routers import catalog, datasets, preprocess, runs


ensure_storage()
app = FastAPI(title='AutoAI', version='0.1.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

if STATIC_DIR.exists():
    app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')

app.include_router(catalog.router)
app.include_router(datasets.router)
app.include_router(preprocess.router)
app.include_router(runs.router)

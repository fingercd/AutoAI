from __future__ import annotations

import json
import shutil
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import numpy as np

from .hplc import preprocess_hplc_files_with_preview
from .parsers import preprocess_raw_files_with_preview, summarize_modeling_csv
from .paths import DEFAULT_DATA, PREPROCESSED_DIR, RUNS_DIR, STATIC_DIR, UPLOADS_DIR, ensure_storage
from .training import list_runs, train_model


ensure_storage()
app = FastAPI(title="AutoAI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


UI_PAGES = {
    "workbench": "ui-workbench.html",
    "wizard": "ui-wizard.html",
    "dashboard": "ui-dashboard.html",
    "console": "ui-console.html",
    "minimal-lab": "ui-minimal-lab.html",
    "swiss": "ui-swiss.html",
    "dark-instrument": "ui-dark-instrument.html",
    "warm-paper": "ui-warm-paper.html",
}


def _save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "upload.csv").suffix or ".csv"
    target = UPLOADS_DIR / f"{uuid.uuid4().hex}{suffix}"
    with target.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    return target


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_status(run_id: str, payload: dict[str, Any]) -> None:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status_file = run_dir / "status.json"
    previous: dict[str, Any] = {}
    if status_file.exists():
        try:
            previous = json.loads(status_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
    status_file.write_text(json.dumps({"run_id": run_id, **previous, **payload}, ensure_ascii=False, indent=2), encoding="utf-8")


def _background_train(run_id: str, data_path: str, config: dict[str, Any]) -> None:
    try:
        _write_status(run_id, {"status": "running", "data_path": data_path, "test_data_path": config.get("test_data_path"), "started_at": _now_iso()})
        train_model(data_path, config, run_id=run_id)
    except Exception as exc:
        _write_status(run_id, {"status": "failed", "error": str(exc), "traceback": traceback.format_exc(), "completed_at": _now_iso()})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/ui")
def ui_gallery() -> FileResponse:
    return FileResponse(STATIC_DIR / "ui-gallery.html")


@app.get("/ui/{name}")
def ui_variant(name: str) -> FileResponse:
    filename = UI_PAGES.get(name)
    if not filename:
        raise HTTPException(status_code=404, detail="UI 方案不存在")
    return FileResponse(STATIC_DIR / filename)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/sample/summary")
def sample_summary() -> dict[str, Any]:
    if not DEFAULT_DATA.exists():
        raise HTTPException(status_code=404, detail="项目根目录未找到 data.csv")
    return summarize_modeling_csv(DEFAULT_DATA)


@app.post("/api/datasets/upload")
def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    path = _save_upload(file)
    try:
        summary = summarize_modeling_csv(path)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"dataset_path": str(path.resolve()), "summary": summary}


@app.post("/api/preprocess/{kind}")
def preprocess(
    kind: str,
    files: list[UploadFile] = File(...),
    start_row: int = Form(1),
    end_row: int | None = Form(None),
    range_mode: str = Form("row"),
    x_min: float | None = Form(None),
    x_max: float | None = Form(None),
    baseline_method: str = Form("arPLS"),
    baseline_order: str = Form("range_then_baseline"),
    hplc_interpolate: bool = Form(True),
    hplc_subtract_min: bool = Form(True),
    hplc_normalize_area: bool = Form(True),
) -> dict[str, Any]:
    if kind not in {"raman", "chromatography", "hplc"}:
        raise HTTPException(status_code=400, detail="kind 必须是 raman、chromatography 或 hplc")
    original_names = [Path(file.filename or f"sample_{idx}").stem for idx, file in enumerate(files, start=1)]
    saved_files = [_save_upload(file) for file in files]
    try:
        if kind == "hplc":
            result = preprocess_hplc_files_with_preview(
                saved_files,
                start_row=start_row,
                end_row=end_row,
                range_mode=range_mode,
                x_min=x_min,
                x_max=x_max,
                display_names=original_names,
                interpolate=hplc_interpolate,
                subtract_min=hplc_subtract_min,
                normalize_area=hplc_normalize_area,
            )
        else:
            result = preprocess_raw_files_with_preview(
                saved_files,
                kind=kind,
                start_row=start_row,
                end_row=end_row,
                range_mode=range_mode,
                x_min=x_min,
                x_max=x_max,
                baseline_method=baseline_method,
                baseline_order=baseline_order,
                display_names=original_names,
            )
        frame = result["frame"]
        output = PREPROCESSED_DIR / f"{kind}_{uuid.uuid4().hex[:10]}.csv"
        frame.to_csv(output, index=False, encoding="utf-8-sig")
        response: dict[str, Any] = {
            "output_path": str(output.resolve()),
            "download_url": f"/api/files?path={output.resolve()}",
            "rows": int(len(frame)),
            "range_mode": range_mode,
            "x_min": x_min,
            "x_max": x_max,
            "curves": result["curves"],
            "preview": frame.head(5).drop(columns=["XXX", "Intensity"]).to_dict(orient="records"),
        }
        if kind == "hplc":
            response.update({
                "baseline_order": None,
                "baseline_method": None,
                "hplc_interpolate": hplc_interpolate,
                "hplc_subtract_min": hplc_subtract_min,
                "hplc_normalize_area": hplc_normalize_area,
                "common_time": result.get("common_time", []),
            })
            if result.get("common_time"):
                npy_path = PREPROCESSED_DIR / f"common_time_{output.stem}.npy"
                np.save(npy_path, np.array(result["common_time"], dtype=np.float32))
                response["common_time_path"] = str(npy_path.resolve())
        elif kind == "raman":
            response.update({
                "baseline_order": baseline_order,
                "baseline_method": baseline_method,
            })
        else:
            response["baseline_order"] = None
            response["baseline_method"] = None
        return response
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/training/runs")
def create_run(background_tasks: BackgroundTasks, payload: dict[str, Any]) -> dict[str, Any]:
    data_path = payload.get("data_path") or str(DEFAULT_DATA.resolve())
    if not Path(data_path).exists():
        raise HTTPException(status_code=404, detail="训练数据文件不存在")
    test_data_path = payload.get("test_data_path")
    if test_data_path and not Path(test_data_path).exists():
        raise HTTPException(status_code=404, detail="测试数据文件不存在")
    run_id = uuid.uuid4().hex[:12]
    config = payload.get("config") or {}
    if test_data_path:
        config = {**config, "test_data_path": test_data_path}
    _write_status(run_id, {"status": "pending", "data_path": data_path, "test_data_path": test_data_path, "config": config, "created_at": _now_iso()})
    background_tasks.add_task(_background_train, run_id, data_path, config)
    return {"run_id": run_id, "status": "pending"}


@app.get("/api/training/runs")
def get_runs() -> list[dict[str, Any]]:
    return list_runs()


@app.get("/api/training/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    status_file = RUNS_DIR / run_id / "status.json"
    if not status_file.exists():
        raise HTTPException(status_code=404, detail="run 不存在")
    return json.loads(status_file.read_text(encoding="utf-8"))


@app.get("/api/training/runs/{run_id}/artifact/{name}")
def get_run_artifact(run_id: str, name: str) -> FileResponse:
    allowed = {
        "config.json",
        "label_map.json",
        "split.json",
        "metrics.json",
        "cv_metrics.json",
        "fold_metrics.csv",
        "history.csv",
        "predictions.csv",
        "cv_predictions.csv",
        "hyperparameter_search.csv",
        "feature_importance.json",
        "feature_importance.csv",
        "sample_feature_importance.json",
        "sample_feature_importance.csv",
        "model.pt",
        "model.pkl",
        "status.json",
    }
    if name not in allowed:
        raise HTTPException(status_code=400, detail="不允许下载该文件")
    path = RUNS_DIR / run_id / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(path, filename=name)


@app.get("/api/files")
def get_file(path: str) -> FileResponse:
    target = Path(path).resolve()
    roots = [UPLOADS_DIR.resolve(), PREPROCESSED_DIR.resolve(), RUNS_DIR.resolve()]
    if not any(str(target).startswith(str(root)) for root in roots):
        raise HTTPException(status_code=403, detail="不允许访问该路径")
    if not target.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(target, filename=target.name)

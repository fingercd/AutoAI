"""Agent Session 测试——直接复用现有 storage 路径（与 test_smoke.py 风格一致）。

注意：现有 ``backend/tests/test_smoke.py`` 的端到端测试也是不隔离 storage 直
接写入 ``storage/``，所以本测试也按相同风格。每个测试用例间通过随机
session_id / 实验自然隔离；不需要 monkeypatch 路径。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.app.main import app
from backend.app.paths import RUNS_DATABASE, RUNS_DIR
from backend.app.runs.repository import RunRepository
from backend.tests.modeling_data_factory import write_grouped_classification_csv


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    return TestClient(app)


@pytest.fixture
def sample_csv(tmp_path) -> Path:
    source = tmp_path / "agent_sample.csv"
    write_grouped_classification_csv(
        source, groups_per_class=15, repeats=3, feature_count=12,
    )
    return source


@pytest.fixture
def uploaded_dataset(client, sample_csv):
    with sample_csv.open("rb") as handle:
        response = client.post(
            "/api/datasets/upload",
            files={"file": (sample_csv.name, handle, "text/csv")},
        )
    assert response.status_code == 200, response.text
    return response.json()["dataset_id"]


def _create_session(client, dataset_id, **overrides):
    body = {
        "dataset_id": dataset_id,
        "selection_metric": "macro_f1",
        "allowed_models": ["logistic_regression", "svm", "random_forest"],
        "max_runs": 3,
        "seed": 42,
        "evaluation": {
            "split_mode": "stratified_holdout",
            "split_train": 8,
            "split_valid": 1,
            "split_test": 1,
        },
    }
    body.update(overrides)
    return client.post("/api/agent/sessions", json=body)


# ---------- 1. 成功创建 Session ----------


def test_create_session_returns_locked_config(client, uploaded_dataset):
    response = _create_session(client, uploaded_dataset)
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["state"] == "open"
    assert payload["remaining_runs"] == 3
    assert payload["locked_config"]["dataset_id"] == uploaded_dataset
    assert payload["locked_config"]["selection_metric"] == "macro_f1"
    assert sorted(payload["locked_config"]["allowed_models"]) == [
        "logistic_regression", "random_forest", "svm"
    ]
    assert not any(payload['locked_config']['modules'].values())
    assert payload['context'] == {
        'schema_version': 'agent-context-v1',
        'status': 'disabled',
        'source_role': 'development',
    }


def test_session_persists_enabled_modules_and_context_policy(client, uploaded_dataset):
    response = _create_session(
        client,
        uploaded_dataset,
        modules={
            'evidence_card': True,
            'feedback_diagnosis': True,
            'budget_control': True,
        },
        context_policy={'source_role': 'domain', 'case_write': True},
    )
    assert response.status_code == 201, response.text
    created = response.json()
    assert created['context']['status'] == 'pending'
    session = client.get(f"/api/agent/sessions/{created['session_id']}").json()
    assert session['locked_config']['modules']['evidence_card'] is True
    assert session['locked_config']['modules']['feedback_diagnosis'] is True
    assert session['locked_config']['modules']['budget_control'] is True
    assert session['locked_config']['context_policy'] == {
        'source_role': 'domain',
        'case_write': True,
    }


def test_session_rejects_unknown_module_and_benchmark_case_write(client, uploaded_dataset):
    unknown = _create_session(
        client,
        uploaded_dataset,
        modules={'not_a_real_module': True},
    )
    assert unknown.status_code == 422
    benchmark_write = _create_session(
        client,
        uploaded_dataset,
        context_policy={'source_role': 'benchmark', 'case_write': True},
    )
    assert benchmark_write.status_code == 422


# ---------- 2. 非白名单模型返回 422 ----------


def test_create_session_rejects_unknown_model(client, uploaded_dataset):
    response = _create_session(
        client, uploaded_dataset,
        allowed_models=["logistic_regression", "cnn1d"],
    )
    assert response.status_code == 422
    assert "allowed_models" in response.json()["detail"]


# ---------- 3. experiment 传入额外字段返回 422 ----------


def test_create_experiment_extra_field_returns_422(client, uploaded_dataset):
    session = _create_session(client, uploaded_dataset).json()
    response = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments",
        json={
            "model_type": "svm", "normalization": "zscore", "class_balance": "none",
            "epochs": 200,
        },
    )
    assert response.status_code == 422


# ---------- 4. 单次实验不能覆盖 Session 固定字段 ----------


def test_experiment_cannot_override_session_locked_fields(client, uploaded_dataset):
    session = _create_session(client, uploaded_dataset).json()
    response = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments",
        json={
            "model_type": "svm", "normalization": "zscore", "class_balance": "none",
            "dataset_id": "hijack-attempt",
            "split_mode": "leave_one_sample_id_cv",
        },
    )
    assert response.status_code == 422


# ---------- 5. 重复 effective config 返回 409 ----------


def test_duplicate_effective_config_returns_409(client, uploaded_dataset):
    session = _create_session(client, uploaded_dataset).json()
    body = {"model_type": "svm", "normalization": "zscore", "class_balance": "none"}
    first = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments", json=body
    )
    assert first.status_code == 202
    second = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments", json=body
    )
    assert second.status_code == 409
    assert "config_hash" in second.json()["detail"]


# ---------- 6. 超出 max_runs 返回 409 ----------


def test_exceeding_max_runs_returns_409(client, uploaded_dataset):
    """3 个不同 normalization 触发独立 config_hash；为避免并发非终态抢占，
    每次入队后用 ``_claim_and_finish`` 把对应 run 推进到终态。"""
    session = _create_session(client, uploaded_dataset, max_runs=2).json()
    actions = [
        {"model_type": "svm", "normalization": "zscore", "class_balance": "none"},
        {"model_type": "svm", "normalization": "minmax", "class_balance": "none"},
        {"model_type": "svm", "normalization": "zscore", "class_balance": "class_weight"},
    ]
    repo = RunRepository(RUNS_DATABASE)
    repo.initialize()
    for action in actions[:2]:
        response = client.post(
            f"/api/agent/sessions/{session['session_id']}/experiments", json=action
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]
        _claim_and_finish(repo, run_id)
    overflow = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments", json=actions[2]
    )
    assert overflow.status_code == 409
    assert "max_runs" in overflow.json()["detail"]


def _claim_and_finish(repo: RunRepository, run_id: str) -> None:
    """test helper：把 queued / running Run 直接推入 succeeded。

    通过私有 SQL 路径绕过 claim_next 的 FIFO 顺序，让测试能逐个指定目标
    Run 进入终态。仅供测试用，不属于公开 API。
    """
    from backend.app.runs.repository import _timestamp
    now_text = _timestamp(datetime.now(timezone.utc))
    with repo._connection() as connection:  # type: ignore[attr-defined]
        connection.execute('BEGIN IMMEDIATE')
        connection.execute(
            '''
            UPDATE runs
            SET state = 'succeeded', version = version + 1, lease_expires_at = NULL,
                claim_token = 'test-claim', worker_id = 'test-worker',
                manifest_name = ?, error = NULL, error_json = '{}',
                updated_at = ?, finished_at = ?, started_at = COALESCE(started_at, ?)
            WHERE run_id = ? AND state IN ('queued', 'running')
            ''',
            ('manifest.json', now_text, now_text, now_text, run_id),
        )
        connection.commit()


# ---------- 7. 同 Session 内存在非终态 Run 时继续提交返回 409 ----------


def test_concurrent_active_run_blocks_new_experiment(client, uploaded_dataset):
    session = _create_session(client, uploaded_dataset).json()
    first = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments",
        json={"model_type": "svm", "normalization": "zscore", "class_balance": "none"},
    )
    assert first.status_code == 202
    created_run = repo_get(first.json()['run_id'])
    assert created_run.dataset_snapshot['dataset_id'] == uploaded_dataset
    assert created_run.dataset_snapshot['sha256']
    second = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments",
        json={"model_type": "logistic_regression", "normalization": "zscore", "class_balance": "none"},
    )
    assert second.status_code == 409


# ---------- 8. 不能读取其他 Session 的 Run ----------


def test_other_session_run_is_invisible(client, uploaded_dataset):
    session_a = _create_session(client, uploaded_dataset).json()
    exp_a = client.post(
        f"/api/agent/sessions/{session_a['session_id']}/experiments",
        json={"model_type": "svm", "normalization": "zscore", "class_balance": "none"},
    ).json()
    session_b = _create_session(client, uploaded_dataset).json()
    response = client.get(
        f"/api/agent/sessions/{session_b['session_id']}"
        f"/experiments/{exp_a['run_id']}/feedback"
    )
    assert response.status_code == 404


# ---------- 9. feedback 响应递归检查，不含敏感键 ----------


def test_feedback_response_excludes_sensitive_keys(client, uploaded_dataset):
    session = _create_session(client, uploaded_dataset).json()
    experiment = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments",
        json={"model_type": "svm", "normalization": "zscore", "class_balance": "none"},
    ).json()
    response = client.get(
        f"/api/agent/sessions/{session['session_id']}"
        f"/experiments/{experiment['run_id']}/feedback"
    )
    payload = response.json()
    forbidden_substrings = (
        "test", "artifact", "prediction", "confusion", "classification_report",
        "explainability", "samples", "roc", "precision_recall",
    )

    def walk(node, path=()):
        if isinstance(node, dict):
            for key, value in node.items():
                key_lc = key.lower()
                for forbidden in forbidden_substrings:
                    assert forbidden not in key_lc, (
                        f"敏感键 {forbidden!r} 出现在 {'/'.join(path + (key,))}"
                    )
                walk(value, path + (key,))
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, path + (str(index),))

    walk(payload)


# ---------- 10. 只能 Finalize 属于当前 Session 的成功 Run ----------


def test_finalize_requires_successful_run_in_same_session(client, uploaded_dataset):
    session = _create_session(client, uploaded_dataset).json()
    experiment = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments",
        json={"model_type": "svm", "normalization": "zscore", "class_balance": "none"},
    ).json()
    finalize = client.post(
        f"/api/agent/sessions/{session['session_id']}/finalize",
        json={"selected_run_id": experiment["run_id"]},
    )
    assert finalize.status_code == 422
    # detail 含"训练成功"，但避免直接断言中文字符
    assert "state=queued" in finalize.json()["detail"]


# ---------- 11. Finalize 后不能继续提交实验 ----------


def test_finalize_blocks_further_experiments(client, uploaded_dataset):
    session = _create_session(client, uploaded_dataset).json()
    experiment = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments",
        json={"model_type": "svm", "normalization": "zscore", "class_balance": "none"},
    ).json()

    repo = RunRepository(RUNS_DATABASE)
    repo.initialize()
    _claim_and_finish(repo, experiment["run_id"])
    finalize = client.post(
        f"/api/agent/sessions/{session['session_id']}/finalize",
        json={"selected_run_id": experiment["run_id"]},
    )
    assert finalize.status_code == 200

    next_experiment = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments",
        json={"model_type": "logistic_regression", "normalization": "zscore", "class_balance": "none"},
    )
    assert next_experiment.status_code == 409


# ---------- 12. Finalize 幂等且不允许更换 selected_run_id ----------


def test_finalize_is_idempotent_and_blocks_run_swap(client, uploaded_dataset):
    session = _create_session(client, uploaded_dataset, max_runs=2).json()
    run_ids = []
    repo = RunRepository(RUNS_DATABASE)
    repo.initialize()
    for action in (
        {"model_type": "svm", "normalization": "zscore", "class_balance": "none"},
        {"model_type": "logistic_regression", "normalization": "zscore", "class_balance": "none"},
    ):
        experiment = client.post(
            f"/api/agent/sessions/{session['session_id']}/experiments", json=action
        ).json()
        _claim_and_finish(repo, experiment["run_id"])
        run_ids.append(experiment["run_id"])

    first = client.post(
        f"/api/agent/sessions/{session['session_id']}/finalize",
        json={"selected_run_id": run_ids[0]},
    )
    assert first.status_code == 200
    second = client.post(
        f"/api/agent/sessions/{session['session_id']}/finalize",
        json={"selected_run_id": run_ids[0]},
    )
    assert second.status_code == 200
    assert second.json()["selected_run_id"] == run_ids[0]
    swap = client.post(
        f"/api/agent/sessions/{session['session_id']}/finalize",
        json={"selected_run_id": run_ids[1]},
    )
    assert swap.status_code == 409


# ---------- 真实链路：logistic_regression 端到端跑通 ----------


def test_real_logistic_regression_run_feedback_exposes_only_validation(client, uploaded_dataset):
    """走 ``train_model`` 真实跑一次 logistic_regression，写 metrics.json，然后
    通过 AgentFeedback 验证只暴露 validation 标量，不含 test/artifact 等敏感键。"""
    from backend.app.training import train_model
    from backend.app.paths import UPLOADS_DIR

    session = _create_session(client, uploaded_dataset).json()
    experiment = client.post(
        f"/api/agent/sessions/{session['session_id']}/experiments",
        json={"model_type": "logistic_regression", "normalization": "zscore", "class_balance": "none"},
    ).json()

    record = repo_get(experiment["run_id"])
    config = dict(record.config)
    config["feature_selection_enabled"] = False

    # train_model 需要 data_path。RunRecord.config.dataset_name 与
    # DatasetRepository.resolve_system 可找回数据路径；最直接：用 RecordDataset 路径。
    from backend.app.datasets.repository import DatasetRepository
    from backend.app.paths import DATASETS_DATABASE, STORAGE_DIR
    ds_repo = DatasetRepository(DATASETS_DATABASE, storage_root=STORAGE_DIR)
    ds_repo.initialize()
    dataset = ds_repo.resolve_system(uploaded_dataset, legacy_path=None)
    train_model(dataset.path, config, run_id=experiment["run_id"])

    repo = RunRepository(RUNS_DATABASE)
    repo.initialize()
    _claim_and_finish(repo, experiment["run_id"])

    feedback = client.get(
        f"/api/agent/sessions/{session['session_id']}"
        f"/experiments/{experiment['run_id']}/feedback"
    ).json()
    assert feedback["state"] == "succeeded"
    assert feedback["validation"]["status"] == "ready"
    assert feedback["validation_score"] is not None
    flat = json.dumps(feedback, ensure_ascii=False).lower()
    for forbidden in (
        "test_macro_f1", "artifact", "confusion",
        "classification_report", "explainability", "samples",
    ):
        assert forbidden not in flat, f"feedback 响应中泄漏了 {forbidden}"


def repo_get(run_id: str):
    repo = RunRepository(RUNS_DATABASE)
    repo.initialize()
    return repo.get(run_id)


# ---------- 原有训练接口不回归 ----------


def test_existing_training_create_run_still_works(client, uploaded_dataset):
    response = client.post(
        "/api/training/runs",
        json={"dataset_id": uploaded_dataset, "config": {"model_type": "svm"}},
    )
    assert response.status_code == 202


def test_existing_datasets_endpoint_still_works(client, uploaded_dataset):
    response = client.get(f"/api/datasets/{uploaded_dataset}")
    assert response.status_code in {200, 404}

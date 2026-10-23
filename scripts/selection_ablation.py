#!/usr/bin/env python3
"""Approved single-model selection ablation. prepare/run/report never train."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import urllib.error
import urllib.request

from backend.app.benchmark_tracking import OnlineJournal, atomic_json, read_json
from backend.app.selection_ablation import (
    CONDITIONS, MODELS, PROTOCOL, SETTINGS, canonical, digest,
    make_request, parse_action, summarize, validate_scores,
)

REPO = Path(__file__).resolve().parents[1]
TASK = "第一步：同条件知识选模对照-v1"
SOURCE_FILES = ("backend/app/selection_ablation.py", "scripts/selection_ablation.py",
                "backend/app/benchmark_tracking.py", "backend/tests/test_selection_ablation.py",
                "docs/research/step1-selection-ablation-v1.md", "scripts/serve_selection_llm.py")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(path, value):
    previous = read_json(path)
    if previous is not None and previous != value:
        raise ValueError("Refusing to overwrite a frozen file: " + str(path))
    if previous is None:
        atomic_json(path, value)


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(root, source):
    approval = read_json(root / "approval.json")
    if not approval or approval.get("protocol") != PROTOCOL or approval.get("user_approved") is not True:
        raise ValueError("Explicit approval for this protocol is required")
    score_path = REPO / "docs/research/results/seven-model-scores.json"
    scores = read_json(score_path)
    table = validate_scores(scores)
    source_manifest_path = source / "approval/selection-manifest-v2.json"
    old_manifest = read_json(source_manifest_path)
    if len(table) != 129 or len(old_manifest["tasks"]) != 129:
        raise ValueError("Approved scope requires exactly 129 tasks")
    expected_settings = SETTINGS
    if old_manifest["settings"] != expected_settings:
        raise ValueError("Frozen source settings changed")
    preflight = read_json(root / "preflight.json")
    if not preflight or preflight.get("verified") is not True:
        raise ValueError("Model asset and source preflight is required")
    entries, keys = [], set()
    for index, item in enumerate(old_manifest["tasks"]):
        original = read_json(Path(item["input_path"]))
        key = (original["suite"], original["task_id"])
        if key in keys or key not in table:
            raise ValueError("Input tasks do not match score tasks")
        keys.add(key)
        if digest(original["request"]) != item["request_sha256"]:
            raise ValueError("Source request hash mismatch")
        score = table[key]
        if any(original[field] != score[field] for field in ("dataset_sha256", "partition_digest", "family_id")):
            raise ValueError("Frozen task binding mismatch")
        conditions = CONDITIONS if index % 2 == 0 else tuple(reversed(CONDITIONS))
        for condition in conditions:
            request = make_request(original, condition)
            request_key = condition + "-" + item["key"]
            path = root / "inputs" / (request_key + ".json")
            frozen = dict(key=request_key, suite=key[0], task_id=key[1], family_id=score["family_id"],
                          condition=condition, request=request, request_sha256=digest(request))
            freeze(path, frozen)
            entries.append({k: frozen[k] for k in ("key", "suite", "task_id", "condition", "request_sha256")})
    if keys != set(table):
        raise ValueError("Missing frozen input")
    source_hashes = {name: file_sha(REPO / name) for name in SOURCE_FILES}
    config = dict(protocol=PROTOCOL, task_count=129, normal_requests=258, max_physical_requests=258,
        classifier_fits=0, retry_requests=0, settings=SETTINGS,
        conditions=list(CONDITIONS), fixed_strategies=list(MODELS), primary_metric="test_balanced_accuracy",
        selection="One model per condition, no validation-based model competition",
        order="Original task order, paired conditions alternate first position by task index",
        failed_choices="No fallback; report failure and paired coverage",
        comparison_scope="Exploratory tasks previously used for knowledge construction",
        source_hashes=source_hashes, original_manifest_sha256=file_sha(source_manifest_path),
        score_table_sha256=file_sha(score_path), original_input_count=129,
        git_head=preflight["git_head"], code_tag=preflight["code_tag"],
        model_asset=preflight["health"]["asset"], model_weight_sha256=preflight["weight_sha256"],
        model_service_sha256=preflight["service_source_sha256"],
        python_version=platform.python_version(), command=preflight["run_command"])
    manifest = dict(protocol=PROTOCOL, entries=entries, public_config=config,
        score_table=str(score_path), preflight_sha256=file_sha(root / "preflight.json"),
        approval_sha256=file_sha(root / "approval.json"))
    freeze(root / "manifest.json", manifest)
    print(json.dumps(dict(prepared=len(entries), tasks=len(keys), classifier_fits=0), ensure_ascii=False))


def verify_frozen(root):
    manifest = read_json(root / "manifest.json")
    if not manifest or manifest["protocol"] != PROTOCOL:
        raise ValueError("Prepared manifest required")
    for relative, expected in manifest["public_config"]["source_hashes"].items():
        if file_sha(REPO / relative) != expected:
            raise ValueError("Code changed after freeze: " + relative)
    if file_sha(root / "approval.json") != manifest["approval_sha256"]:
        raise ValueError("Approval changed")
    if file_sha(root / "preflight.json") != manifest["preflight_sha256"]:
        raise ValueError("Preflight changed")
    if file_sha(manifest["score_table"]) != manifest["public_config"]["score_table_sha256"]:
        raise ValueError("Score table changed")
    return manifest


def ensure_health(root, opener, endpoint):
    expected = read_json(root / "preflight.json")["health"]
    with opener.open(endpoint + "/health", timeout=10) as response:
        actual = json.load(response)
    for key in ("model", "asset", "dtype", "gpu_physical_index", "service_context_window",
                "enable_thinking", "asset_metadata"):
        if actual.get(key) != expected.get(key):
            raise ValueError("Model service identity changed: " + key)


def response_choice(payload):
    if payload.get("model") != SETTINGS["model"] or len(payload.get("choices", [])) != 1:
        raise ValueError("Unexpected provider model or choice count")
    choice = payload["choices"][0]
    message = choice["message"]
    if message.get("tool_calls") or not isinstance(message.get("content"), str):
        raise ValueError("Expected one json_action text response")
    # Truncation can only use the inherited deterministic outer-brace repair.
    return parse_action(message["content"])


def emit_receipt(online, receipt):
    directory = online.task_dir(TASK) / "events"
    existing = {read_json(p)["data"].get("request_key")
                for p in directory.glob("*.json") if read_json(p)["kind"] == "lm.request"}
    if receipt["key"] in existing:
        return
    online.emit(TASK, "lm.request", dict(request_key=receipt["key"], model_id=SETTINGS["model"],
        budget=1, condition=receipt["condition"], phase="selection", retry_index=0,
        status=receipt["outcome"], usage=receipt.get("usage", {}),
        measurement=dict(model_id=SETTINGS["model"], phase=receipt["condition"],
            request_started_at_utc=receipt["started_at_utc"],
            request_duration_seconds=receipt.get("request_duration_seconds")),
        accepted=int(receipt["outcome"] == "accepted"),
        raw_json_valid=int(receipt.get("choice", {}).get("raw_json_valid", False))))


def request_one(root, item, online, opener, endpoint):
    path = root / "receipts" / (item["key"] + ".json")
    response_path = root / "responses" / (item["key"] + ".json")
    receipt = read_json(path)
    if receipt is not None:
        if receipt["request_sha256"] != item["request_sha256"]:
            raise ValueError("Receipt request binding changed")
        if "outcome" in receipt:
            emit_receipt(online, receipt)
            return receipt
        if not response_path.is_file():
            raise RuntimeError("Prior request outcome is uncertain; never automatically resend " + item["key"])
    data = read_json(root / "inputs" / (item["key"] + ".json"))
    if digest(data["request"]) != item["request_sha256"]:
        raise ValueError("Prepared request changed")
    if receipt is None:
        online.ensure_online(TASK, timeout=180)
        ensure_health(root, opener, endpoint)
        receipt = dict(item, started_at_utc=now(), state="dispatched")
        atomic_json(path, receipt)
        started = time.monotonic()
        try:
            request = urllib.request.Request(endpoint + "/v1/chat/completions",
                data=canonical(data["request"]), headers={"Content-Type": "application/json"}, method="POST")
            try:
                with opener.open(request, timeout=180) as response:
                    status, raw = response.status, response.read(32769)
            except urllib.error.HTTPError as response:
                status, raw = response.code, response.read(32769)
            # Save the observed response durably before parsing. No network replay.
            atomic_json(response_path, dict(http_status=status, body_hex=raw.hex(),
                response_limit_exceeded=len(raw) > 32768,
                request_duration_seconds=time.monotonic() - started))
        except Exception as error:
            receipt.update(outcome="failed", state="recorded", error_class=type(error).__name__,
                request_duration_seconds=time.monotonic() - started, ended_at_utc=now(),
                usage={}, response_uncertain=True)
            atomic_json(path, receipt)
            emit_receipt(online, receipt)
            return receipt
    response = read_json(response_path)
    receipt.update(state="recorded", http_status=response["http_status"], ended_at_utc=now(),
        request_duration_seconds=response["request_duration_seconds"], usage={})
    try:
        if response["response_limit_exceeded"]:
            raise ValueError("Provider response too large")
        payload = json.loads(bytes.fromhex(response["body_hex"]).decode("utf-8"))
        receipt.update(usage=payload.get("usage", {}), server_request_id=payload.get("id"))
        if response["http_status"] != 200:
            raise ValueError("Provider rejected request")
        receipt.update(choice=response_choice(payload), outcome="accepted")
    except (ValueError, KeyError, TypeError) as error:
        receipt.update(outcome="failed", error_class=type(error).__name__)
    atomic_json(path, receipt)
    emit_receipt(online, receipt)
    return receipt


def build_report(root):
    manifest = verify_frozen(root)
    receipts = [read_json(root / "receipts" / (item["key"] + ".json")) for item in manifest["entries"]]
    if any(r is None or "outcome" not in r for r in receipts):
        raise ValueError("Do not evaluate an incomplete selection run")
    summary, rows = summarize(read_json(Path(manifest["score_table"])), receipts)
    summary["cost"] = dict(physical_requests=len(receipts), additional_requests=0,
        known_prompt_tokens=sum(r.get("usage", {}).get("prompt_tokens", 0) for r in receipts),
        known_completion_tokens=sum(r.get("usage", {}).get("completion_tokens", 0) for r in receipts),
        known_total_tokens=sum(r.get("usage", {}).get("total_tokens", 0) for r in receipts),
        unknown_usage_requests=sum(not all(type(r.get("usage", {}).get(k)) is int
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")) for r in receipts),
        request_seconds=sum(r.get("request_duration_seconds", 0) for r in receipts))
    freeze(root / "report/summary.json", summary)
    freeze(root / "report/tasks.json", rows)
    lines = ["# 同条件知识选模对照", "", "协议：" + PROTOCOL, "",
        "本轮零新增分类训练；以下为已见任务的探索性回放，不是独立泛化验证。", "",
        "|策略|有效任务数|平均平衡准确率|", "|---|---:|---:|"]
    names = dict(with_knowledge="有检索知识", without_knowledge="无检索知识", uniform_random="七模型均匀随机期望")
    for name, value in summary["methods"].items():
        score = value["mean_balanced_accuracy"]
        lines.append("|" + names.get(name, name) + "|" + str(value["tasks"]) + "|" +
                     (f"{score * 100:.4f}%" if score is not None else "无合法选择") + "|")
    pair = summary["paired"]
    delta = pair["mean_delta_percentage_points"]
    lines += ["", "配对任务数：" + str(pair["paired_tasks"]),
              "知识相对无知识的平均变化：" + (f"{delta:+.4f} 个百分点" if delta is not None else "不可计算"),
              f"胜/平/负：{pair['wins']}/{pair['ties']}/{pair['losses']}", "",
              "七种固定策略全部列出，不以本批测试结果挑出一个再称其为预先固定策略。",
              "无知识仅指不提供检索卡片，不代表大模型没有预训练知识。",
              "存在失败时须同时报告覆盖率；不同分母的均值不能视为完整配对比较。", "",
              "完整分类别、选择分布及成本见同目录 summary.json。"]
    (root / "report/report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run(root, endpoint):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "runner.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = verify_frozen(root)
        online = OnlineJournal(root)
        task = read_json(online.task_dir(TASK) / "task.json", {})
        if task.get("finished"):
            verified = online.wait_verified(TASK, timeout=300)
            atomic_json(root / "cloud-verification.json", verified)
            print("Existing completed run verified; no calls repeated.", flush=True)
            return
        online.start_task(TASK, manifest["public_config"])
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        receipts = []
        for item in manifest["entries"]:
            receipt = request_one(root, item, online, opener, endpoint)
            receipts.append(receipt)
            if receipt.get("http_status") in (400, 401, 403, 500) or receipt.get("response_uncertain"):
                raise RuntimeError("Provider failure recorded; no automatic retry: " + item["key"])
            status = dict(protocol=PROTOCOL, phase="selecting", completed=len(receipts), total=258,
                accepted=sum(r["outcome"] == "accepted" for r in receipts), classifier_fits=0,
                pid=os.getpid(), updated_at=now())
            atomic_json(root / "status.json", status)
            print(json.dumps(status, ensure_ascii=False), flush=True)
        online.ensure_online(TASK, timeout=180)
        summary = build_report(root)
        cloud_summary = dict(任务数=summary["tasks"], 物理选模请求数=len(receipts), 新增分类拟合数=0,
            有效配对任务数=summary["paired"]["paired_tasks"],
            知识组胜出任务数=summary["paired"]["wins"], 持平任务数=summary["paired"]["ties"],
            知识组下降任务数=summary["paired"]["losses"], 用量未知请求数=summary["cost"]["unknown_usage_requests"],
            已知总token数=summary["cost"]["known_total_tokens"])
        for name, value in summary["methods"].items():
            if value["mean_balanced_accuracy"] is not None:
                cloud_summary[name + "_平均平衡准确率百分比"] = 100 * value["mean_balanced_accuracy"]
        delta = summary["paired"]["mean_delta_percentage_points"]
        if delta is not None:
            cloud_summary["知识相对无知识_平均变化百分点"] = delta
        for suite, pair in summary["by_suite"].items():
            if pair["mean_delta_percentage_points"] is not None:
                cloud_summary[suite + "_知识增益百分点"] = pair["mean_delta_percentage_points"]
        online.finish_task(TASK, cloud_summary)
        verified = online.wait_verified(TASK, timeout=300)
        atomic_json(root / "cloud-verification.json", verified)
        atomic_json(root / "status.json", dict(phase="completed", completed=258,
            accepted=sum(r["outcome"] == "accepted" for r in receipts), classifier_fits=0,
            cloud_ref=verified["cloud_ref"], updated_at=now()))
        print(json.dumps(dict(completed=258, cloud_ref=verified["cloud_ref"]), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "report"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=REPO / "work/knowledge-rerun-20261014")
    parser.add_argument("--endpoint", default="http://127.0.0.1:18101")
    args = parser.parse_args()
    root = args.root.resolve()
    if REPO / "work" not in root.parents:
        raise ValueError("Run outputs must remain under Pan/work")
    if args.endpoint not in ("http://127.0.0.1:18101", "http://localhost:18101"):
        raise ValueError("Use the registered local Qwen3.5-9B service")
    if args.action == "prepare":
        prepare(root, args.source)
    elif args.action == "run":
        try:
            run(root, args.endpoint)
        except BaseException as error:
            atomic_json(root / "status.json", dict(phase="needs_attention",
                error_class=type(error).__name__, message=str(error), updated_at=now()))
            raise
    else:
        summary = build_report(root)
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

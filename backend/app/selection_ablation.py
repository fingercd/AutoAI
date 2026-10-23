"""Selection-only ablation over frozen requests; never trains a classifier."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math

PROTOCOL = "step1-selection-ablation-v1"
MODELS = ("pls_da", "pca_lda", "logistic_regression", "svm",
          "random_forest", "xgboost", "cnn1d")
CONDITIONS = ("with_knowledge", "without_knowledge")
SETTINGS = dict(model="qwen35_9b", temperature=0, top_p=1, max_tokens=1024,
                stream=False, chat_template_kwargs={"enable_thinking": False})
FORBIDDEN = {"test", "test_scores", "test_metrics", "validation", "metrics",
             "prediction_rows", "run_id", "artifact_directory", "task_id",
             "task_name", "family_id", "record_path", "dataset_sha256", "winner"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError("Non-finite JSON number")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)


def check_input(value):
    if isinstance(value, dict):
        if FORBIDDEN.intersection(value):
            raise ValueError("Result or task-identity field in selector input")
        for item in value.values():
            check_input(item)
    elif isinstance(value, list):
        for item in value:
            check_input(item)


def make_request(frozen, condition):
    """Only retrieved_knowledge changes; settings, system prompt and order stay."""
    if condition not in CONDITIONS:
        raise ValueError("Unknown condition")
    request = deepcopy(frozen["request"])
    if any(request.get(key) != value for key, value in SETTINGS.items()):
        raise ValueError("Frozen generation settings do not match approved protocol")
    if len(request["messages"]) != 2 or [m["role"] for m in request["messages"]] != ["system", "user"]:
        raise ValueError("Expected frozen system/user pair")
    semantic = strict_json(request["messages"][1]["content"])
    if semantic != frozen["semantic_input"]:
        raise ValueError("Frozen semantic input does not match actual request")
    if len(semantic["candidates"]) != len(MODELS) or {item["model_id"] for item in semantic["candidates"]} != set(MODELS):
        raise ValueError("Candidate order differs from the approved seven-model menu")
    check_input(semantic)
    if condition == "without_knowledge":
        semantic["retrieved_knowledge"] = []
        request["messages"][1]["content"] = json.dumps(semantic, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return request


def parse_action(content):
    """Preserve the prior deterministic single outer-brace repair, no LM retry."""
    operations = []
    try:
        action = strict_json(content)
    except (ValueError, TypeError):
        if not isinstance(content, str):
            raise ValueError("Expected text response")
        action = strict_json(content + "}")
        operations.append("append_final_object_brace")
    if not isinstance(action, dict) or set(action) != {"tool_name", "arguments", "rationale"}:
        raise ValueError("Invalid action fields")
    args = action["arguments"]
    if action["tool_name"] != "submit_ml_experiment" or not isinstance(args, dict) or set(args) != {"model_id"}:
        raise ValueError("Invalid action or argument fields")
    if not isinstance(args["model_id"], str) or args["model_id"] not in MODELS:
        raise ValueError("Invalid model ID")
    rationale = action["rationale"]
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 1000:
        raise ValueError("Invalid rationale")
    if rationale != rationale.strip():
        operations.append("trim_rationale_edge_whitespace")
    return dict(model_id=args["model_id"], rationale=rationale.strip(),
                raw_json_valid="append_final_object_brace" not in operations,
                format_operations=operations)


def validate_scores(rows):
    keys = [(r["suite"], r["task_id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate task in score table")
    for row in rows:
        if set(row["models"]) != set(MODELS):
            raise ValueError("Incomplete seven-model score table")
        for model in MODELS:
            metrics = row["models"][model]["metrics"]
            for metric in ("test_balanced_accuracy", "test_accuracy", "test_macro_f1"):
                value = metrics[metric]
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Invalid frozen classifier score")
    return {key: row for key, row in zip(keys, rows)}


def summarize(scores, receipts):
    """Evaluator only: called after all selection requests have been recorded."""
    table = validate_scores(scores)
    seen = set()
    choices = {}
    for receipt in receipts:
        key = (receipt["suite"], receipt["task_id"], receipt["condition"])
        if key in seen or key[:2] not in table or key[2] not in CONDITIONS:
            raise ValueError("Duplicate or unbound receipt")
        seen.add(key)
        if receipt["outcome"] == "accepted":
            model = receipt["choice"]["model_id"]
            if model not in MODELS:
                raise ValueError("Invalid recorded choice")
            choices[key] = model
    task_rows = []
    methods = {name: [] for name in (*CONDITIONS, "uniform_random", *("fixed_" + m for m in MODELS))}
    distributions = {condition: Counter() for condition in CONDITIONS}
    for key, row in table.items():
        values = {m: row["models"][m]["metrics"]["test_balanced_accuracy"] for m in MODELS}
        item = dict(suite=key[0], task_id=key[1], family_id=row["family_id"])
        for condition in CONDITIONS:
            model = choices.get((*key, condition))
            score = values[model] if model else None
            item[condition] = dict(model_id=model, balanced_accuracy=score)
            if score is not None:
                methods[condition].append(score)
                distributions[condition][model] += 1
        item["uniform_random"] = sum(values.values()) / len(MODELS)
        methods["uniform_random"].append(item["uniform_random"])
        for model, value in values.items():
            methods["fixed_" + model].append(value)
        task_rows.append(item)
    def paired(rows):
        diffs = [r["with_knowledge"]["balanced_accuracy"] - r["without_knowledge"]["balanced_accuracy"]
                 for r in rows if all(r[c]["balanced_accuracy"] is not None for c in CONDITIONS)]
        return dict(tasks=len(rows), paired_tasks=len(diffs),
                    mean_delta_percentage_points=100 * sum(diffs) / len(diffs) if diffs else None,
                    wins=sum(d > 1e-12 for d in diffs),
                    ties=sum(abs(d) <= 1e-12 for d in diffs),
                    losses=sum(d < -1e-12 for d in diffs))
    summary = dict(protocol=PROTOCOL, tasks=len(scores), request_receipts=len(receipts),
        new_classifier_fits=0, primary_comparison_complete=len(choices) == 2 * len(scores),
        methods={name: dict(tasks=len(values),
            mean_balanced_accuracy=sum(values) / len(values) if values else None)
            for name, values in methods.items()},
        paired=paired(task_rows),
        by_suite={suite: paired([r for r in task_rows if r["suite"] == suite])
                  for suite in sorted({r["suite"] for r in task_rows})},
        selection_counts={c: dict(distributions[c]) for c in CONDITIONS},
        failures={c: sum(r["condition"] == c and r["outcome"] != "accepted" for r in receipts)
                  for c in CONDITIONS},
        interpretation="Exploratory reuse of tasks seen during knowledge construction; not independent generalization. Fixed strategies are all reported, not selected on this test matrix.")
    return summary, task_rows

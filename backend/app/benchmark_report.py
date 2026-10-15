"""Offline benchmark aggregation for fixed classification evaluations.

Declare expected rounds on tasks as round IDs or objects like
{"round_id": "r0", "eligible_models": {"1": ["m1", "m2"], "3": ["m1"]}}.
A task-level eligible_models list applies to both budgets. Records may also carry
eligible_models for their task/round/budget. Without a declaration, observed model
IDs define the set; declare eligibility to detect an entirely missing planned row.
This module does not train models or call an LLM.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_METRICS = ("balanced_accuracy", "accuracy", "macro_f1")
_BUDGETS = (1, 3)
_METHODS = {
    "R0": ("random", 1), "R1": ("random", 3),
    "A0": ("agent_choice", 1), "A1": ("agent_choice", 3),
    "F0": ("development_fixed", 1), "F1": ("development_fixed", 3),
    "P0": ("validation_best", 1), "P1": ("validation_best", 3),
}
_PAIRS = (
    ("A0", "R0"), ("A1", "R1"),
    ("F0", "R0"), ("F0", "A0"), ("F0", "P0"),
    ("F1", "R1"), ("F1", "A1"), ("F1", "P1"),
)
_COST_FIELDS = {
    "actual_fits": ("actual_fits",),
    "fit_duration_sum_seconds": ("wall_seconds", "wall_time_seconds"),
    "driver_wall_duration_sum_seconds": ("driver_wall_seconds",),
    "actual_cached_fits": ("actual_cached_fits",),
    "actual_cached_seconds": ("actual_cached_seconds",),
    "uncached_estimated_fits": ("uncached_estimated_fits",),
    "uncached_estimated_seconds": ("uncached_estimated_seconds",),
    "known_cached_source_fit_seconds": ("known_cached_source_fit_seconds",),
    "known_lm_tokens": ("known_lm_tokens", "knownLMtokens"),
    "unknown_count": ("unknown_count", "unknown_lm_count"),
    "cached_lm_tokens_estimate": ("cached_lm_tokens_estimate", "cached_tokens_estimate", "cached_estimate"),
}


def _id(value: Any, field: str) -> str:
    if value is None:
        raise ValueError(f"benchmark_invalid_{field}")
    result = str(value)
    if not result:
        raise ValueError(f"benchmark_empty_{field}")
    return result


def _budget(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("benchmark_invalid_budget")
    try:
        result = int(value)
        if float(value) != result:
            raise ValueError("benchmark_invalid_budget")
    except (TypeError, ValueError) as exc:
        raise ValueError("benchmark_invalid_budget") from exc
    if result not in _BUDGETS:
        raise ValueError("benchmark_invalid_budget")
    return result


def _number(value: Any, code: str) -> float:
    if isinstance(value, bool):
        raise ValueError(code)
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(code) from exc
    if not math.isfinite(value):
        raise ValueError(code)
    return value


def _str_list(value: Any, code: str) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(code)
    result = [_id(item, code) for item in value]
    if len(result) != len(set(result)):
        raise ValueError(code)
    return result


def _eligibility(value: Any) -> dict[int, list[str]]:
    if isinstance(value, Mapping):
        parsed = {}
        for budget in _BUDGETS:
            for key in (budget, str(budget)):
                if key in value:
                    parsed[budget] = _str_list(value[key], "benchmark_invalid_eligible_models")
                    break
        if not parsed:
            raise ValueError("benchmark_invalid_eligibility")
        return parsed
    models = _str_list(value, "benchmark_invalid_eligible_models")
    return {budget: list(models) for budget in _BUDGETS}


def _normalise_tasks(tasks: list[dict]) -> tuple[dict, dict]:
    if not isinstance(tasks, list):
        raise ValueError("benchmark_tasks_must_be_list")
    task_map, declared = {}, {}
    for raw in tasks:
        if not isinstance(raw, Mapping):
            raise ValueError("benchmark_invalid_task")
        task_id = _id(raw.get("task_id"), "task_id")
        if task_id in task_map:
            raise ValueError("benchmark_duplicate_task")
        role = _id(raw.get("role"), "role")
        if role not in {"development", "locked", "excluded"}:
            raise ValueError("benchmark_invalid_task_role")
        labels = None if raw.get("labels") is None else _str_list(
            raw["labels"], "benchmark_invalid_labels"
        )
        if labels == []:
            raise ValueError("benchmark_invalid_labels")
        task = {
            "task_id": task_id,
            "suite": _id(raw.get("suite"), "suite"),
            "family_id": _id(raw.get("family_id"), "family_id"),
            "role": role,
            "labels": labels,
            "round_ids": [],
        }
        rounds = raw.get("rounds") or raw.get("expected_rounds") or raw.get("round_ids") or []
        if not isinstance(rounds, (list, tuple)):
            raise ValueError("benchmark_invalid_rounds")
        for item in rounds:
            if isinstance(item, Mapping):
                round_id = _id(item.get("round_id"), "round_id")
                task["round_ids"].append(round_id)
                elig = item.get("eligible_models_by_budget", item.get(
                    "eligible_models", item.get("eligibility")
                ))
                if elig is not None:
                    for budget, models in _eligibility(elig).items():
                        key = (task_id, round_id, budget)
                        if key in declared and declared[key] != models:
                            raise ValueError("benchmark_conflicting_eligibility")
                        declared[key] = models
            else:
                task["round_ids"].append(_id(item, "round_id"))
        if len(task["round_ids"]) != len(set(task["round_ids"])):
            raise ValueError("benchmark_duplicate_round")
        by_round = raw.get("eligibility_by_round") or {}
        if not isinstance(by_round, Mapping):
            raise ValueError("benchmark_invalid_eligibility")
        for round_id, value in by_round.items():
            round_id = _id(round_id, "round_id")
            task["round_ids"].append(round_id)
            for budget, models in _eligibility(value).items():
                key = (task_id, round_id, budget)
                if key in declared and declared[key] != models:
                    raise ValueError("benchmark_conflicting_eligibility")
                declared[key] = models
        task["round_ids"] = sorted(set(task["round_ids"]))
        task_map[task_id] = task
    return task_map, declared


def _normalise_rows(value: Any, task_id: str, round_id: str, model_id: str) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("benchmark_invalid_prediction_rows")
    rows, seen = [], set()
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError("benchmark_invalid_prediction_row")
        sample_id = _id(raw.get("sample_id"), "sample_id")
        if sample_id in seen:
            raise ValueError("benchmark_duplicate_sample_in_record")
        seen.add(sample_id)
        probabilities = raw.get("probabilities")
        if not isinstance(probabilities, Mapping):
            raise ValueError("benchmark_invalid_probabilities")
        probabilities = {
            _id(label, "probability_label"): _number(prob, "benchmark_invalid_probability")
            for label, prob in probabilities.items()
        }
        if any(prob < 0 or prob > 1 for prob in probabilities.values()):
            raise ValueError("benchmark_invalid_probability")
        rows.append({
            "sample_id": sample_id,
            "true_label": _id(raw.get("true_label"), "label"),
            "predicted_label": _id(raw.get("predicted_label"), "label"),
            "probabilities": probabilities,
            "task_id": task_id,
            "round_id": round_id,
            "model_id": model_id,
        })
    return rows


def _normalise_records(records: list[dict], tasks: dict) -> tuple[dict, dict, list]:
    if not isinstance(records, list):
        raise ValueError("benchmark_records_must_be_list")
    record_map, record_elig, failures = {}, {}, []
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ValueError("benchmark_invalid_record")
        task_id = _id(raw.get("task_id"), "task_id")
        if task_id not in tasks:
            raise ValueError("benchmark_unknown_task")
        round_id, model_id = _id(raw.get("round_id"), "round_id"), _id(raw.get("model_id"), "model_id")
        budget = _budget(raw.get("budget"))
        status = _id(raw.get("status"), "status").lower()
        if status not in {"completed", "failed"}:
            raise ValueError("benchmark_invalid_record_status")
        key = (task_id, round_id, budget, model_id)
        if key in record_map:
            raise ValueError("benchmark_duplicate_record")
        rows = _normalise_rows(raw.get("prediction_rows", []), task_id, round_id, model_id) if status == "completed" else []
        validation = _number(raw.get("validation_score"), "benchmark_invalid_validation_score") if status == "completed" else None
        cost = raw.get("cost") if isinstance(raw.get("cost"), Mapping) else {}
        record_map[key] = {
            "task_id": task_id, "round_id": round_id, "budget": budget,
            "model_id": model_id, "status": status, "validation_score": validation,
            "prediction_rows": rows, "failure_reason": str(raw.get("failure_reason") or ""),
            "cost": cost,
        }
        eligibility = raw.get("eligible_models_by_budget", raw.get("eligible_models", raw.get("planned_models")))
        if eligibility is not None:
            parsed = _eligibility(eligibility)
            if budget in parsed:
                elig_key = (task_id, round_id, budget)
                if elig_key in record_elig and record_elig[elig_key] != parsed[budget]:
                    raise ValueError("benchmark_conflicting_eligibility")
                record_elig[elig_key] = parsed[budget]
        if status == "failed":
            failures.append({
                "task_id": task_id, "round_id": round_id, "budget": budget,
                "model_id": model_id, "status": status,
                "failure_reason": record_map[key]["failure_reason"],
            })
    return record_map, record_elig, failures


def _normalise_choices(choices: list[dict], tasks: dict) -> tuple[dict, list]:
    if not isinstance(choices, list):
        raise ValueError("benchmark_choices_must_be_list")
    choice_map, invalid = {}, []
    duplicate_keys = set()
    for raw in choices:
        if not isinstance(raw, Mapping):
            raise ValueError("benchmark_invalid_choice")
        task_id = _id(raw.get("task_id"), "task_id")
        if task_id not in tasks:
            raise ValueError("benchmark_unknown_choice_task")
        round_id, budget = _id(raw.get("round_id"), "round_id"), _budget(raw.get("budget"))
        model_id = _id(raw.get("model_id"), "model_id")
        key = (task_id, round_id, budget)
        if key in choice_map or key in duplicate_keys:
            choice_map.pop(key, None)
            duplicate_keys.add(key)
            invalid.append({
                "task_id": task_id, "round_id": round_id, "budget": budget,
                "model_id": model_id, "reason": "duplicate_choice",
            })
        else:
            choice_map[key] = model_id
    return choice_map, invalid


def _resolve_eligible(task_id: str, round_id: str, budget: int, declared: dict,
                      record_elig: dict, records: dict, choices: dict) -> list[str]:
    key = (task_id, round_id, budget)
    task_models, record_models = declared.get(key), record_elig.get(key)
    if task_models is not None and record_models is not None and task_models != record_models:
        raise ValueError("benchmark_conflicting_eligibility")
    models = task_models if task_models is not None else record_models
    observed = sorted(k[3] for k in records if k[:3] == key)
    if models is not None:
        if set(observed) - set(models):
            raise ValueError("benchmark_unexpected_model_record")
        return list(models)
    if observed:
        return observed
    return [choices[key]] if key in choices else []


def _labels_and_validate_samples(tasks: dict, records: dict) -> None:
    labels_seen = defaultdict(set)
    for record in records.values():
        if record["status"] != "completed":
            continue
        for row in record["prediction_rows"]:
            labels_seen[record["task_id"]].update(
                (row["true_label"], row["predicted_label"], *row["probabilities"].keys())
            )
    for task_id, task in tasks.items():
        if task["labels"] is None:
            task["labels"] = sorted(labels_seen[task_id])
        allowed = set(task["labels"])
        if labels_seen[task_id] - allowed:
            raise ValueError("benchmark_label_outside_full_labels")

    rows_by_round, sample_round, sample_true = defaultdict(list), {}, {}
    for record in records.values():
        if record["status"] != "completed" or not record["prediction_rows"]:
            continue
        rows_by_round[(record["task_id"], record["round_id"])].append(record)
        for row in record["prediction_rows"]:
            sample_key = (record["task_id"], row["sample_id"])
            if sample_key in sample_true and sample_true[sample_key] != row["true_label"]:
                raise ValueError("benchmark_inconsistent_sample_label")
            sample_true[sample_key] = row["true_label"]
            round_key = sample_key
            if round_key in sample_round and sample_round[round_key] != record["round_id"]:
                raise ValueError("benchmark_sample_retested_across_rounds")
            sample_round[round_key] = record["round_id"]
    for values in rows_by_round.values():
        signatures = {
            tuple(sorted((row["sample_id"], row["true_label"]) for row in record["prediction_rows"]))
            for record in values
        }
        if len(signatures) != 1:
            raise ValueError("benchmark_models_do_not_share_test_samples")


def _confusion(rows: Sequence[dict], labels: Sequence[str]) -> np.ndarray:
    indexes = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for row in rows:
        matrix[indexes[row["true_label"]], indexes[row["predicted_label"]]] += 1
    return matrix


def _metrics(matrix: np.ndarray) -> dict:
    n = matrix.shape[0]
    if n == 0 or matrix.sum() == 0:
        return {metric: None for metric in _METRICS}
    support, predicted, tp = matrix.sum(axis=1), matrix.sum(axis=0), np.diag(matrix)
    recall = np.divide(tp, support, out=np.zeros(n), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(n), where=(support + predicted) > 0)
    return {
        "balanced_accuracy": float(recall.mean()),
        "accuracy": float(tp.sum() / matrix.sum()),
        "macro_f1": float(f1.mean()),
    }


def _model_score(rounds: Sequence[dict], model: str, labels: Sequence[str]) -> dict:
    matrix = sum((item["confusions"][model] for item in rounds), np.zeros((len(labels), len(labels))))
    return _metrics(matrix)


def _selected_score(rounds: Sequence[dict], models: Sequence[str], labels: Sequence[str]) -> dict:
    matrix = sum((item["confusions"][model] for item, model in zip(rounds, models)),
                 np.zeros((len(labels), len(labels))))
    return _metrics(matrix)


def _seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join((str(seed), *parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _random_score(rounds: Sequence[dict], labels: Sequence[str], task: str,
                  budget: int, seed: int, draws: int) -> dict:
    model_matrices = []
    expected = np.zeros((len(labels), len(labels)))
    for item in rounds:
        models = sorted(item["eligible_models"])
        matrices = np.stack([item["confusions"][model] for model in models])
        model_matrices.append(matrices)
        expected += matrices.mean(axis=0)
    result = _metrics(expected)
    if len(rounds) == 1:
        result["macro_f1"] = statistics.fmean(_metrics(matrix)["macro_f1"] or 0
                                                for matrix in model_matrices[0])
    else:
        rng = np.random.default_rng(_seed(seed, task, str(budget), "random-macro-f1"))
        selections = [rng.integers(0, len(matrices), size=draws) for matrices in model_matrices]
        # Macro-F1 only needs class marginals and TP, not a draws-by-C-by-C tensor.
        tp = np.zeros((draws, len(labels)))
        support = np.zeros_like(tp)
        predicted = np.zeros_like(tp)
        for matrices, selected in zip(model_matrices, selections):
            tp += np.diagonal(matrices, axis1=1, axis2=2)[selected]
            support += matrices.sum(axis=2)[selected]
            predicted += matrices.sum(axis=1)[selected]
        denom = support + predicted
        f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
        result["macro_f1"] = float(f1.mean(axis=1).mean())
    return result


def _coverage(task: dict, budget: int, round_ids: Sequence[str], records: dict,
              declared: dict, record_elig: dict, choices: dict) -> dict:
    details, rounds, reasons = [], [], []
    if not round_ids:
        reasons.append("no_rounds")
    for round_id in round_ids:
        eligible = _resolve_eligible(task["task_id"], round_id, budget,
                                     declared, record_elig, records, choices)
        selected, missing, failed, empty = {}, [], [], []
        for model in eligible:
            record = records.get((task["task_id"], round_id, budget, model))
            if record is None:
                missing.append(model)
            elif record["status"] == "failed":
                failed.append({"model_id": model, "failure_reason": record["failure_reason"]})
            elif not record["prediction_rows"]:
                empty.append(model)
            else:
                selected[model] = record
        codes = []
        if not eligible:
            codes.append("no_eligible_models")
        if missing:
            codes.append("missing_planned_models")
        if failed:
            codes.append("failed_models")
        if empty:
            codes.append("empty_predictions")
        complete = bool(eligible) and not codes
        rounds.append({
            "round_id": round_id, "complete": complete, "eligible_models": eligible,
            "missing_models": missing, "failed_models": failed,
            "empty_prediction_models": empty, "reason_codes": codes,
        })
        if not complete:
            reasons.extend(f"{round_id}:{code}" for code in codes)
            continue
        reference_rows = next(iter(selected.values()))["prediction_rows"]
        details.append({
            "round_id": round_id, "eligible_models": eligible, "records": selected,
            "rows": reference_rows,
            "confusions": {
                model: _confusion(record["prediction_rows"], task["labels"])
                for model, record in selected.items()
            },
        })
    complete = bool(round_ids) and len(details) == len(round_ids)
    if not complete and not reasons:
        reasons.append("incomplete_round_matrix")
    return {
        "coverage": {
            "task_id": task["task_id"], "role": task["role"], "budget": budget, "complete": complete,
            "round_count": len(round_ids), "complete_round_count": len(details),
            "reason_codes": reasons, "rounds": rounds,
        },
        "round_infos": details,
    }


def _fixed_models(tasks: dict, coverage: dict) -> dict:
    """Choose F from development scores and the shared legal-model intersection.

    Locked tasks contribute only declared eligible-model IDs here. Their Test
    predictions and scores do not affect whether a fixed model is selected.
    """
    locked = [t for t in tasks.values() if t["role"] == "locked"]
    development = [t for t in tasks.values() if t["role"] == "development"]
    result = {}
    for budget in _BUDGETS:
        item = {"budget": budget, "available": False, "model_id": None,
                "development_mean_balanced_accuracy": None, "candidate_models": [], "reason": None}
        if not development:
            item["reason"] = "no_development_tasks"
            result[str(budget)] = item
            continue
        if not locked:
            item["reason"] = "no_locked_tasks"
            result[str(budget)] = item
            continue
        incomplete_development = [
            task["task_id"] for task in development
            if not coverage[(task["task_id"], budget)]["coverage"]["complete"]
        ]
        if incomplete_development:
            item["reason"] = "development_task_matrix_incomplete"
            item["unavailable_development_tasks"] = incomplete_development
            result[str(budget)] = item
            continue

        all_rounds = []
        missing_locked_eligibility = []
        for task in (*development, *locked):
            round_rows = coverage[(task["task_id"], budget)]["coverage"]["rounds"]
            if not round_rows:
                if task["role"] == "locked":
                    missing_locked_eligibility.append(task["task_id"])
                continue
            for round_row in round_rows:
                eligible = set(round_row["eligible_models"])
                if task["role"] == "locked" and not eligible:
                    missing_locked_eligibility.append(f"{task['task_id']}:{round_row['round_id']}")
                all_rounds.append((task, round_row, eligible))
        if missing_locked_eligibility:
            item["reason"] = "locked_round_eligibility_missing"
            item["unavailable_locked_rounds"] = missing_locked_eligibility
            result[str(budget)] = item
            continue

        common: set[str] | None = None
        for _, _, eligible in all_rounds:
            common = eligible if common is None else common & eligible
        candidates = sorted(common or set())
        item["candidate_models"] = candidates
        if not candidates:
            item["reason"] = "no_common_complete_model"
            result[str(budget)] = item
            continue

        means = {
            model: statistics.fmean(
                _model_score(coverage[(task["task_id"], budget)]["round_infos"],
                             model, task["labels"])["balanced_accuracy"]
                for task in development
            )
            for model in candidates
        }
        chosen = sorted(candidates, key=lambda model: (-means[model], model))[0]
        item.update({
            "available": True,
            "model_id": chosen,
            "development_mean_balanced_accuracy": means[chosen],
            "candidate_development_means": means,
        })
        result[str(budget)] = item
    return result

def _method_task_scores(rounds: Sequence[dict], task: dict, budget: int,
                        choices: dict, fixed: str | None, seed: int, draws: int) -> tuple[dict, dict]:
    selected, reasons = {"A": [], "F": [], "P": []}, {"A": [], "F": [], "P": []}
    for item in rounds:
        round_id, eligible = item["round_id"], sorted(item["eligible_models"])
        choice = choices.get((task["task_id"], round_id, budget))
        if choice is None:
            reasons["A"].append(f"{round_id}:missing_choice")
        elif choice not in eligible:
            reasons["A"].append(f"{round_id}:invalid_choice")
        else:
            selected["A"].append(choice)
        if fixed is None:
            reasons["F"].append("fixed_model_unavailable")
        elif fixed not in eligible:
            reasons["F"].append(f"{round_id}:fixed_model_ineligible")
        else:
            selected["F"].append(fixed)
        picked = sorted(eligible, key=lambda model: (
            -item["records"][model]["validation_score"], model
        ))
        if picked:
            selected["P"].append(picked[0])
        else:
            reasons["P"].append(f"{round_id}:no_validation_candidate")
    scores = {"R": _random_score(rounds, task["labels"], task["task_id"], budget, seed, draws)}
    for name in ("A", "F", "P"):
        if not reasons[name] and len(selected[name]) == len(rounds):
            scores[name] = _selected_score(rounds, selected[name], task["labels"])
    return scores, reasons


def _mean_metrics(rows: Sequence[dict]) -> dict:
    return {
        metric: statistics.fmean(row[metric] for row in rows if row.get(metric) is not None)
        if any(row.get(metric) is not None for row in rows) else None
        for metric in _METRICS
    }


def _group_summary(rows: Sequence[dict], tasks: Sequence[dict], dimension: str) -> dict:
    task_by_id = {task["task_id"]: task for task in tasks}
    output = {}
    for group in sorted({task[dimension] for task in tasks}):
        grouped = [row for row in rows if task_by_id[row["task_id"]][dimension] == group]
        output[group] = {
            "expected_task_count": sum(task[dimension] == group for task in tasks),
            "task_count": len(grouped),
            "metrics": _mean_metrics(grouped),
        }
    return output


def _rank(values: list[tuple[str, float]]) -> dict[str, float]:
    values.sort(key=lambda x: x[1], reverse=True)
    result, start = {}, 0
    while start < len(values):
        end = start + 1
        while end < len(values) and abs(values[end][1] - values[start][1]) <= 1e-12:
            end += 1
        rank = ((start + 1) + end) / 2
        for method, _ in values[start:end]:
            result[method] = rank
        start = end
    return result


def _average_ranks(rows: Sequence[dict], locked: Sequence[dict]) -> dict:
    indexed = {(row["task_id"], row["budget"]): row for row in rows}
    result = {}
    for budget in _BUDGETS:
        methods = [method for method, (_, b) in _METHODS.items() if b == budget]
        result[str(budget)] = {}
        for metric in _METRICS:
            ranks = defaultdict(list)
            for task in locked:
                row = indexed.get((task["task_id"], budget))
                if row is None:
                    continue
                for method, value in _rank([
                    (m, row[m][metric]) for m in methods
                    if row.get(m) is not None and row[m].get(metric) is not None
                ]).items():
                    ranks[method].append(value)
            result[str(budget)][metric] = {
                method: {"average_rank": statistics.fmean(ranks[method]) if ranks[method] else None,
                         "task_count": len(ranks[method])}
                for method in methods
            }
    return result


def _bootstrap_ci(values: list[dict], seed: int, draws: int,
                  comparison: str, metric: str) -> tuple[float | None, float | None, int]:
    if not values:
        return None, None, 0
    families = defaultdict(list)
    for row in values:
        families[row["family_id"]].append(row["difference"])
    ids = sorted(families)
    rng = np.random.default_rng(_seed(seed, comparison, metric, "family-bootstrap"))
    sampled = rng.integers(0, len(ids), size=(draws, len(ids)))
    estimates = np.empty(draws)
    for i, choice in enumerate(sampled):
        cluster_values = [v for index in choice for v in families[ids[int(index)]]]
        estimates[i] = np.mean(cluster_values)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high), len(ids)


def _paired(rows: Sequence[dict], locked: Sequence[dict], seed: int, draws: int) -> list[dict]:
    indexed = {(row["task_id"], row["budget"]): row for row in rows}
    output = []
    for left, right in _PAIRS:
        budget = _METHODS[left][1]
        for metric in _METRICS:
            values = []
            for task in locked:
                row = indexed.get((task["task_id"], budget))
                if row is None or left not in row or right not in row:
                    continue
                delta = row[left][metric] - row[right][metric]
                values.append({"task_id": task["task_id"], "family_id": task["family_id"],
                               "difference": float(delta)})
            low, high, family_count = _bootstrap_ci(values, seed, draws, f"{left}-{right}", metric)
            differences = [v["difference"] for v in values]
            output.append({
                "comparison": f"{left}-{right}", "left_method": left, "right_method": right,
                "budget": budget, "metric": metric, "task_count": len(values),
                "paired_task_ids": sorted(v["task_id"] for v in values),
                "family_count": family_count,
                "mean_difference": statistics.fmean(differences) if differences else None,
                "confidence_interval_95": {
                    "lower": low, "upper": high,
                    "method": "family_cluster_percentile_bootstrap", "draws": draws,
                },
                "win_count": sum(v > 1e-12 for v in differences),
                "tie_count": sum(abs(v) <= 1e-12 for v in differences),
                "loss_count": sum(v < -1e-12 for v in differences),
            })
    return output


def _cost_summaries(records: dict, tasks: dict) -> dict:
    groups = defaultdict(list)
    for record in records.values():
        groups[(tasks[record["task_id"]]["role"], record["budget"])].append(record)
    output = {}
    for (role, budget), values in sorted(groups.items()):
        fields = {}
        for field, aliases in _COST_FIELDS.items():
            numbers = []
            for record in values:
                raw = next((record["cost"][name] for name in aliases if name in record["cost"]), None)
                try:
                    number = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    numbers.append(number)
            fields[field] = {
                "sum": float(sum(numbers)) if numbers else None,
                "known_record_count": len(numbers),
                "unknown_record_count": len(values) - len(numbers),
            }
        output.setdefault(role, {})[str(budget)] = {
            "record_count": len(values), "provided_costs_only": True, "fields": fields,
        }
    return output


def _run_cost_details(records: dict) -> list[dict]:
    fields = (
        "actual_fits", "wall_seconds", "driver_wall_seconds",
        "actual_cached_fits", "actual_cached_seconds",
        "uncached_estimated_fits", "uncached_estimated_seconds",
        "known_cached_source_fit_seconds", "known_lm_tokens", "unknown_count",
        "cached_lm_tokens_estimate",
    )
    result = []
    for key, record in sorted(records.items()):
        cost = record["cost"]
        result.append({
            "task_id": record["task_id"], "round_id": record["round_id"],
            "model_id": record["model_id"], "budget": record["budget"],
            "status": record["status"], "run_id": record.get("run_id"),
            "fit_duration_seconds": cost.get("wall_seconds"),
            "cost_fields": {field: cost[field] for field in fields if field in cost},
            "uncached_estimate_scope": cost.get("uncached_estimate_scope"),
        })
    return result


def _task_wall_time(driver_state: Mapping | None, tasks: Mapping[str, dict]) -> dict:
    if driver_state is not None and not isinstance(driver_state, Mapping):
        raise ValueError("benchmark_invalid_driver_state")
    state = driver_state or {}
    by_task = dict(state.get("task_wall_seconds") or {})
    clocks: dict[str, dict] = {}
    for clock in state.get("task_clocks") or []:
        if not isinstance(clock, Mapping) or not clock.get("task_id"):
            continue
        task_id = str(clock["task_id"])
        if task_id in clocks and clocks[task_id].get("task_wall_seconds") != clock.get("task_wall_seconds"):
            raise ValueError("benchmark_conflicting_task_clocks")
        clocks[task_id] = dict(clock)
        if task_id not in by_task and clock.get("task_wall_seconds") is not None:
            by_task[task_id] = clock["task_wall_seconds"]
    rows = []
    role_values: dict[str, list[float]] = defaultdict(list)
    for task_id, task in tasks.items():
        value = by_task.get(task_id)
        if value is not None:
            value = _number(value, "benchmark_invalid_task_wall_seconds")
            role_values[task["role"]].append(value)
        clock = clocks.get(task_id, {})
        rows.append({
            "task_id": task_id, "suite": task["suite"], "family_id": task["family_id"],
            "role": task["role"], "task_wall_seconds": value,
            "cloud_verified_at": clock.get("cloud_verified_at"),
        })
    return {
        "available": driver_state is not None,
        "source": "driver_state.task_wall_seconds or driver_state.task_clocks[].task_wall_seconds",
        "role_sums": {
            role: {"sum": sum(values), "known_task_count": len(values)}
            for role, values in sorted(role_values.items())
        },
        "tasks": rows,
    }


def summarize(records: list[dict], choices: list[dict], tasks: list[dict],
              seed: int = 42, random_draws: int = 10000,
              bootstrap_draws: int = 2000,
              driver_state: dict | None = None) -> dict:
    """Aggregate task-level scores, selections, coverage, and paired uncertainty."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("benchmark_invalid_seed")
    if isinstance(random_draws, bool) or not isinstance(random_draws, int) or random_draws <= 0:
        raise ValueError("benchmark_invalid_random_draws")
    if isinstance(bootstrap_draws, bool) or not isinstance(bootstrap_draws, int) or bootstrap_draws <= 0:
        raise ValueError("benchmark_invalid_bootstrap_draws")
    task_map, declared = _normalise_tasks(tasks)
    record_map, record_elig, failures = _normalise_records(records, task_map)
    choice_map, invalid_choices = _normalise_choices(choices, task_map)

    for task_id, task in task_map.items():
        rounds = set(task["round_ids"])
        rounds.update(k[1] for k in record_map if k[0] == task_id)
        rounds.update(k[1] for k in choice_map if k[0] == task_id)
        task["round_ids"] = sorted(rounds)

    for key, model in list(choice_map.items()):
        eligible = _resolve_eligible(*key, declared, record_elig, record_map, choice_map)
        if model not in eligible:
            invalid_choices.append({
                "task_id": key[0], "round_id": key[1], "budget": key[2],
                "model_id": model, "reason": "model_not_eligible",
            })
    bad_keys = {(x["task_id"], x["round_id"], x["budget"]) for x in invalid_choices}
    valid_choices = {k: v for k, v in choice_map.items() if k not in bad_keys}
    _labels_and_validate_samples(task_map, record_map)

    coverage = {}
    coverage_rows = []
    for task in task_map.values():
        if task["role"] == "excluded":
            continue
        for budget in _BUDGETS:
            details = _coverage(task, budget, task["round_ids"], record_map,
                                declared, record_elig, choice_map)
            coverage[(task["task_id"], budget)] = details
            coverage_rows.append(details["coverage"])

    fixed = _fixed_models(task_map, coverage)
    locked = [t for t in task_map.values() if t["role"] == "locked"]
    development = [t for t in task_map.values() if t["role"] == "development"]
    score_rows, method_rows = [], {m: {} for m in _METHODS}
    method_incomplete = {m: [] for m in _METHODS}
    for task in locked:
        for budget in _BUDGETS:
            task_id = task["task_id"]
            matrix = coverage[(task_id, budget)]
            if not matrix["coverage"]["complete"]:
                for method, (_, b) in _METHODS.items():
                    if b == budget:
                        method_incomplete[method].append({
                            "task_id": task_id, "reason_codes": matrix["coverage"]["reason_codes"],
                        })
                continue
            rounds = matrix["round_infos"]
            scores, reasons = _method_task_scores(
                rounds, task, budget, valid_choices, fixed[str(budget)]["model_id"],
                seed, random_draws,
            )
            row = {
                "task_id": task_id, "suite": task["suite"], "family_id": task["family_id"],
                "budget": budget, "round_count": len(rounds),
                "sample_count": sum(len(item["rows"]) for item in rounds),
            }
            suffix = {1: "0", 3: "1"}[budget]
            for prefix, name in (("R", "R"), ("A", "A"), ("F", "F"), ("P", "P")):
                method = prefix + suffix
                if name in scores:
                    row[method] = scores[name]
                    method_rows[method][task_id] = row
                else:
                    method_incomplete[method].append({
                        "task_id": task_id, "reason_codes": reasons[name] or ["method_unavailable"],
                    })
            score_rows.append(row)

    methods = {}
    for method, (_, budget) in _METHODS.items():
        metric_rows = [
            {"task_id": task["task_id"], "suite": task["suite"], "family_id": task["family_id"],
             **method_rows[method][task["task_id"]][method]}
            for task in locked if task["task_id"] in method_rows[method]
        ]
        methods[method] = {
            "budget": budget, "selection": _METHODS[method][0],
            "task_count": len(metric_rows), "expected_locked_task_count": len(locked),
            "metrics": _mean_metrics(metric_rows),
            "by_suite": _group_summary(metric_rows, locked, "suite"),
            "by_family": _group_summary(metric_rows, locked, "family_id"),
            "incomplete_tasks": method_incomplete[method],
        }

    rows = [{
        "task_id": row["task_id"], "suite": row["suite"], "family_id": row["family_id"],
        "budget": row["budget"], **{m: row[m] for m in _METHODS if m in row},
    } for row in score_rows]
    paired_differences = _paired(rows, locked, seed, bootstrap_draws)
    paired_task_sets = {
        row["comparison"]: {"budget": row["budget"], "task_count": row["task_count"],
                            "task_ids": row["paired_task_ids"]}
        for row in paired_differences if row["metric"] == "balanced_accuracy"
    }
    roles = {}
    for role in ("development", "locked", "excluded"):
        role_tasks = [t for t in task_map.values() if t["role"] == role]
        roles[role] = {
            "task_count": len(role_tasks),
            "tasks_by_suite": {
                suite: sum(t["suite"] == suite for t in role_tasks)
                for suite in sorted({t["suite"] for t in role_tasks})
            },
            "matrix_complete_by_budget": {
                str(b): sum(bool(coverage.get((t["task_id"], b), {}).get("coverage", {}).get("complete"))
                            for t in role_tasks)
                for b in _BUDGETS
            },
        }
    return {
        "schema_version": "benchmark-report-v1", "seed": seed,
        "random_draws": random_draws, "bootstrap_draws": bootstrap_draws,
        "classification_scope": "所有输入任务按分类任务处理；正式效应统计仅使用 role=locked。",
        "scope": {
            "formal_role": "locked", "formal_task_count": len(locked),
            "formal_suites": sorted({t["suite"] for t in locked}),
            "development_task_count": len(development),
            "development_excluded_from_formal_effects": True,
            "excluded_task_count": sum(t["role"] == "excluded" for t in task_map.values()),
            "roles": roles,
        },
        "metric_definitions": {
            "task_weighting": "任务等权；先合并任务各轮held-out预测，再计算指标。",
            "balanced_accuracy": "完整标签表上的各类召回率平均；无支持类别召回率记0。",
            "macro_f1": "完整标签表上的Macro-F1；零分母类别记0。",
            "random": "每轮在eligible模型中等概率选择；BA与Accuracy解析期望，多轮Macro-F1随机模拟。",
            "validation_best": "每轮仅按validation_score最大选择，平分按model_id字母序；Test不参与选优。",
            "fixed": "依development任务等权平均BA选择固定模型；锁定任务完整性只作为启用门槛。",
        },
        "task_coverage": coverage_rows,
        "method_coverage": {
            m: {"complete_task_count": methods[m]["task_count"],
                "complete_task_ids": sorted(method_rows[m]),
                "expected_locked_task_count": len(locked),
                "incomplete_tasks": method_incomplete[m]} for m in _METHODS
        },
        "failures": failures, "invalid_choices": invalid_choices,
        "fixed_models": fixed, "methods": methods, "task_scores": rows,
        "paired_differences": paired_differences,
        "common_complete_task_subsets": paired_task_sets,
        "average_ranks": _average_ranks(rows, locked),
        "cost_summaries": _cost_summaries(record_map, task_map),
        "run_cost_details": _run_cost_details(record_map),
        "task_wall_time": _task_wall_time(driver_state, task_map),
    }


def write_report(report: dict, out_dir: Path) -> dict[str, Path]:
    """Write JSON, task scores, paired differences, and a Chinese Markdown report."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "report_json": directory / "report.json",
        "task_scores_csv": directory / "task_scores.csv",
        "paired_differences_csv": directory / "paired_differences.csv",
        "report_markdown": directory / "report.md",
    }
    paths["report_json"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    score_fields = ["task_id", "suite", "family_id", "budget", "round_count", "sample_count", *_METHODS]
    with paths["task_scores_csv"].open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=score_fields)
        writer.writeheader()
        for row in report.get("task_scores", []):
            output = {field: row.get(field) for field in score_fields}
            for method in _METHODS:
                output[method] = json.dumps(output[method], ensure_ascii=False, sort_keys=True) if output[method] is not None else ""
            writer.writerow(output)
    paired_fields = [
        "comparison", "left_method", "right_method", "budget", "metric", "task_count",
        "family_count", "paired_task_ids", "mean_difference", "ci95_lower", "ci95_upper",
        "win_count", "tie_count", "loss_count",
    ]
    with paths["paired_differences_csv"].open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=paired_fields)
        writer.writeheader()
        for row in report.get("paired_differences", []):
            interval = row["confidence_interval_95"]
            writer.writerow({
                "comparison": row["comparison"], "left_method": row["left_method"],
                "right_method": row["right_method"], "budget": row["budget"],
                "metric": row["metric"], "task_count": row["task_count"],
                "family_count": row["family_count"],
                "paired_task_ids": json.dumps(row["paired_task_ids"], ensure_ascii=False),
                "mean_difference": row["mean_difference"],
                "ci95_lower": interval["lower"], "ci95_upper": interval["upper"],
                "win_count": row["win_count"], "tie_count": row["tie_count"],
                "loss_count": row["loss_count"],
            })

    lines = [
        "# Benchmark 统计报告", "",
        f"正式统计范围为 {report['scope']['formal_task_count']} 个锁定分类任务；"
        f"{report['scope']['development_task_count']} 个开发任务仅用于确定固定模型，不进入正式效应统计。",
        f"正式 suite：{', '.join(report['scope']['formal_suites']) or '无'}。",
        "",
        "R 每轮从当轮合法模型均匀抽取；balanced accuracy 与 accuracy 报告解析期望，多轮 Macro-F1 按固定 seed 模拟。"
        "P 仅依据 validation_score 选模型，Test 只用于计算最终分数。",
        "", "## 方法结果", "",
        "| 方法 | 预算 | 完整任务 | Balanced accuracy | Macro-F1 | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, result in report["methods"].items():
        metrics = result["metrics"]
        fmt = lambda value: "—" if value is None else f"{value:.4f}"
        lines.append(
            f"| {method} | {result['budget']} | {result['task_count']}/{result['expected_locked_task_count']} | "
            f"{fmt(metrics['balanced_accuracy'])} | {fmt(metrics['macro_f1'])} | {fmt(metrics['accuracy'])} |"
        )
    lines.extend([
        "", "## 按 suite 汇总", "",
        "| 方法 | 预算 | suite | 完整/预期任务 | Balanced accuracy | Macro-F1 | Accuracy |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for method, result in report["methods"].items():
        for group, values in result["by_suite"].items():
            metrics = values["metrics"]
            lines.append(
                f"| {method} | {result['budget']} | {group} | "
                f"{values['task_count']}/{values['expected_task_count']} | "
                f"{fmt(metrics['balanced_accuracy'])} | {fmt(metrics['macro_f1'])} | "
                f"{fmt(metrics['accuracy'])} |"
            )
    lines.extend([
        "", "## 按 family 汇总", "",
        "| 方法 | 预算 | family_id | 完整/预期任务 | Balanced accuracy | Macro-F1 | Accuracy |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for method, result in report["methods"].items():
        for group, values in result["by_family"].items():
            metrics = values["metrics"]
            lines.append(
                f"| {method} | {result['budget']} | {group} | "
                f"{values['task_count']}/{values['expected_task_count']} | "
                f"{fmt(metrics['balanced_accuracy'])} | {fmt(metrics['macro_f1'])} | "
                f"{fmt(metrics['accuracy'])} |"
            )
    lines.extend(["", "## 开发任务确定的固定模型", ""])
    for budget in _BUDGETS:
        item = report["fixed_models"][str(budget)]
        if item["available"]:
            lines.append(
                f"- budget={budget}：模型 {item['model_id']}，开发任务平均BA "
                f"{item['development_mean_balanced_accuracy']:.4f}；候选 {', '.join(item['candidate_models'])}。"
            )
        else:
            lines.append(f"- budget={budget}：不可用（{item['reason']}）。")
    lines.extend([
        "", "## 配对差", "",
        "差值为左侧方法减右侧方法；95% 区间按 family_id 聚类 bootstrap；tie 阈值为 1e-12。",
        "", "| 比较 | 指标 | 任务数 | 平均差 | 95% CI | Win/Tie/Loss |",
        "|---|---|---:|---:|---|---:|",
    ])
    for row in report["paired_differences"]:
        ci = row["confidence_interval_95"]
        ci_text = "—" if ci["lower"] is None else f"[{ci['lower']:.4f}, {ci['upper']:.4f}]"
        mean = "—" if row["mean_difference"] is None else f"{row['mean_difference']:.4f}"
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['task_count']} | {mean} | {ci_text} | "
            f"{row['win_count']}/{row['tie_count']}/{row['loss_count']} |"
        )
    lines.extend(["", "## 平均秩", "",
                  "| 预算 | 指标 | 方法 | 平均秩 | 任务数 |",
                  "|---:|---|---|---:|---:|"])
    for budget, metrics_by_name in report["average_ranks"].items():
        for metric, method_values in metrics_by_name.items():
            for method, value in method_values.items():
                rank = "—" if value["average_rank"] is None else f"{value['average_rank']:.3f}"
                lines.append(f"| {budget} | {metric} | {method} | {rank} | {value['task_count']} |")
    lines.extend(["", "## 配对共同完整任务", ""])
    for comparison, subset in report["common_complete_task_subsets"].items():
        ids = ", ".join(subset["task_ids"]) or "无"
        lines.append(f"- {comparison}（{subset['task_count']} 个）：{ids}")
    lines.extend([
        "", "## 覆盖与失败", "",
        f"失败记录 {len(report['failures'])} 项；无效选择 {len(report['invalid_choices'])} 项。"
        "完整任务要求每个计划轮次的所有eligible模型均成功并共享相同Test样本。",
    ])
    for item in report["task_coverage"]:
        if not item["complete"]:
            reasons = ", ".join(item["reason_codes"]) or "incomplete"
            lines.append(f"- {item['role']}任务 {item['task_id']}，budget={item['budget']}：{reasons}")
    lines.extend([
        "", "## 费用字段", "",
        "fit_duration_sum_seconds 是各 Run 自身 cost.wall_seconds 的总和，不是任务实际墙钟时间；每个 Run 的 wall_seconds 保存在 report.json 的 run_cost_details。",
        "actual_cached_* 为实际复用记录；uncached_estimated_* 是基于缓存源拟合用时的估计，不代表实际重跑。LM token 只取 records.cost 明确给出的字段。",
        "", "| 任务角色 | 预算 | 记录数 | 实际拟合数 | Fit duration sum (s) | 实际缓存拟合 | 实际缓存秒数 | 未缓存估计拟合 | 未缓存估计秒数 | 已知 LM token | 未知 token 数 | 缓存 token 估计 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    cost_fmt = lambda value: "—" if value is None else f"{value:.3f}"
    for role, budgets in report["cost_summaries"].items():
        for budget, value in budgets.items():
            fields = value["fields"]
            def cost_value(name):
                return fields.get(name, {}).get("sum")
            lines.append(
                f"| {role} | {budget} | {value['record_count']} | "
                f"{cost_fmt(cost_value('actual_fits'))} | "
                f"{cost_fmt(cost_value('fit_duration_sum_seconds'))} | "
                f"{cost_fmt(cost_value('actual_cached_fits'))} | "
                f"{cost_fmt(cost_value('actual_cached_seconds'))} | "
                f"{cost_fmt(cost_value('uncached_estimated_fits'))} | "
                f"{cost_fmt(cost_value('uncached_estimated_seconds'))} | "
                f"{cost_fmt(cost_value('known_lm_tokens'))} | "
                f"{cost_fmt(cost_value('unknown_count'))} | "
                f"{cost_fmt(cost_value('cached_lm_tokens_estimate'))} |"
            )
    lines.extend([
        "", "### 任务实际墙钟时间", "",
        "task_wall_seconds 来自 driver_state 的任务时钟；driver_state 未提供时保留为未知，不以 Fit duration sum 代替。",
        "", "| 任务 | 角色 | task_wall_seconds | 云端核验时间 |",
        "|---|---|---:|---|",
    ])
    for item in report["task_wall_time"]["tasks"]:
        lines.append(
            f"| {item['task_id']} | {item['role']} | {cost_fmt(item['task_wall_seconds'])} | "
            f"{item['cloud_verified_at'] or '—'} |"
        )
    paths["report_markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths

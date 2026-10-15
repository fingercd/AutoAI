from itertools import product
import json

from backend.app.benchmark_report import summarize, write_report


LABELS = ["a", "b"]


def _task(task_id, role, family_id, round_models):
    return {
        "task_id": task_id,
        "suite": "suite-x",
        "family_id": family_id,
        "role": role,
        "labels": LABELS,
        "rounds": [
            {
                "round_id": round_id,
                "eligible_models": {"1": models, "3": models},
            }
            for round_id, models in round_models.items()
        ],
    }


def _rows(task_id, round_id, true_labels, predicted_labels):
    return [
        {
            "sample_id": f"{task_id}-{round_id}-{index}",
            "true_label": truth,
            "predicted_label": predicted,
            "probabilities": {"a": 0.7 if predicted == "a" else 0.3,
                              "b": 0.7 if predicted == "b" else 0.3},
        }
        for index, (truth, predicted) in enumerate(zip(true_labels, predicted_labels))
    ]


def _record(task_id, round_id, model_id, budget, rows, validation=0.5, eligible=None):
    return {
        "task_id": task_id,
        "round_id": round_id,
        "model_id": model_id,
        "budget": budget,
        "status": "completed",
        "validation_score": validation,
        "prediction_rows": rows,
        "eligible_models": eligible or [model_id],
    }


def _add_round(records, task_id, round_id, models, truths, predictions, validations=None):
    validations = validations or {}
    rows = {
        model: _rows(task_id, round_id, truths, predictions[model])
        for model in models
    }
    for budget in (1, 3):
        for model in models:
            records.append(_record(
                task_id, round_id, model, budget, rows[model],
                validations.get(model, 0.5), models,
            ))


def _score(rows):
    scores = []
    for label in LABELS:
        tp = sum(truth == label and predicted == label for truth, predicted in rows)
        support = sum(truth == label for truth, _ in rows)
        predicted_support = sum(predicted == label for _, predicted in rows)
        denom = support + predicted_support
        scores.append(2 * tp / denom if denom else 0)
    return sum(scores) / len(scores)

def test_random_macro_f1_uses_pooled_random_draws_not_mean_round_f1():
    task = _task("locked-1", "locked", "fam-1", {
        "r0": ["m1", "m2"], "r1": ["m1", "m2"],
    })
    records = []
    _add_round(records, "locked-1", "r0", ["m1", "m2"], ["a", "b"], {
        "m1": ["a", "a"], "m2": ["b", "b"],
    })
    _add_round(records, "locked-1", "r1", ["m1", "m2"], ["a", "b"], {
        "m1": ["a", "b"], "m2": ["a", "a"],
    })
    choices = [
        {"task_id": "locked-1", "round_id": rid, "budget": budget, "model_id": "m1"}
        for rid in ("r0", "r1") for budget in (1, 3)
    ]
    report = summarize(records, choices, [task], random_draws=40000, bootstrap_draws=50)

    combinations = []
    per_round_means = []
    predictions = {
        "r0": {"m1": ["a", "a"], "m2": ["b", "b"]},
        "r1": {"m1": ["a", "b"], "m2": ["a", "a"]},
    }
    for rid in ("r0", "r1"):
        round_scores = []
        for model in ("m1", "m2"):
            round_rows = list(zip(["a", "b"], predictions[rid][model]))
            round_scores.append(_score(round_rows))
        per_round_means.append(sum(round_scores) / len(round_scores))
    for first, second in product(("m1", "m2"), repeat=2):
        pooled = []
        for rid, model in zip(("r0", "r1"), (first, second)):
            pooled.extend(zip(["a", "b"], predictions[rid][model]))
        combinations.append(_score(pooled))
    exact = sum(combinations) / len(combinations)
    random_f1 = report["methods"]["R0"]["metrics"]["macro_f1"]
    assert abs(random_f1 - exact) < 0.01
    assert abs(exact - sum(per_round_means) / 2) > 0.001
    assert report["methods"]["R0"]["metrics"]["accuracy"] == 0.625


def test_family_cluster_bootstrap_and_paired_win_tie_loss_are_reported():
    tasks, records, choices = [], [], []
    family_ids = ["family-a", "family-a", "family-b", "family-c"]
    for index, family in enumerate(family_ids):
        task_id = f"locked-{index}"
        tasks.append(_task(task_id, "locked", family, {"r0": ["good", "bad"]}))
        _add_round(records, task_id, "r0", ["good", "bad"], ["a", "b"], {
            "good": ["a", "b"], "bad": ["a", "a"],
        })
        choices.extend(
            {"task_id": task_id, "round_id": "r0", "budget": budget, "model_id": "good"}
            for budget in (1, 3)
        )
    one = summarize(records, choices, tasks, random_draws=100, bootstrap_draws=400)
    two = summarize(records, choices, tasks, random_draws=100, bootstrap_draws=400)
    difference = next(
        row for row in one["paired_differences"]
        if row["comparison"] == "A0-R0" and row["metric"] == "balanced_accuracy"
    )
    assert difference["task_count"] == 4
    assert difference["family_count"] == 3
    assert difference["win_count"] == 4
    assert difference["tie_count"] == difference["loss_count"] == 0
    assert difference["confidence_interval_95"]["method"] == "family_cluster_percentile_bootstrap"
    assert difference["confidence_interval_95"]["lower"] == 0.25
    assert difference == next(
        row for row in two["paired_differences"]
        if row["comparison"] == "A0-R0" and row["metric"] == "balanced_accuracy"
    )


def test_incomplete_planned_model_makes_whole_task_incomplete():
    task = _task("locked-1", "locked", "fam-1", {"r0": ["good", "missing"]})
    good_rows = _rows("locked-1", "r0", ["a", "b"], ["a", "b"])
    records = [
        _record("locked-1", "r0", "good", 1, good_rows, eligible=["good", "missing"]),
    ]
    choices = [{"task_id": "locked-1", "round_id": "r0", "budget": 1, "model_id": "good"}]
    report = summarize(records, choices, [task], random_draws=10, bootstrap_draws=20)
    entry = next(x for x in report["task_coverage"] if x["task_id"] == "locked-1" and x["budget"] == 1)
    assert entry["complete"] is False
    assert entry["rounds"][0]["missing_models"] == ["missing"]
    assert report["methods"]["R0"]["task_count"] == 0
    assert report["methods"]["A0"]["task_count"] == 0


def test_fixed_model_is_unavailable_without_development_tasks():
    task = _task("locked-1", "locked", "fam-1", {"r0": ["m1", "m2"]})
    records = []
    _add_round(records, "locked-1", "r0", ["m1", "m2"], ["a", "b"], {
        "m1": ["a", "b"], "m2": ["a", "a"],
    })
    choices = [
        {"task_id": "locked-1", "round_id": "r0", "budget": budget, "model_id": "m1"}
        for budget in (1, 3)
    ]
    report = summarize(records, choices, [task], random_draws=10, bootstrap_draws=20)
    assert report["fixed_models"]["1"]["available"] is False
    assert report["fixed_models"]["1"]["reason"] == "no_development_tasks"
    assert report["methods"]["F0"]["task_count"] == 0
    assert report["methods"]["F0"]["metrics"]["balanced_accuracy"] is None


def test_round_specific_eligibility_and_invalid_choice_are_preserved():
    task = _task("locked-1", "locked", "fam-1", {
        "r0": ["m1", "m2"], "r1": ["m2", "m3"],
    })
    records = []
    _add_round(records, "locked-1", "r0", ["m1", "m2"], ["a", "b"], {
        "m1": ["a", "b"], "m2": ["a", "a"],
    })
    _add_round(records, "locked-1", "r1", ["m2", "m3"], ["a", "b"], {
        "m2": ["b", "b"], "m3": ["a", "b"],
    })
    choices = [
        {"task_id": "locked-1", "round_id": "r0", "budget": budget, "model_id": "m3"}
        for budget in (1, 3)
    ] + [
        {"task_id": "locked-1", "round_id": "r1", "budget": budget, "model_id": "m3"}
        for budget in (1, 3)
    ]
    report = summarize(records, choices, [task], random_draws=50, bootstrap_draws=20)
    assert report["methods"]["R0"]["metrics"]["accuracy"] == 0.75
    assert report["methods"]["A0"]["task_count"] == 0
    assert len(report["invalid_choices"]) == 2
    assert {x["reason"] for x in report["invalid_choices"]} == {"model_not_eligible"}


def test_validation_portfolio_uses_validation_not_test_and_writes_outputs(tmp_path):
    development = _task("dev-1", "development", "dev-family", {"r0": ["oracle", "valid"]})
    locked = _task("locked-1", "locked", "locked-family", {"r0": ["oracle", "valid"]})
    records = []
    _add_round(records, "dev-1", "r0", ["oracle", "valid"], ["a", "b"], {
        "oracle": ["a", "a"], "valid": ["a", "b"],
    }, {"oracle": 0.4, "valid": 0.9})
    _add_round(records, "locked-1", "r0", ["oracle", "valid"], ["a", "b"], {
        "oracle": ["a", "b"], "valid": ["a", "a"],
    }, {"oracle": 0.1, "valid": 0.99})
    choices = [
        {"task_id": task_id, "round_id": "r0", "budget": budget, "model_id": "oracle"}
        for task_id in ("dev-1", "locked-1") for budget in (1, 3)
    ]
    report = summarize(records, choices, [development, locked], random_draws=10, bootstrap_draws=20)
    assert report["scope"]["formal_task_count"] == 1
    assert report["scope"]["development_excluded_from_formal_effects"] is True
    assert report["methods"]["A0"]["metrics"]["balanced_accuracy"] == 1.0
    assert report["methods"]["A0"]["by_suite"]["suite-x"]["metrics"]["balanced_accuracy"] == 1.0
    assert report["methods"]["A0"]["by_family"]["locked-family"]["task_count"] == 1
    assert report["average_ranks"]["1"]["balanced_accuracy"]["A0"]["average_rank"] == 1.0
    assert report["common_complete_task_subsets"]["A0-R0"]["task_ids"] == ["locked-1"]
    assert report["methods"]["P0"]["metrics"]["balanced_accuracy"] == 0.5
    assert report["fixed_models"]["1"]["model_id"] == "valid"
    assert "choices" not in report and "prediction_rows" not in report

    paths = write_report(report, tmp_path / "report")
    assert set(paths) == {"report_json", "task_scores_csv", "paired_differences_csv", "report_markdown"}
    assert all(path.is_file() for path in paths.values())
    saved = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    assert saved["scope"]["formal_suites"] == ["suite-x"]
    markdown = paths["report_markdown"].read_text(encoding="utf-8")
    assert "固定模型" in markdown and "paired" not in markdown.lower()



def test_fixed_model_uses_only_dev_scores_and_locked_eligibility():
    development = _task("dev-1", "development", "dev-family", {"r0": ["m1", "m2"]})
    locked = _task("locked-1", "locked", "locked-family", {"r0": ["m1"]})
    records = []
    _add_round(records, "dev-1", "r0", ["m1", "m2"], ["a", "b"], {
        "m1": ["a", "b"], "m2": ["a", "a"],
    })
    report = summarize(records, [], [development, locked], random_draws=10, bootstrap_draws=20)
    assert report["fixed_models"]["1"]["available"] is True
    assert report["fixed_models"]["1"]["candidate_models"] == ["m1"]
    assert report["fixed_models"]["1"]["model_id"] == "m1"
    assert report["methods"]["F0"]["task_count"] == 0
    assert report["methods"]["F0"]["metrics"]["balanced_accuracy"] is None
    assert not any(row["task_id"] == "locked-1" for row in report["task_scores"])


def test_driver_task_wall_and_cached_costs_have_separate_scopes():
    task = _task("locked-1", "locked", "fam-1", {"r0": ["m1", "m2"]})
    records = []
    _add_round(records, "locked-1", "r0", ["m1", "m2"], ["a", "b"], {
        "m1": ["a", "b"], "m2": ["a", "a"],
    })
    records[0]["cost"] = {
        "actual_fits": 2, "wall_seconds": 12.5, "driver_wall_seconds": 15.0,
        "actual_cached_fits": 1, "actual_cached_seconds": 4.0,
        "uncached_estimated_fits": 3, "uncached_estimated_seconds": 22.5,
        "known_cached_source_fit_seconds": 14.0,
        "known_lm_tokens": 100, "unknown_count": 1, "cached_lm_tokens_estimate": 5,
    }
    state = {
        "task_wall_seconds": {"locked-1": 35.0},
        "task_clocks": [{"task_id": "locked-1", "task_wall_seconds": 35.0,
                         "cloud_verified_at": 1234.0}],
    }
    report = summarize(records, [], [task], random_draws=10, bootstrap_draws=20,
                       driver_state=state)
    fields = report["cost_summaries"]["locked"]["1"]["fields"]
    assert fields["fit_duration_sum_seconds"]["sum"] == 12.5
    assert fields["driver_wall_duration_sum_seconds"]["sum"] == 15.0
    assert fields["actual_cached_seconds"]["sum"] == 4.0
    assert fields["uncached_estimated_seconds"]["sum"] == 22.5
    assert "wall_seconds" not in fields
    assert report["run_cost_details"][0]["fit_duration_seconds"] == 12.5
    wall = report["task_wall_time"]["tasks"][0]
    assert wall["task_wall_seconds"] == 35.0
    assert wall["cloud_verified_at"] == 1234.0



def test_random_macro_f1_matches_same_seed_pooled_prediction_reference_with_missing_labels():
    import hashlib
    import numpy as np
    import pytest
    from sklearn.metrics import f1_score
    from backend.app.benchmark_report import _confusion, _random_score

    # Each round misses true classes; ghost is predicted-only and zero has no support.
    labels = ['a', 'b', 'c', 'ghost', 'zero']
    truths = [['a', 'a', 'a'], ['b', 'b'], ['c', 'a']]
    predictions = [
        {'m2': ['a', 'ghost', 'b'], 'm1': ['a', 'a', 'a']},
        {'m3': ['ghost', 'b'], 'm1': ['b', 'b'], 'm2': ['a', 'a']},
        {'m2': ['a', 'c'], 'm1': ['c', 'a']},
    ]
    rounds = [dict(eligible_models=list(models), confusions={
        model: _confusion([dict(true_label=truth, predicted_label=prediction)
                           for truth, prediction in zip(round_truths, model_predictions)], labels)
        for model, model_predictions in models.items()
    }) for round_truths, models in zip(truths, predictions)]

    # Keep the existing generator's per-round vector draws and sorted model ordering.
    task, budget, draws = 'missing-label-reference', 3, 10000
    seed_digest = hashlib.sha256(f'42|{task}|{budget}|random-macro-f1'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(seed_digest[:8], 'big'))
    model_orders = [sorted(models) for models in predictions]
    selections = [rng.integers(0, len(models), size=draws) for models in model_orders]
    pooled_truths = [truth for round_truths in truths for truth in round_truths]

    def straightforward_f1(pooled_predictions):
        scores = []
        for label in labels:
            true_support = sum(truth == label for truth in pooled_truths)
            predicted_support = sum(prediction == label for prediction in pooled_predictions)
            tp = sum(truth == label and prediction == label
                     for truth, prediction in zip(pooled_truths, pooled_predictions))
            denominator = true_support + predicted_support
            scores.append(2 * tp / denominator if denominator else 0.0)
        return sum(scores) / len(labels)

    # Independently check that the reference retains explicit-label/zero_division=0 semantics.
    for chosen in product(*model_orders):
        pooled = [prediction for models, model in zip(predictions, chosen)
                  for prediction in models[model]]
        assert straightforward_f1(pooled) == pytest.approx(
            f1_score(pooled_truths, pooled, labels=labels, average='macro', zero_division=0),
            rel=0, abs=1e-15,
        )

    reference = []
    for draw in range(draws):
        pooled = [prediction
                  for models, order, selected in zip(predictions, model_orders, selections)
                  for prediction in models[order[selected[draw]]]]
        reference.append(straightforward_f1(pooled))
    actual = _random_score(rounds, labels, task, budget, 42, draws)
    assert actual['macro_f1'] == pytest.approx(np.mean(reference), rel=0, abs=1e-15)

    expected_matrix = sum((np.mean(list(item['confusions'].values()), axis=0)
                           for item in rounds), np.zeros((len(labels), len(labels))))
    true_support = expected_matrix.sum(axis=1)
    recall = np.divide(np.diag(expected_matrix), true_support,
                       out=np.zeros(len(labels)), where=true_support > 0)
    assert actual['balanced_accuracy'] == pytest.approx(recall.mean(), rel=0, abs=1e-15)
    assert actual['accuracy'] == pytest.approx(
        np.trace(expected_matrix) / expected_matrix.sum(), rel=0, abs=1e-15)

"""Recompute published task-level results offline; no training, LLM or network access."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1] / "docs/research/results"
METRICS = ("test_balanced_accuracy", "test_accuracy", "test_macro_f1")
MODELS = {"pls_da", "pca_lda", "logistic_regression", "svm", "random_forest", "xgboost", "cnn1d"}


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main():
    matrix = read("seven-model-scores.json")
    indexed = {(row["suite"], row["task_id"]): row for row in matrix}
    assert len(matrix) == len(indexed) == 129
    assert all(set(row["models"]) == MODELS and row["round_id"] == "01" for row in matrix)
    sources = Counter(record["source"] for row in matrix for record in row["models"].values())
    assert sources == {"reused": 873, "new_training": 30}
    checked = {}
    for version in ("initial", "revised"):
        choices = read(f"{version}-selections.json")
        summary = read(f"{version}-summary.json")["overall"]
        assert len(choices) == 129
        assert {(r["suite"], r["task_id"]) for r in choices} == set(indexed)
        assert dict(Counter(r["model_id"] for r in choices)) == {
            model: count for model, count in summary["model_counts"].items() if count
        }
        values = {}
        for metric in METRICS:
            selected, random = [], []
            for choice in choices:
                row = indexed[choice["suite"], choice["task_id"]]
                a = row["models"][choice["model_id"]]["metrics"][metric]
                b = mean(record["metrics"][metric] for record in row["models"].values())
                recorded = choice["metrics"][metric]
                assert abs(a - recorded["selected"]) < 1e-12
                assert abs(b - recorded["uniform_random_expectation"]) < 1e-12
                assert abs(b - row["uniform_random_expectation"][metric]) < 1e-12
                assert abs(a - b - recorded["delta"]) < 1e-12
                selected.append(a)
                random.append(b)
            aggregate = summary["metrics"][metric]
            assert abs(mean(selected) - aggregate["selected_mean"]) < 1e-12
            assert abs(mean(random) - aggregate["uniform_random_mean"]) < 1e-12
            assert abs(100 * (mean(selected) - mean(random)) - aggregate["delta_percentage_points"]) < 1e-10
            values[metric] = aggregate
        differences = [r["metrics"]["test_balanced_accuracy"]["delta"] for r in choices]
        assert sum(d > 1e-12 for d in differences) == summary["higher"]
        assert sum(abs(d) <= 1e-12 for d in differences) == summary["tied"]
        assert sum(d < -1e-12 for d in differences) == summary["lower"]
        checked[version] = values
    previous = read("previous-vs-revised.json")
    assert len(previous["tasks"]) == 129
    for metric in METRICS:
        delta = 100 * (checked["revised"][metric]["selected_mean"] - checked["initial"][metric]["selected_mean"])
        assert abs(delta - previous["summary"]["overall_changes_percentage_points"][metric]) < 1e-10
    print(json.dumps({"passed": True, "tasks": 129, "score_cells": 903, "versions": 2,
                      "new_classifier_fits": 0, "llm_requests": 0, "metrics": checked}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

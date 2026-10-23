"""Offline unit tests: no classifier training, real LLM, GPU or network calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.app.selection_ablation import (
    CONDITIONS, MODELS, SETTINGS, digest, make_request, parse_action,
    strict_json, summarize, validate_scores,
)

SPEC = importlib.util.spec_from_file_location("selection_ablation_runner",
    Path(__file__).resolve().parents[2] / "scripts/selection_ablation.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fixture():
    semantic = dict(version="old", task={"train_summary": {"train_groups": 20}},
        candidates=[{"model_id": m} for m in reversed(MODELS)],
        retrieved_knowledge=[{"id": "k1", "text": "example"}])
    request = dict(SETTINGS, messages=[{"role": "system", "content": "unchanged"},
        {"role": "user", "content": json.dumps(semantic, ensure_ascii=False, separators=(",", ":"))}])
    return dict(semantic_input=semantic, request=request)


def score():
    return dict(suite="example", task_id="one", family_id="f1", models={
        model: {"metrics": dict(test_balanced_accuracy=(i + 1) / 10,
            test_accuracy=(i + 1) / 10, test_macro_f1=(i + 1) / 10)}
        for i, model in enumerate(MODELS)})


def action():
    return json.dumps(dict(tool_name="submit_ml_experiment",
        arguments={"model_id": "svm"}, rationale=" evidence "))


class CoreTests(unittest.TestCase):
    def test_knowledge_request_byte_structure_unchanged(self):
        f = fixture()
        self.assertEqual(make_request(f, "with_knowledge"), f["request"])

    def test_only_knowledge_removed(self):
        f = fixture()
        before = deepcopy(f)
        request = make_request(f, "without_knowledge")
        expected = deepcopy(f["semantic_input"])
        expected["retrieved_knowledge"] = []
        self.assertEqual(request["messages"][1]["content"],
            json.dumps(expected, ensure_ascii=False, separators=(",", ":")))
        self.assertEqual(request["messages"][0], f["request"]["messages"][0])
        self.assertEqual(f, before)

    def test_changed_generation_settings_rejected(self):
        f = fixture()
        f["request"]["temperature"] = 1
        with self.assertRaises(ValueError):
            make_request(f, "with_knowledge")

    def test_result_leakage_rejected(self):
        f = fixture()
        f["semantic_input"]["task"]["test"] = 0.9
        f["request"]["messages"][1]["content"] = json.dumps(f["semantic_input"])
        with self.assertRaises(ValueError):
            make_request(f, "with_knowledge")

    def test_input_request_disagreement_rejected(self):
        f = fixture()
        f["semantic_input"]["task"]["train_groups"] = 1
        with self.assertRaises(ValueError):
            make_request(f, "with_knowledge")

    def test_legal_action_and_outer_brace(self):
        self.assertEqual(parse_action(action())["model_id"], "svm")
        repaired = parse_action(action()[:-1])
        self.assertFalse(repaired["raw_json_valid"])
        self.assertIn("append_final_object_brace", repaired["format_operations"])

    def test_duplicate_or_extra_fields_rejected(self):
        for value in ['{"x":1,"x":2}', '{"x":NaN}']:
            with self.assertRaises(ValueError):
                strict_json(value)
        data = json.loads(action())
        data["extra"] = 1
        with self.assertRaises(ValueError):
            parse_action(json.dumps(data))

    def test_invalid_models_and_nested_truncation_rejected(self):
        for value in [action().replace('"svm"', '"unknown"'),
                      '{"tool_name":"submit_ml_experiment","arguments":{"model_id":"svm"']:
            with self.assertRaises(ValueError):
                parse_action(value)

    def test_all_fixed_strategies_and_random_expectation(self):
        receipts = [dict(suite="example", task_id="one", condition=c,
            outcome="accepted", choice={"model_id": m})
            for c, m in zip(CONDITIONS, ("pls_da", "svm"))]
        result, rows = summarize([score()], receipts)
        self.assertTrue(result["primary_comparison_complete"])
        self.assertEqual(len(result["methods"]), 10)
        self.assertAlmostEqual(result["methods"]["uniform_random"]["mean_balanced_accuracy"], 0.4)
        self.assertAlmostEqual(result["paired"]["mean_delta_percentage_points"], -30)
        self.assertEqual(result["paired"]["losses"], 1)

    def test_failure_has_no_fallback_or_fake_complete_comparison(self):
        receipts = [dict(suite="example", task_id="one", condition=c,
                        outcome="failed") for c in CONDITIONS]
        result, _ = summarize([score()], receipts)
        self.assertFalse(result["primary_comparison_complete"])
        self.assertEqual(result["paired"]["paired_tasks"], 0)
        self.assertIsNone(result["methods"]["with_knowledge"]["mean_balanced_accuracy"])

    def test_duplicate_receipts_and_scores_rejected(self):
        r = dict(suite="example", task_id="one", condition=CONDITIONS[0], outcome="failed")
        with self.assertRaises(ValueError):
            summarize([score()], [r, r])
        with self.assertRaises(ValueError):
            validate_scores([score(), score()])


class Journal:
    def __init__(self, root):
        self.root = root
        self.count = 0
        (root / "events").mkdir(parents=True)

    def task_dir(self, task):
        return self.root

    def ensure_online(self, *args, **kwargs):
        pass

    def emit(self, task, kind, data):
        self.count += 1
        runner.atomic_json(self.root / "events" / f"{self.count}.json", dict(kind=kind, data=data))


class Response:
    status = 200
    def __init__(self, body):
        self.body = body
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def read(self, limit):
        return self.body[:limit]


class Opener:
    def __init__(self, body):
        self.body, self.calls = body, 0
    def open(self, *args, **kwargs):
        self.calls += 1
        return Response(self.body)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / "work")
        self.root = Path(self.tmp.name)
        self.item = dict(key="k", suite="example", task_id="one", condition="with_knowledge",
                         request_sha256=digest(fixture()["request"]))
        runner.atomic_json(self.root / "inputs/k.json", dict(request=fixture()["request"]))
        self.journal = Journal(self.root / "tracking")

    def tearDown(self):
        self.tmp.cleanup()

    def test_completed_request_never_repeated(self):
        payload = dict(model="qwen35_9b", id="fake",
            choices=[{"message": {"content": action()}}],
            usage=dict(prompt_tokens=1, completion_tokens=2, total_tokens=3))
        opener = Opener(json.dumps(payload).encode())
        with patch.object(runner, "ensure_health"):
            a = runner.request_one(self.root, self.item, self.journal, opener, "http://localhost")
            b = runner.request_one(self.root, self.item, self.journal, opener, "http://localhost")
        self.assertEqual(a, b)
        self.assertEqual(opener.calls, 1)
        self.assertEqual(self.journal.count, 1)

    def test_unknown_request_never_resent(self):
        runner.atomic_json(self.root / "receipts/k.json", dict(self.item, state="dispatched"))
        opener = Opener(b"")
        with self.assertRaises(RuntimeError):
            runner.request_one(self.root, self.item, self.journal, opener, "http://localhost")
        self.assertEqual(opener.calls, 0)

    def test_malformed_response_is_preserved_without_retry(self):
        opener = Opener(b"invalid provider json")
        with patch.object(runner, "ensure_health"):
            receipt = runner.request_one(self.root, self.item, self.journal, opener, "http://localhost")
        self.assertEqual(receipt["outcome"], "failed")
        saved = runner.read_json(self.root / "responses/k.json")
        self.assertEqual(bytes.fromhex(saved["body_hex"]), b"invalid provider json")
        self.assertEqual(opener.calls, 1)


    def test_report_reads_persisted_score_path(self):
        score_path = self.root / "scores.json"
        runner.atomic_json(score_path, [score()])
        entries = []
        for i, condition in enumerate(CONDITIONS):
            key = "report-" + str(i)
            entries.append({"key": key})
            runner.atomic_json(self.root / "receipts" / (key + ".json"), dict(
                suite="example", task_id="one", condition=condition, outcome="accepted",
                choice={"model_id": "svm"}, request_duration_seconds=1,
                usage=dict(prompt_tokens=1, completion_tokens=2, total_tokens=3)))
        manifest = dict(entries=entries, score_table=str(score_path))
        with patch.object(runner, "verify_frozen", return_value=manifest):
            result = runner.build_report(self.root)
        self.assertTrue(result["primary_comparison_complete"])
        self.assertEqual(result["cost"]["physical_requests"], 2)
        self.assertTrue((self.root / "report/report.md").is_file())


class HardwareTests(unittest.TestCase):
    def test_v100_and_a100_policy(self):
        spec = importlib.util.spec_from_file_location("service", Path(__file__).resolve().parents[2] / "scripts/serve_selection_llm.py")
        service = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(service)
        self.assertEqual(service.hardware_policy((7, 0))["dtype"], "float16")
        self.assertTrue(service.hardware_policy((7, 0))["sdpa_math_only"])
        self.assertEqual(service.hardware_policy((8, 0))["dtype"], "bfloat16")
        self.assertFalse(service.hardware_policy((8, 0))["sdpa_math_only"])
        with self.assertRaises(ValueError):
            service.hardware_policy((6, 0))


if __name__ == "__main__":
    unittest.main()

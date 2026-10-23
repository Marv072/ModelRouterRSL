import math
import tempfile
import unittest
from pathlib import Path

from app.core.rsl import RSLStore, classify, select_model
from app.core.routing_policy import RoutingPolicy


class LearningTests(unittest.TestCase):
    def setUp(self):
        Path(".test-tmp").mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=".test-tmp")
        self.path = str(Path(self.tmp.name) / "rsl.db")
        self.store = RSLStore(self.path)
        self.policy = RoutingPolicy(rsl_min_samples=1, rsl_exploration_weight=0)

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, id="one", model="cheap", score=0.2, profile="p", category="math", cost=0.001):
        self.store.record(id=id, request_id=id, profile=profile, category=category, model=model,
            score=score, source="test", reason="fixture", cost=cost, latency_ms=100)

    def test_feedback_changes_route_and_survives_restart(self):
        self.record()
        self.record("two", "strong", 0.95)
        result = select_model(RSLStore(self.path), {"cheap": 0.001, "strong": 0.001}, self.policy, "p", "math")
        self.assertEqual(result["selected_model"], "strong")

    def test_cold_start_explores_even_with_zero_bonus(self):
        self.record(score=1)
        result = select_model(self.store, {"cheap": 0, "strong": 1}, self.policy, "p", "math")
        self.assertEqual(result["selected_model"], "strong")

    def test_profile_and_category_isolation(self):
        self.record()
        self.assertEqual(self.store.stats("other"), [])
        self.assertEqual(self.store.stats("p", "coding"), [])

    def test_correction_replaces_quality_without_changing_cost_or_count(self):
        self.record(score=0)
        self.store.correct("one", 1, "corrected")
        self.store.correct("one", 0.9, "corrected again")
        row = self.store.stats("p")[0]
        self.assertEqual(row["samples"], 1)
        self.assertEqual(row["quality"], 0.9)
        self.assertEqual(row["avg_cost"], 0.001)
        self.assertEqual(self.store.feedback("p")[0]["score"], 0)

    def test_duplicate_attempt_does_not_double_count(self):
        self.record(); self.record()
        self.assertEqual(self.store.stats("p")[0]["attempts"], 1)

    def test_pending_quality_and_unknown_cost_not_zero_or_success(self):
        self.record(score=None, cost=None)
        row = self.store.stats("p")[0]
        self.assertEqual(row["samples"], 0)
        self.assertIsNone(row["quality"])
        self.assertIsNone(row["avg_cost"])
        self.assertEqual(row["unknown_cost_attempts"], 1)

    def test_actual_cost_replaces_static_prior(self):
        self.record(model="cheap", score=0.9, cost=0.01)
        self.record("two", "strong", 0.9, cost=0.001)
        result = select_model(self.store, {"cheap": 0, "strong": 1}, self.policy, "p", "math", explore=False)
        self.assertEqual(result["selected_model"], "strong")

    def test_invalid_feedback_rejected(self):
        for score in (math.nan, math.inf, -1, 2):
            with self.assertRaises(ValueError):
                self.record(score=score)
        with self.assertRaises(KeyError):
            self.store.correct("missing", 1, "")

    def test_task_classification(self):
        self.assertEqual(classify([{"role": "user", "content": "写 Python 函数"}]), "coding")
        self.assertEqual(classify([{"role": "user", "content": "计算 1+1"}]), "math")
        self.assertEqual(classify([{"role": "user", "content": "翻译 hello"}]), "translation")

    def test_frozen_selection_does_not_learn(self):
        self.record()
        before = self.store.stats("p")
        select_model(self.store, {"cheap": 0}, self.policy, "p", "math", explore=False)
        self.assertEqual(self.store.stats("p"), before)


if __name__ == "__main__":
    unittest.main()

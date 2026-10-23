import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from examples.offline_demo import configure_demo_engine, DemoProvider


class APITests(unittest.TestCase):
    def setUp(self):
        Path(".test-tmp").mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=".test-tmp")
        config = Path(self.tmp.name) / "providers.yaml"
        config.write_text("providers: {}\n", encoding="utf-8")
        self.env = patch.dict(os.environ, {
            "ROUTER_CONFIG": str(config), "ROUTER_DB": str(Path(self.tmp.name) / "router.db"),
            "RSL_DB": str(Path(self.tmp.name) / "rsl.db"), "ROUTER_RUNTIME_MODE": "internal",
            "ROUTER_INTERNAL_API_KEY": "test-runtime", "ADMIN_API_KEY": "test-admin",
            "ROUTER_ENABLE_STANDALONE_KEYS": "false", "ROUTER_ENABLE_ADMIN_API": "true",
            "AUTO_SYNC_PROVIDER_MODELS": "false"})
        self.env.start()
        self.client = TestClient(app).__enter__()
        configure_demo_engine(app.state.router)
        self.admin = {"Authorization": "Bearer test-admin"}
        self.config = {"rsl_enabled": True, "rsl_min_samples": 1, "rsl_judge_model": "judge",
                       "recommendation_excluded_models": ["judge"]}
        result = self.client.post("/admin/routing-profiles", headers=self.admin,
                                  json={"name": "test", "config_json": json.dumps(self.config)})
        self.assertEqual(result.status_code, 200, result.text)
        self.pid = str(result.json()["id"])
        self.headers = {"Authorization": "Bearer test-runtime", "X-Routing-Profile-Id": self.pid}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.env.stop()
        self.tmp.cleanup()

    def update(self, **kwargs):
        self.config.update(kwargs)
        result = self.client.put(f"/admin/routing-profiles/{self.pid}", headers=self.admin,
                                json={"config_json": json.dumps(self.config)})
        self.assertEqual(result.status_code, 200)

    def chat(self, **kwargs):
        return self.client.post("/v1/chat/completions", headers=self.headers,
            json={"model": "auto", "messages": [{"role": "user", "content": "计算 2+2"}], **kwargs})

    def test_full_learning_loop_and_actual_spend(self):
        first = self.chat(); self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["actual_model"], "budget")
        for _ in range(4):
            last = self.chat()
            self.assertEqual(last.status_code, 200, last.text)
        self.assertEqual(last.json()["actual_model"], "strong")
        metadata = first.json()["rsl"]
        self.assertTrue(metadata["cost_complete"])
        self.assertAlmostEqual(metadata["known_total_cost"], 0.000003 + 0.000015)

    def test_preview_is_read_only_and_matches_next_choice(self):
        preview = self.client.post("/admin/rsl/preview", headers=self.admin,
            json={"routing_profile_id": self.pid, "messages": [{"role": "user", "content": "计算 2+2"}]})
        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertEqual(app.state.rsl_store.stats(self.pid), [])
        self.assertEqual(preview.json()["selected_model"], self.chat().json()["actual_model"])

    def test_allowlist_limits_routing_and_judge(self):
        self.headers["X-Allowed-Models"] = "strong,judge"
        result = self.chat()
        self.assertEqual(result.json()["actual_model"], "strong")
        self.headers["X-Allowed-Models"] = "strong"
        self.assertEqual(self.chat().status_code, 403)

    def test_no_candidates_returns_503(self):
        self.headers["X-Allowed-Models"] = "missing"
        self.assertEqual(self.chat().status_code, 503)

    def test_human_correction_and_unknown_attempt(self):
        attempt = self.chat().json()["rsl"]["attempts"][0]
        payload = {"attempt_id": attempt["id"], "score": 0.8, "reason": "human correction"}
        for _ in range(2):
            self.assertEqual(self.client.post("/admin/rsl/feedback", headers=self.admin, json=payload).status_code, 200)
        row = app.state.rsl_store.stats(self.pid)[0]
        self.assertEqual(row["samples"], 1)
        self.assertEqual(row["quality"], 0.8)
        payload["attempt_id"] = "unknown"
        self.assertEqual(self.client.post("/admin/rsl/feedback", headers=self.admin, json=payload).status_code, 404)

    def test_admin_and_runtime_auth_required(self):
        self.assertEqual(self.client.get("/admin/rsl/stats").status_code, 401)
        self.assertEqual(self.client.post("/v1/chat/completions", json={"model": "auto", "messages": []}).status_code, 401)

    def test_no_judge_leaves_pending(self):
        self.update(rsl_judge_model=None)
        result = self.chat().json()
        self.assertIsNone(result["rsl"]["attempts"][0]["score"])
        self.assertEqual(app.state.rsl_store.stats(self.pid)[0]["samples"], 0)

    def test_invalid_judge_never_rewards_answer(self):
        class InvalidJudge(DemoProvider):
            async def chat_completions(self, body, config):
                result = await super().chat_completions(body, config)
                if config.name == "judge":
                    result.choices[0].message.content = '{"score": NaN}'
                return result
        app.state.router.providers["demo"] = InvalidJudge()
        result = self.chat().json()
        self.assertIsNone(result["rsl"]["attempts"][0]["score"])

    def test_cascade_records_each_attempt_and_total_cost(self):
        result = self.chat(cascade={"enabled": True, "chain": ["budget", "strong"], "min_content_length": 1})
        self.assertEqual(result.status_code, 200, result.text)
        data = result.json()
        self.assertEqual(data["actual_model"], "strong")
        self.assertEqual([a["score"] for a in data["rsl"]["attempts"]], [0, 1])
        self.assertAlmostEqual(data["rsl"]["known_total_cost"], 0.000003 + 0.000030 + 2 * 0.000015)

    def test_buffered_stream_finishes_and_learns_once(self):
        result = self.chat(stream=True)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertIn("data: [DONE]", result.text)
        events = [json.loads(line[6:]) for line in result.text.splitlines() if line.startswith("data: {")]
        self.assertEqual(events[-1]["rsl"]["stream_mode"], "buffered")
        self.assertEqual(app.state.rsl_store.stats(self.pid)[0]["attempts"], 1)

    def test_provider_failure_records_zero_and_falls_back(self):
        class Failing(DemoProvider):
            async def chat_completions(self, body, config):
                if config.name == "budget":
                    raise RuntimeError("simulated failure")
                return await super().chat_completions(body, config)
        app.state.router.providers["demo"] = Failing()
        result = self.chat(cascade={"enabled": True, "chain": ["budget", "strong"], "min_content_length": 1})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["actual_model"], "strong")
        self.assertEqual(result.json()["rsl"]["attempts"][0]["source"], "provider_error")
        self.assertFalse(result.json()["rsl"]["cost_complete"])

    def test_all_failures_return_error_and_record_attempts(self):
        class Failing(DemoProvider):
            async def chat_completions(self, body, config):
                raise RuntimeError("simulated failure")
        app.state.router.providers["demo"] = Failing()
        result = self.chat()
        self.assertEqual(result.status_code, 502, result.text)
        self.assertEqual(app.state.rsl_store.stats(self.pid)[0]["quality"], 0)

    def test_disabled_rsl_and_explicit_model_use_original_route(self):
        self.update(rsl_enabled=False, default_model="strong", recommendation_enabled=False)
        result = self.chat()
        self.assertEqual(result.status_code, 200, result.text)
        self.assertIsNone(result.json().get("rsl"))
        self.assertEqual(app.state.rsl_store.stats(self.pid), [])
        self.update(rsl_enabled=True)
        result = self.chat(model="strong")
        self.assertEqual(result.status_code, 200, result.text)
        self.assertIsNone(result.json().get("rsl"))

    def test_profile_rejects_invalid_learning_settings(self):
        for config in ('[]', '{"rsl_cost_scale": 0}', '{"rsl_exploration_weight": -1}', 'not json'):
            result = self.client.post("/admin/routing-profiles", headers=self.admin,
                                      json={"name": "bad", "config_json": config})
            self.assertEqual(result.status_code, 422, result.text)

    def test_reused_trace_id_still_records_distinct_billable_attempts(self):
        self.headers["X-Request-Id"] = "same-trace"
        first = self.chat().json()["rsl"]["attempts"][0]["id"]
        second = self.chat().json()["rsl"]["attempts"][0]["id"]
        self.assertNotEqual(first, second)
        self.assertEqual(sum(r["attempts"] for r in app.state.rsl_store.stats(self.pid)), 2)

    def test_missing_usage_keeps_cost_unknown(self):
        class NoUsage(DemoProvider):
            async def chat_completions(self, body, config):
                response = await super().chat_completions(body, config)
                response.usage = None
                return response
        app.state.router.providers["demo"] = NoUsage()
        result = self.chat().json()["rsl"]
        self.assertFalse(result["cost_complete"])
        self.assertIsNone(result["attempts"][0]["cost"])

    def test_returned_model_tracks_actual_answer_after_later_failure(self):
        class LaterFailure(DemoProvider):
            async def chat_completions(self, body, config):
                if config.name == "strong":
                    raise RuntimeError("later model failed")
                return await super().chat_completions(body, config)
        app.state.router.providers["demo"] = LaterFailure()
        data = self.chat(cascade={"enabled": True, "chain": ["budget", "strong"], "min_content_length": 1}).json()
        self.assertEqual(data["actual_model"], "budget")
        self.assertEqual(data["cascade"]["final_model"], "budget")
        self.assertFalse(data["cascade"]["validation_passed"])

    def test_real_evaluation_driver_with_local_provider(self):
        from experiments.evaluate_real import main
        self.update(rsl_judge_model=None, keyword_routing_enabled=False,
                    semantic_routing_enabled=False, cascade_enabled=False)
        train = Path(self.tmp.name) / "train.jsonl"
        test = Path(self.tmp.name) / "test.jsonl"
        output = Path(self.tmp.name) / "result.json"
        train.write_text('\n'.join(json.dumps({"id": f"train-{i}", "prompt": f"计算 2+2 case {i}", "expected": "4"}) for i in range(2)), encoding="utf-8")
        test.write_text(json.dumps({"id": "test-0", "prompt": "计算 2+2 test", "expected": "4"}), encoding="utf-8")

        class LocalClient:
            def __enter__(inner):
                return self.client
            def __exit__(inner, *args):
                pass

        argv = ["evaluate_real", "--profile", self.pid, "--train", str(train), "--test", str(test), "--output", str(output)]
        with patch("sys.argv", argv), patch("experiments.evaluate_real.httpx.Client", return_value=LocalClient()):
            main()
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["summary"]["rsl_test"]["accuracy"], 1)
        self.assertEqual(sum(r["attempts"] for r in app.state.rsl_store.stats(self.pid)), 2)


if __name__ == "__main__":
    unittest.main()

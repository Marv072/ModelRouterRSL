"""Budget-limited REAL experiment through the app's HTTP endpoints.

48 online training calls + both models on 24 held-out cases = 96 calls.
Frozen RSL is evaluated by selecting its model's measured held-out response.
Run once; never automatically resumes/repeats paid calls after a failure.
"""
import argparse
import asyncio
import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

from dotenv import dotenv_values
from fastapi.testclient import TestClient

from app.core.config import ApiKeyConfig, ModelConfig, ModelPricing, ProviderConfig
from app.core.rsl import classify
from app.providers.base import KeyPool
from app.providers.openai import OpenAIProvider
from experiments.deepseek_cases import make_cases, grade

POLICY = dict(rsl_enabled=True, rsl_judge_model=None, rsl_min_samples=3,
    rsl_quality_weight=1.0, rsl_cost_weight=0.2, rsl_latency_weight=0.1,
    rsl_exploration_weight=0.15, rsl_cost_scale=0.005, rsl_latency_scale_ms=5000,
    allowed_models=["budget", "strong"], recommendation_enabled=False,
    recommendation_apply_to_explicit=False, keyword_routing_enabled=False,
    semantic_routing_enabled=False, cascade_enabled=False)
MODELS = {"budget": ("deepseek-flash", 2.0, 8.0), "strong": ("deepseek-v4-pro", 9.0, 27.0)}


class BudgetProvider(OpenAIProvider):
    def __init__(self, config, ledger):
        super().__init__(config, KeyPool(config.keys))
        self.ledger = Path(ledger)
        self.observations = []

    async def chat_completions(self, request, model_config):
        if self.config.max_retries != 0 or len(self.key_pool.keys) != 1:
            raise ValueError("Budget runner requires no retries and one key")
        if model_config.name not in {v[0] for v in MODELS.values()}:
            raise ValueError("Unexpected model")
        if not request.thinking or request.thinking.type != "disabled":
            raise ValueError("Thinking must be explicitly disabled")
        if request.stream or not request.max_tokens or not 1 <= request.max_tokens <= 256:
            raise ValueError("Output limit required")
        if len(request.messages) != 1 or request.messages[0].role != "user" or len(request.messages[0].content.encode("utf-8")) > 1800:
            raise ValueError("Input exceeds prepaid bound")
        ledger = json.loads(self.ledger.read_text(encoding="utf-8"))
        if ledger["reserved_cny"] + 0.03 > min(5.0, ledger["budget_cny"]):
            raise ValueError("Budget exhausted; no request sent")
        ledger["reserved_cny"] = round(ledger["reserved_cny"] + 0.03, 4)
        entry = dict(model=model_config.name, reserved_cny=0.03, status="reserved",
                     timestamp_utc=datetime.now(timezone.utc).isoformat())
        ledger["calls"].append(entry)
        self.ledger.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        tick = time.monotonic()
        observation = dict(model=model_config.name)
        try:
            response = await super().chat_completions(request, model_config)
            usage = response.usage
            observation["usage"] = usage.model_dump() if usage else None
            observation["http_status"] = 200
            entry["status"] = "200"
            return response
        except Exception as error:
            observation["error_type"] = type(error).__name__
            entry["status"] = "failed_no_retry"
            raise
        finally:
            observation["provider_latency_ms"] = (time.monotonic() - tick) * 1000
            self.observations.append(observation)
            self.ledger.write_text(json.dumps(ledger, indent=2), encoding="utf-8")


def upper_cost(usage, route):
    if not usage:
        return None
    _, p, c = MODELS[route]
    return (usage["prompt_tokens"] * p + usage["completion_tokens"] * c) / 1_000_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Authorize this run's paid calls")
    parser.add_argument("--output", default="runs/deepseek-real-v1")
    args = parser.parse_args()
    train, test = make_cases(6023, 16, "train"), make_cases(6024, 8, "test")
    assert not {c["prompt"] for c in train} & {c["prompt"] for c in test}
    assert all(classify([{"role": "user", "content": c["prompt"]}]) == c["category"] for c in train + test)
    assert all(len(c["prompt"].encode()) <= 1800 for c in train + test)
    manifest = dict(train_seed=6023, test_seed=6024, training_cases=48, held_out_cases=24,
        paid_call_limit=96, max_output_tokens=256, thinking="disabled", policy=POLICY,
        scoring="computed integer / JSON reference; no model judge", test_method="paired frozen-policy replay",
        price_basis="2026-10-23 CNY peak cache-miss; conservative upper estimate",
        price_source="https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
        max_reserved_cny_this_run=2.88, models=MODELS,
        data_sha256=hashlib.sha256(json.dumps([train,test],sort_keys=True).encode()).hexdigest())
    if not args.execute:
        print(json.dumps(manifest, indent=2)); return
    env = dotenv_values(".env")
    base = (env.get("OPENAI_BASE_URL") or "").rstrip("/")
    url = urlsplit(base)
    if url.scheme != "https" or url.hostname != "api.deepseek.com":
        raise ValueError("Expected official DeepSeek HTTPS endpoint")
    if not all(env.get(k) for k in ("OPENAI_KEY_1", "ADMIN_API_KEY", "ROUTER_INTERNAL_API_KEY")):
        raise ValueError("Missing credentials in .env")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    ledger = Path("runs/deepseek-budget.json")
    if not ledger.exists():
        ledger.write_text(json.dumps(dict(budget_cny=5.0,reserved_cny=0.0,calls=[])), encoding="utf-8")
    prior = json.loads(ledger.read_text(encoding="utf-8"))
    if prior["reserved_cny"] + 2.88 > min(5.0, prior["budget_cny"]):
        raise ValueError("Insufficient remaining reservation for full experiment")
    # Exclusive runner lock; do not run another billable script concurrently.
    lock = Path("runs/deepseek-experiment.lock")
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        run(output, ledger, env, base, manifest, train, test)
    finally:
        lock.unlink()


def run(output, ledger, env, base, manifest, train, test):
    from app.main import app
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "dataset.json").write_text(json.dumps(dict(train=train, test=test), indent=2), encoding="utf-8")
    empty_config = output / "empty.yaml"
    empty_config.write_text("providers: {}\n", encoding="utf-8")
    settings = dict(ROUTER_CONFIG=str(empty_config), ROUTER_DB=str(output / "router.db"),
        RSL_DB=str(output / "rsl.db"), ROUTER_RUNTIME_MODE="internal", ROUTER_ENABLE_ADMIN_API="true",
        ROUTER_ENABLE_STANDALONE_KEYS="false", AUTO_SYNC_PROVIDER_MODELS="false",
        ADMIN_API_KEY=env["ADMIN_API_KEY"], ROUTER_INTERNAL_API_KEY=env["ROUTER_INTERNAL_API_KEY"])
    results = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                   manifest=manifest, records=[], training_checkpoints=[], frozen_routes={})
    def save():
        (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with patch.dict(os.environ, settings), TestClient(app) as client:
        provider = BudgetProvider(ProviderConfig(base_url=base, timeout=60, max_retries=0,
            keys=[ApiKeyConfig(key=env["OPENAI_KEY_1"], rpm=0)], models=[]), ledger)
        app.state.router.providers = {"openai": provider}
        app.state.router.model_map = {r:("openai", ModelConfig(name=name, routing_keys=[r],
            pricing=ModelPricing(prompt=p, completion=c))) for r,(name,p,c) in MODELS.items()}
        admin = {"Authorization": "Bearer " + env["ADMIN_API_KEY"]}
        result = client.post("/admin/routing-profiles", headers=admin,
            json={"name":"deepseek-real-v1", "config_json":json.dumps(POLICY)})
        result.raise_for_status()
        pid = str(result.json()["id"])
        runtime = {"Authorization":"Bearer "+env["ROUTER_INTERNAL_API_KEY"], "X-Routing-Profile-Id":pid}

        def preview(case):
            r = client.post("/admin/rsl/preview", headers=admin, json={"routing_profile_id":pid,
                "messages":[{"role":"user","content":case["prompt"]}], "explore":False})
            r.raise_for_status(); return r.json()

        def invoke(case, route, phase):
            tick = time.monotonic()
            r = client.post("/v1/chat/completions", headers=runtime, json={"model":route,
                "messages":[{"role":"user","content":case["prompt"]}], "temperature":0,
                "thinking":{"type":"disabled"}, "max_tokens":256, "cascade":{"enabled":False}})
            if r.status_code != 200:
                results["status"] = "failed_http_" + str(r.status_code)
                results["failed_case"] = case["id"]
                results["provider_observations"] = provider.observations
                save()
                raise RuntimeError("Request failed; stopped without retries. See local results.")
            data = r.json()
            actual = data.get("actual_model") or route
            answer = data["choices"][0]["message"]["content"]
            row = dict(case_id=case["id"], category=case["category"], phase=phase, route=actual,
                answer=answer, expected=case["expected"], score=grade(answer,case), usage=data.get("usage"),
                finish_reason=data["choices"][0].get("finish_reason"),
                latency_ms=(time.monotonic()-tick)*1000,
                provider_latency_ms=provider.observations[-1]["provider_latency_ms"],
                cost_upper_cny=upper_cost(data.get("usage"),actual))
            if data.get("rsl"):
                row["decision"] = {k:data["rsl"][k] for k in ("selected_model","reason","learning_samples","candidates")}
            results["records"].append(row); save()
            return row,data

        for i,case in enumerate(train):
            row,data = invoke(case,"auto","train")
            feedback = client.post("/admin/rsl/feedback", headers=admin,
                json={"attempt_id":data["rsl"]["attempts"][-1]["id"], "score":row["score"],
                      "reason":"computed reference: "+case["grader"]})
            feedback.raise_for_status()
            if (i+1)%12 == 0:
                results["training_checkpoints"].append(dict(round=i+1, stats=app.state.rsl_store.stats(pid)))
                save(); print(f"Training {i+1}/48 complete",flush=True)
        before = app.state.rsl_store.stats(pid)
        # Freeze all choices before making any held-out completion call.
        for case in test:
            results["frozen_routes"][case["id"]] = preview(case)
        save()
        for i,case in enumerate(test):
            for route in (("budget","strong") if i%2==0 else ("strong","budget")):
                invoke(case,route,"test")
            if (i+1)%6 == 0:
                print(f"Paired test {i+1}/24 complete",flush=True)
        if before != app.state.rsl_store.stats(pid):
            raise RuntimeError("Held-out calls changed training statistics")
        results.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
            final_stats=before, provider_observations=provider.observations,
            budget_reserved_cny_total=json.loads(ledger.read_text(encoding="utf-8"))["reserved_cny"])
        save()
    print("Completed real experiment: " + str(output / "results.json"),flush=True)


if __name__ == "__main__":
    main()

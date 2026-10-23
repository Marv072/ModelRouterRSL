"""Exercise real HTTP routes and SQLite using a local deterministic provider.

Run python -m examples.offline_demo. No keys, sockets or billable calls.
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.config import ModelConfig, ModelPricing
from app.schemas.chat import ChatResponse, ChatChoice, ChatMessage, Usage


class DemoProvider:
    async def chat_completions(self, body, config):
        if config.name == "judge":
            payload = json.loads(body.messages[-1].content)
            score = 1.0 if payload["answer"] == "4" else 0.0
            content = json.dumps({"score": score, "reason": "Demo exact-answer check: expected 4"})
        else:
            content = "4" if config.name == "strong" else "5"
        return ChatResponse(id="demo-response", created=0, model=config.name,
            choices=[ChatChoice(message=ChatMessage(role="assistant", content=content), finish_reason="stop")],
            usage=Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30))


def configure_demo_engine(engine):
    engine.providers = {"demo": DemoProvider()}
    engine.model_map = {
        name: ("demo", ModelConfig(name=name, routing_keys=[name],
               pricing=ModelPricing(prompt=price, completion=price)))
        for name, price in (("budget", 0.1), ("strong", 1.0), ("judge", 0.5))
    }


def main():
    from app.main import app
    Path(".demo").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=".demo") as tmp:
        config = Path(tmp) / "providers.yaml"
        config.write_text("providers: {}\n", encoding="utf-8")
        env = {"ROUTER_CONFIG": str(config), "ROUTER_DB": str(Path(tmp) / "router.db"),
               "RSL_DB": str(Path(tmp) / "rsl.db"), "ROUTER_RUNTIME_MODE": "internal",
               "ROUTER_INTERNAL_API_KEY": "demo-internal", "ADMIN_API_KEY": "demo-admin",
               "ROUTER_ENABLE_STANDALONE_KEYS": "false", "ROUTER_ENABLE_ADMIN_API": "true"}
        with patch.dict(os.environ, env), TestClient(app) as client:
            configure_demo_engine(app.state.router)
            admin = {"Authorization": "Bearer demo-admin"}
            profile = client.post("/admin/routing-profiles", headers=admin, json={
                "name": "demo", "config_json": json.dumps({"rsl_enabled": True,
                "rsl_judge_model": "judge", "rsl_min_samples": 1,
                "recommendation_excluded_models": ["judge"]})})
            profile.raise_for_status()
            pid = str(profile.json()["id"])
            headers = {"Authorization": "Bearer demo-internal", "X-Routing-Profile-Id": pid}
            for i in range(6):
                result = client.post("/v1/chat/completions", headers=headers,
                    json={"model": "auto", "messages": [{"role": "user", "content": "计算 2+2，只返回数字"}]})
                result.raise_for_status()
                data = result.json()
                print(f"Round {i+1}: {data['actual_model']}, score={data['rsl']['attempts'][0]['score']}, reason={data['rsl']['reason']}")
            stats = client.get("/admin/rsl/stats", headers=admin, params={"routing_profile_id": pid})
            stats.raise_for_status()
            print(json.dumps(stats.json(), indent=2, ensure_ascii=False))
            assert data["actual_model"] == "strong", "Expected feedback to favor the correct demo model"
    print("Offline HTTP demo passed. This validates the loop, not real LLM quality.")


if __name__ == "__main__":
    main()

"""Optional billable benchmark against a running local ModelRouter service.

Train: auto routing followed by exact-answer feedback. Test: frozen preview
followed by explicit calls, so held-out scores never update the RSL ledger.
"""
import argparse
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv


def exact_score(answer, expected):
    return float(answer.strip() == str(expected).strip())


def load_cases(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Dataset must be nonempty with unique IDs")
    for row in rows:
        if not isinstance(row.get("prompt"), str) or "expected" not in row:
            raise ValueError("Each case requires id, prompt and expected")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--baseline", default="budget")
    parser.add_argument("--train", default="experiments/data/train.jsonl")
    parser.add_argument("--test", default="experiments/data/test.jsonl")
    parser.add_argument("--output", default="runs/real-evaluation.json")
    args = parser.parse_args()
    load_dotenv()
    runtime_key, admin_key = os.environ["ROUTER_INTERNAL_API_KEY"], os.environ["ADMIN_API_KEY"]
    train, test = load_cases(args.train), load_cases(args.test)
    if {r["id"] for r in train} & {r["id"] for r in test} or {r["prompt"] for r in train} & {r["prompt"] for r in test}:
        raise ValueError("Training and test datasets must be disjoint")
    runtime = {"Authorization": "Bearer " + runtime_key, "X-Routing-Profile-Id": args.profile}
    admin = {"Authorization": "Bearer " + admin_key}
    output = Path(args.output)
    if output.exists():
        raise ValueError("Output exists; select a new --output path to preserve earlier runs")
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []

    with httpx.Client(base_url=args.base_url, timeout=180) as client:
        def call(method, path, **kwargs):
            response = client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()

        profile = call("GET", f"/admin/routing-profiles/{args.profile}", headers=admin)
        config = json.loads(profile["config_json"])
        if not config.get("rsl_enabled") or config.get("rsl_judge_model") or config.get("cascade_enabled") or config.get("cascade_override_chain"):
            raise ValueError("Use a dedicated RSL profile with no judge or cascade for exact-answer evaluation")
        if config.get("recommendation_apply_to_explicit") or config.get("keyword_routing_enabled") is not False or config.get("semantic_routing_enabled") is not False:
            raise ValueError("Disable keyword/semantic routing and explicit-model recommendation for frozen test calls")
        if call("GET", "/admin/rsl/stats", headers=admin, params={"routing_profile_id": args.profile})["data"]:
            raise ValueError("Use a fresh profile to prevent prior-data contamination")
        model_response = call("GET", "/admin/models", headers=admin)
        model_rows = model_response if isinstance(model_response, list) else model_response.get("data", model_response.get("models", []))
        prices = {r["routing_key"]: (r["prompt_price"], r["completion_price"]) for r in model_rows}

        def request_case(case, model, phase):
            tick = time.monotonic()
            data = call("POST", "/v1/chat/completions", headers=runtime, json={"model": model,
                "messages": [{"role": "user", "content": case["prompt"]}], "temperature": 0,
                "max_tokens": 128, "cascade": {"enabled": False}})
            latency = (time.monotonic() - tick) * 1000
            score = exact_score(data["choices"][0]["message"]["content"], case["expected"])
            actual = data.get("actual_model") or model
            usage = data.get("usage")
            cost = None
            if actual in prices and usage:
                p, c = prices[actual]
                cost = (usage["prompt_tokens"] * p + usage["completion_tokens"] * c) / 1_000_000
            records.append(dict(id=case["id"], phase=phase, model=actual, score=score, cost=cost,
                                latency_ms=latency, answer=data["choices"][0]["message"]["content"]))
            output.write_text(json.dumps({"status": "in_progress", "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
            return data, score

        for case in train:
            data, score = request_case(case, "auto", "train")
            attempt = data["rsl"]["attempts"][-1]
            call("POST", "/admin/rsl/feedback", headers=admin,
                 json={"attempt_id": attempt["id"], "score": score, "reason": "exact-match reference answer"})
        stats_before = call("GET", "/admin/rsl/stats", headers=admin, params={"routing_profile_id": args.profile})
        for index, case in enumerate(test):
            preview = call("POST", "/admin/rsl/preview", headers=admin, json={"routing_profile_id": args.profile,
                "messages": [{"role": "user", "content": case["prompt"]}], "explore": False})
            order = [(args.baseline, "baseline_test"), (preview["selected_model"], "rsl_test")]
            for model, phase in order if index % 2 == 0 else reversed(order):
                request_case(case, model, phase)
        stats_after = call("GET", "/admin/rsl/stats", headers=admin, params={"routing_profile_id": args.profile})
        if stats_before != stats_after:
            raise RuntimeError("Test phase changed learning statistics; check profile configuration")
    summary = {}
    for phase in ("train", "baseline_test", "rsl_test"):
        rows = [r for r in records if r["phase"] == phase]
        summary[phase] = dict(count=len(rows), accuracy=sum(r["score"] for r in rows) / len(rows),
            known_cost=sum(r["cost"] or 0 for r in rows), unknown_cost_requests=sum(r["cost"] is None for r in rows),
            avg_latency_ms=sum(r["latency_ms"] for r in rows) / len(rows))
    output.write_text(json.dumps(dict(status="complete", profile_config=config, summary=summary, records=records),
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

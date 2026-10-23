"""Reproducible synthetic bandit experiment; never calls an external model."""
import argparse
import csv
import json
import random
import statistics
import tempfile
from collections import Counter
from pathlib import Path

from app.core.config import ModelConfig
from app.core.router import RouterEngine
from app.core.routing_policy import RoutingPolicy
from app.core.rsl import RSLStore, select_model

MODELS = ("budget", "specialist", "strong")
CATEGORIES = ("writing", "coding", "math")
QUALITY = {
    "writing": (0.95, 0.70, 0.97),
    "coding": (0.35, 0.95, 0.96),
    "math": (0.45, 0.65, 0.97),
}
COST = dict(zip(MODELS, (0.0005, 0.003, 0.01)))
LATENCY = dict(zip(MODELS, (200, 900, 3000)))


def utility(score, model, policy):
    return (policy.rsl_quality_weight * score
            - policy.rsl_cost_weight * min(COST[model] / policy.rsl_cost_scale, 1)
            - policy.rsl_latency_weight * min(LATENCY[model] / policy.rsl_latency_scale_ms, 1))


def workload(seed, count):
    rng = random.Random(seed)
    result = []
    for i in range(count):
        category = CATEGORIES[i % len(CATEGORIES)]
        outcomes = {m: float(rng.random() < QUALITY[category][j]) for j, m in enumerate(MODELS)}
        result.append((category, outcomes))
    rng.shuffle(result)
    return result


def run(output, seeds=5, train=600, test=300):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    policy = RoutingPolicy(rsl_enabled=True)
    runs, curves, selections = [], [], []
    # Execute the actual original resolver with a declared default-model configuration.
    baseline = RouterEngine(None)
    baseline.model_map = {m: ("synthetic", ModelConfig(name=m, routing_keys=[m])) for m in MODELS}
    original_choice = baseline.resolve_model_target("auto", default_model="budget")
    for seed in range(seeds):
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            store = RSLStore(str(Path(tmp) / "learning.db"))
            block = []
            training_cost = 0.0
            for i, (category, outcomes) in enumerate(workload(seed, train)):
                decision = select_model(store, COST, policy, "experiment", category)
                model = decision["selected_model"]
                score = outcomes[model]
                store.record(id=str(i), request_id=str(i), profile="experiment", category=category,
                    model=model, score=score, source="synthetic_ground_truth", reason="Bernoulli simulation",
                    cost=COST[model], latency_ms=LATENCY[model])
                training_cost += COST[model]
                block.append(utility(score, model, policy))
                if (i + 1) % 60 == 0 or i + 1 == train:
                    curves.append(dict(seed=seed, round=i+1, utility=statistics.mean(block),
                                       cumulative_training_cost=training_cost))
                    block = []
            frozen = {c: select_model(store, COST, policy, "experiment", c, explore=False)["selected_model"] for c in CATEGORIES}
            selections.append(dict(seed=seed, **frozen))
            for strategy in ("original_default", "always_strong", "round_robin", "rsl_frozen"):
                observed, spent, elapsed, rewards = [], [], [], []
                counts = Counter()
                for i, (category, outcomes) in enumerate(workload(10000 + seed, test)):
                    model = {"original_default": original_choice, "always_strong": "strong",
                             "round_robin": MODELS[i % len(MODELS)], "rsl_frozen": frozen[category]}[strategy]
                    observed.append(outcomes[model]); spent.append(COST[model]); elapsed.append(LATENCY[model])
                    rewards.append(utility(outcomes[model], model, policy)); counts[model] += 1
                runs.append(dict(seed=seed, strategy=strategy, quality=statistics.mean(observed),
                    avg_cost=statistics.mean(spent), avg_latency_ms=statistics.mean(elapsed),
                    utility=statistics.mean(rewards), test_cost=sum(spent),
                    training_cost=training_cost if strategy == "rsl_frozen" else 0,
                    selection_counts=dict(counts)))
    summaries = []
    for strategy in ("original_default", "always_strong", "round_robin", "rsl_frozen"):
        selected = [r for r in runs if r["strategy"] == strategy]
        summary = {"strategy": strategy}
        for metric in ("quality", "avg_cost", "avg_latency_ms", "utility", "training_cost", "test_cost"):
            values = [r[metric] for r in selected]
            summary[metric] = statistics.mean(values)
            summary[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else 0
        summaries.append(summary)
    result = dict(kind="SYNTHETIC_SIMULATION_NOT_REAL_LLM_EVALUATION", seeds=seeds,
        train_requests_per_seed=train, test_requests_per_seed=test,
        assumptions={"quality_probabilities": QUALITY, "cost": COST, "latency_ms": LATENCY,
                     "judge_cost": 0, "feedback": "perfect synthetic ground truth", "original_baseline": "default_model=budget; recommendation disabled"},
        policy=policy.model_dump(), summary=summaries, runs=runs, learned_routes=selections)
    (output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "learning_curve.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=curves[0].keys()); writer.writeheader(); writer.writerows(curves)
    lines = ["# 可复现模拟实验结果", "", "**这是合成环境实验，不是真实大模型效果测评。**", "",
        f"{seeds} 个随机种子；每个种子 {train} 条学习请求、{test} 条独立测试请求。测试时冻结统计并关闭探索。",
        "各策略使用相同测试任务和预生成结果；反馈为无费用的理想客观标签。± 为跨种子样本标准差。", "",
        "| 策略 | 质量均值 ± SD | 每请求成本 | 延迟 ms | 综合效用 ± SD | 学习总成本 | 测试总成本 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['strategy']} | {r['quality']:.3f} ± {r['quality_std']:.3f} | {r['avg_cost']:.5f} | {r['avg_latency_ms']:.0f} | {r['utility']:.3f} ± {r['utility_std']:.3f} | {r['training_cost']:.3f} | {r['test_cost']:.3f} |")
    lines += ["", "## 解读边界", "", "- 原版基线仅代表默认模型为 budget、关闭推荐的配置，不代表所有原版配置。",
        "- 强模型可能取得更高纯质量；RSL 优化的是明确权重下的综合效用。",
        "- 学习费用单独报告；短期部署不能忽略探索成本。",
        "- 成本、延迟及成功概率均人为设定；不支持任何真实供应商优劣结论。",
        "- 学习与测试使用不同种子；测试不回写学习数据。",
        "- 分类已知且反馈无噪声，真实场景中的分类误差、Judge 偏差及模型漂移尚未覆盖。", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/simulation")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--train", type=int, default=600)
    parser.add_argument("--test", type=int, default=300)
    args = parser.parse_args()
    if min(args.seeds, args.train, args.test) <= 0:
        parser.error("seeds, train and test must be positive")
    run(args.output, args.seeds, args.train, args.test)

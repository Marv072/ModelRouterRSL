"""Small, auditable task-conditioned bandit; no foundation-model training.

The ledger is the source of truth. Aggregating it avoids inconsistent cached
statistics and makes human corrections idempotent. Short SQLite transactions
are suitable for this single-machine coursework implementation.
"""
from __future__ import annotations

import math
import sqlite3
from contextlib import closing
from pathlib import Path


class RSLStore:
    def __init__(self, path: str):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as con, con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript("""
                CREATE TABLE IF NOT EXISTS rsl_attempts (
                    id TEXT PRIMARY KEY, request_id TEXT NOT NULL,
                    profile TEXT NOT NULL, category TEXT NOT NULL, model TEXT NOT NULL,
                    score REAL CHECK(score BETWEEN 0 AND 1), source TEXT NOT NULL,
                    reason TEXT NOT NULL, cost REAL, latency_ms REAL NOT NULL,
                    judge_cost REAL, judge_latency_ms REAL NOT NULL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS rsl_scope ON rsl_attempts(profile, category, model);
                CREATE TABLE IF NOT EXISTS rsl_corrections (
                    attempt_id TEXT PRIMARY KEY REFERENCES rsl_attempts(id),
                    score REAL NOT NULL CHECK(score BETWEEN 0 AND 1), reason TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)

    def connect(self):
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def record(self, *, id, request_id, profile, category, model, score, source,
               reason, cost, latency_ms, judge_cost=0.0, judge_latency_ms=0.0):
        if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("score must be finite and in [0, 1]")
        for value in (cost, latency_ms, judge_cost, judge_latency_ms):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("cost and latency must be finite and nonnegative")
        values = (id, request_id, str(profile or ""), category, model, score, source,
                  reason, cost, latency_ms, judge_cost, judge_latency_ms)
        with closing(self.connect()) as con, con:
            con.execute("INSERT INTO rsl_attempts "
                        "(id,request_id,profile,category,model,score,source,reason,cost,latency_ms,judge_cost,judge_latency_ms) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING", values)

    def correct(self, attempt_id, score, reason):
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("score must be finite and in [0, 1]")
        with closing(self.connect()) as con, con:
            if not con.execute("SELECT 1 FROM rsl_attempts WHERE id=?", (attempt_id,)).fetchone():
                raise KeyError(attempt_id)
            con.execute("INSERT INTO rsl_corrections(attempt_id,score,reason) VALUES (?,?,?) "
                        "ON CONFLICT(attempt_id) DO UPDATE SET score=excluded.score, "
                        "reason=excluded.reason, updated_at=CURRENT_TIMESTAMP", (attempt_id, score, reason))

    def stats(self, profile="", category=None):
        query = """SELECT a.category,a.model,COUNT(*) AS attempts,
                   COUNT(COALESCE(c.score,a.score)) AS samples,
                   AVG(COALESCE(c.score,a.score)) AS quality,
                   AVG(a.cost) AS avg_cost, COUNT(a.cost) AS cost_samples,
                   AVG(a.latency_ms) AS avg_latency_ms,
                   SUM(a.cost) AS generation_cost, SUM(a.judge_cost) AS judge_cost,
                   SUM(CASE WHEN a.cost IS NULL OR a.judge_cost IS NULL THEN 1 ELSE 0 END) AS unknown_cost_attempts
                   FROM rsl_attempts a LEFT JOIN rsl_corrections c ON c.attempt_id=a.id
                   WHERE a.profile=?"""
        params = [str(profile or "")]
        if category is not None:
            query += " AND a.category=?"
            params.append(category)
        query += " GROUP BY a.category,a.model ORDER BY a.category,a.model"
        with closing(self.connect()) as con:
            return [dict(r) for r in con.execute(query, params)]

    def feedback(self, profile="", limit=100):
        with closing(self.connect()) as con:
            return [dict(r) for r in con.execute(
                "SELECT a.*,c.score AS human_score,c.reason AS human_reason "
                "FROM rsl_attempts a LEFT JOIN rsl_corrections c ON c.attempt_id=a.id "
                "WHERE a.profile=? ORDER BY a.rowid DESC LIMIT ?", (str(profile or ""), limit))]


def classify(messages):
    text = " ".join(m.content if hasattr(m, "content") else m.get("content", "")
                    for m in messages if (m.role if hasattr(m, "role") else m.get("role")) == "user").lower()
    for category, words in (
        ("coding", ("python", "function", "代码", "编程", "javascript", "sql", "函数")),
        ("math", ("计算", "方程", "数学", "calculate", "equation", "arithmetic")),
        ("translation", ("翻译", "translate", "translation")),
        ("writing", ("写作", "作文", "文章", "邮件", "essay", "write an", "写一")),
    ):
        if any(w in text for w in words):
            return category
    return "general"


def select_model(store, candidates, policy, profile, category, *, explore=True):
    """candidates: model -> cost estimate in currency/request.

    Fixed cost/latency scales keep scores comparable as candidates change.
    Cold-start minimum is enforced, not merely encouraged by a small bonus.
    """
    if not candidates:
        raise ValueError("No available RSL candidates")
    stats = {r["model"]: r for r in store.stats(profile, category)}
    total = sum(stats.get(m, {}).get("samples", 0) for m in candidates)
    ranked = []
    for model, prior_cost in sorted(candidates.items()):
        row = stats.get(model, {})
        n = row.get("samples", 0)
        quality = row.get("quality") if n else 0.5
        cost = row.get("avg_cost")
        cost = prior_cost if cost is None else cost
        latency = row.get("avg_latency_ms", policy.rsl_latency_scale_ms)
        cost_penalty = min(cost / policy.rsl_cost_scale, 1) if cost is not None else 1.0
        latency_penalty = min(latency / policy.rsl_latency_scale_ms, 1)
        bonus = policy.rsl_exploration_weight * math.sqrt(2 * math.log(total + 2) / (n + 1)) if explore else 0
        utility = (policy.rsl_quality_weight * quality - policy.rsl_cost_weight * cost_penalty
                   - policy.rsl_latency_weight * latency_penalty)
        ranked.append(dict(model=model, samples=n, attempts=row.get("attempts", 0), quality=quality,
                           estimated_cost=cost, latency_ms=latency, utility=utility,
                           exploration_bonus=bonus, score=utility + bonus))
    cold = [r for r in ranked if r["attempts"] < policy.rsl_min_samples] if explore else []
    ranked.sort(key=lambda r: (-r["score"], r["model"]))
    winner = min(cold, key=lambda r: (r["attempts"], -r["score"], r["model"])) if cold else ranked[0]
    return dict(selected_model=winner["model"], task_category=category,
                reason="cold_start" if cold else "ucb" if explore else "frozen_policy",
                learning_samples=total, candidates=ranked, algorithm="task-ucb-v1")

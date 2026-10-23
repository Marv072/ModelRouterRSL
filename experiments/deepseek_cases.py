"""Independent generated tasks and computed references; no LLM-generated labels."""
import json
import random


def make_cases(seed, per_category, split):
    rng = random.Random(seed)
    cases = []
    for index in range(per_category):
        a, b = rng.randrange(100, 10000), rng.randrange(100, 10000)
        c, modulus = rng.randrange(100, 9999), rng.choice([97, 101, 113, 127])
        prompt = f"Calculate ({a} * {b} + {c}) modulo {modulus}. Return only the integer, no explanation."
        cases.append(dict(id=f"{split}-math-{index}", category="math", prompt=prompt,
                          expected=(a*b+c) % modulus, grader="integer"))
        start, steps = rng.randrange(1, 20), rng.randrange(9, 21)
        multiplier, offset, mod = rng.randrange(2, 8), rng.randrange(1, 11), rng.randrange(31, 100)
        x, total = start, 0
        for i in range(steps):
            x = (x * multiplier + i + offset) % mod
            if x % 3 != 0:
                total += x
        code = (f"x = {start}\ntotal = 0\nfor i in range({steps}):\n"
                f"    x = (x * {multiplier} + i + {offset}) % {mod}\n"
                "    if x % 3 != 0:\n        total += x\nprint(total)")
        cases.append(dict(id=f"{split}-coding-{index}", category="coding",
            prompt="What does this Python code print? Return only the integer, no explanation.\n"+code,
            expected=total, grader="integer"))
        rows = [dict(id=f"R{j}", active=rng.choice([True, True, False]), points=rng.randrange(0, 40),
                     age=rng.randrange(18, 65)) for j in range(10)]
        age = rng.randrange(25, 45)
        eligible = sorted([r for r in rows if r["active"] and r["age"] >= age],
                          key=lambda r: (-r["points"], r["id"]))[:3]
        cases.append(dict(id=f"{split}-general-{index}", category="general", grader="json",
            prompt=(f"Keep records with active=true and age>={age}. Sort by points descending, then id ascending. "
                    "Take the first 3 (or all if fewer). Return only a JSON array of their id strings, no markdown.\n"
                    + json.dumps(rows, separators=(",", ":"))), expected=[r["id"] for r in eligible]))
    rng.shuffle(cases)
    return cases


def grade(answer, case):
    text = answer.strip()
    try:
        if case["grader"] == "integer":
            # Exact value, allowing whitespace and a leading sign, not extra prose.
            if not text or any(c not in "+-0123456789" for c in text):
                return 0.0
            return float(int(text) == case["expected"])
        return float(json.loads(text) == case["expected"])
    except (ValueError, TypeError):
        return 0.0

"""Frozen paired position experiment; no inference or output-dependent selection."""
from __future__ import annotations

import random
import math
from protocol import digest, minima, one_shot_payload, visible


def rng(*parts):
    return random.Random(int(digest(parts), 16))


def select_tasks(tasks, scope):
    if scope == "full":
        return tasks
    double = [t for t in tasks if len(minima(t)) == 2]
    # Pre-output quality controls, stratified by feasible-list length.
    unique = sorted([t for t in tasks if len(minima(t)) == 1],
                    key=lambda t: (len(t["candidate_options"]), t["task_id"]))
    controls = [unique[round(i * (len(unique) - 1) / 11)] for i in range(12)]
    chosen = {t["task_id"] for t in double + controls}
    return [t for t in tasks if t["task_id"] in chosen]


def conditions(config):
    return ["baseline"] + [f"{split}_{r:02d}_{sign}"
                            for split, n in (("cal", config["calibration_repeats"]),
                                             ("test", config["test_repeats"]))
                            for r in range(n) for sign in ("plus", "minus")]


def indices(task, arm, config):
    n = len(task["candidate_options"])
    if arm == "baseline":
        return list(range(n))
    split, repeat, sign = arm.split("_")
    opts = task["candidate_options"]
    minimum = {o["start_time_utc"] for o in minima(task)}
    targets = [i for i, o in enumerate(opts) if o["start_time_utc"] in minimum]
    # Earliest and latest exact optima; for unique-optimum quality controls,
    # swap the unique minimum with a fixed nonoptimal candidate instead.
    left, right = targets[0], targets[-1]
    if len(targets) == 1 and n > 1:
        right = next(i for i in range(n) if i != left)
    if left == right:
        return list(range(n))
    ordinal = int(repeat) + (config["calibration_repeats"] if split == "test" else 0)
    # Distinct paired backgrounds when the finite permutation space permits it.
    # n=2 has only one paired background; n=3 only three. Reuse is disclosed.
    space = math.factorial(n) // 2 if n < 10 else 1000000
    seen = set()
    for index in range(ordinal + 1):
        if len(seen) == space:
            seen.clear()
        attempt = 0
        while True:
            randomizer = rng(config["shuffle_seed"], task["task_id"], "background", index, attempt)
            positions = sorted(randomizer.sample(range(n), 2))
            rest = [i for i in range(n) if i not in (left, right)]
            randomizer.shuffle(rest)
            result = []
            for pos in range(n):
                if pos == positions[0]:
                    result.append(left)
                elif pos == positions[1]:
                    result.append(right)
                else:
                    result.append(rest.pop())
            if tuple(result) not in seen:
                seen.add(tuple(result))
                break
            attempt += 1
    if sign == "minus":
        a, b = result.index(left), result.index(right)
        result[a], result[b] = result[b], result[a]
    return result


def payload(task, arm, config):
    result = one_shot_payload(task, "B", config)
    result["options"] = [visible(task["candidate_options"][i], config["timezone"])
                         for i in indices(task, arm, config)]
    return result


def paired_group(task):
    n = len(minima(task))
    return "unique_control" if n == 1 else "two_optima" if n == 2 else "multiple_optima"

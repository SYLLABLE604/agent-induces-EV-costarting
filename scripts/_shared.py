"""Shared imports for standalone analysis commands; standard library only."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/two_stage"))
from common import load_tasks, load_pairs, micro, slot  # noqa: E402


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def eligible_pairs(tasks):
    pairs = []
    for n, left in enumerate(tasks):
        for right in tasks[n + 1:]:
            if left.household != right.household and {c.bin for c in left.candidates} & {c.bin for c in right.candidates}:
                pairs.append(tuple(sorted((left.id, right.id))))
    return sorted(pairs)


def validate_pairs(tasks, pairs):
    normalized = [tuple(sorted(pair)) for pair in pairs]
    if len(normalized) != len(set(normalized)) or set(normalized) != set(eligible_pairs(tasks)):
        raise ValueError("Pairs must exactly match unique cross-household feasible-bin overlaps")


def complete_starts(tasks, starts):
    if set(starts) != {task.id for task in tasks}:
        raise ValueError("A proposal map must include every task exactly once")
    if any(starts[task.id] not in task.by_start for task in tasks):
        raise ValueError("Missing or infeasible starts cannot be omitted from the denominator")


def co_start(pairs, starts):
    if not pairs:
        return None
    return sum(slot(starts[left]) == slot(starts[right]) for left, right in pairs) / len(pairs)

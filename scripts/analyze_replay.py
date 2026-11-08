#!/usr/bin/env python3
"""Calculate analytic baselines, tie strata, and optional local model replay rates."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

from _shared import co_start, complete_starts, load_pairs, load_tasks, slot, validate_pairs, write_json


def expected_rate(tasks, pairs, optimal):
    by_id = {task.id: task for task in tasks}
    total = Fraction()
    for left_id, right_id in pairs:
        left, right = by_id[left_id], by_id[right_id]
        a = [c for c in left.candidates if not optimal or c.regret == 0]
        b = [c for c in right.candidates if not optimal or c.regret == 0]
        x, y = Counter(c.bin for c in a), Counter(c.bin for c in b)
        total += Fraction(sum(v * y[k] for k, v in x.items()), len(a) * len(b))
    return float(total / len(pairs)) if pairs else None


def summarize(tasks, pairs, proposals=None):
    if not pairs:
        raise ValueError("At least one eligible pair is needed for a co-start rate")
    validate_pairs(tasks, pairs)
    earliest = {task.id: min(task.optima) for task in tasks}
    by_id = {task.id: task for task in tasks}
    proposals = proposals or {}
    for starts in proposals.values():
        complete_starts(tasks, starts)
    def rates(subset):
        return {"pairs": len(subset), "independent_feasible": expected_rate(tasks, subset, False),
                "independent_minimum_cost": expected_rate(tasks, subset, True),
                "common_earliest_minimum": co_start(subset, earliest),
                "models": {name: co_start(subset, starts) for name, starts in proposals.items()}}
    strata = {}
    for count in range(3):
        subset = [(a, b) for a, b in pairs if int(len(by_id[a].optima) > 1) + int(len(by_id[b].optima) > 1) == count]
        strata[str(count)] = rates(subset)
    return {"tasks": len(tasks), "tied_tasks": sum(len(task.optima) > 1 for task in tasks),
            "overall": rates(pairs), "tie_strata": strata,
            "earliest_matches": {name: sum(starts[task.id] == earliest[task.id] for task in tasks) for name, starts in proposals.items()},
            "units": "Rates are proportions, not percentages. Empty strata have null rates."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, help='JSON mapping model name to a complete {task_id: UTC start} map')
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    proposals = json.loads(args.proposals.read_text()) if args.proposals else None
    write_json(args.out, summarize(load_tasks(args.tasks), load_pairs(args.pairs), proposals))
    print(args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract direct paired-swap statistics without assembling offline fleet plans."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/paired_swap"))
from protocol import minima
from transport import load_outcomes, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="One run.py output directory, not an unmerged cluster root")
    parser.add_argument("--choice", choices=["first", "final"], default="first")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    protocol = read_json(run / "protocol.json")
    config = protocol["identity"]["config"]
    records = load_outcomes(run, protocol["fingerprint"])
    tasks = [json.loads(line) for line in (ROOT / "experiments/paired_swap/data/tasks.jsonl").read_text().splitlines() if line.strip()]
    tasks = [task for task in tasks if len(minima(task)) == 2 and task["task_id"] in protocol["identity"]["task_ids"]]
    if not tasks:
        raise ValueError("No two-optimum tasks in this run")
    key = "first_start_utc" if args.choice == "first" else "system_start_utc"
    result = {"choice": args.choice, "note": "Only test paired swaps; no offline fleet assemblies.", "models": {}}
    for model in config["models"]:
        table = {r["task_id"]: {} for r in records.values() if r["model"] == model}
        for record in records.values():
            if record["model"] == model:
                table[record["task_id"]][record["arm"]] = record
        if not table:
            continue
        changed = both_optimal_changes = invalid = count = 0
        for task in tasks:
            optima = {c["start_time_utc"] for c in minima(task)}
            for repeat in range(config["test_repeats"]):
                arms = [f"test_{repeat:02d}_{sign}" for sign in ["plus", "minus"]]
                if any(arm not in table.get(task["task_id"], {}) for arm in arms):
                    raise ValueError("Incomplete paired test; refusing a partial denominator")
                a, b = [table[task["task_id"]][arm][key] for arm in arms]
                count += 1
                invalid += a is None or b is None
                changed += a is not None and b is not None and a != b
                both_optimal_changes += a in optima and b in optima and a != b
        result["models"][model] = {"tasks": len(tasks), "paired_comparisons": count, "changed_known": changed,
                                   "both_optimal_changes": both_optimal_changes, "unresolved_pairs": invalid,
                                   "switch_rate_over_all_pairs": changed / count}
    if not result["models"]:
        raise ValueError("No model records found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()

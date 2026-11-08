#!/usr/bin/env python3
"""Install the user's local task files into an empty experiment directory."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from _shared import ROOT, eligible_pairs, load_pairs, load_tasks, read_rows, validate_pairs, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--pairs", type=Path, help="Optional fixed pair CSV; otherwise construct all eligible pairs")
    parser.add_argument("--experiment", choices=["two_stage", "paired_swap", "both"], required=True)
    parser.add_argument("--two-optima-only", action="store_true", help="For the paired-swap experiment only")
    args = parser.parse_args()
    if args.two_optima_only and args.experiment != "paired_swap":
        parser.error("--two-optima-only applies only to --experiment paired_swap")
    rows = read_rows(args.tasks)
    typed = load_tasks(args.tasks)
    if not typed:
        raise ValueError("At least one task is required")
    pairs = load_pairs(args.pairs) if args.pairs else eligible_pairs(typed)
    validate_pairs(typed, pairs)
    if args.two_optima_only:
        keep = {task.id for task in typed if len(task.optima) == 2}
        rows = [row for row in rows if row["task_id"] in keep]
        typed = [task for task in typed if task.id in keep]
        pairs = [pair for pair in pairs if set(pair) <= keep]
        if not typed:
            raise ValueError("No tasks have exactly two minimum-cost starts")
    targets = ["two_stage", "paired_swap"] if args.experiment == "both" else [args.experiment]
    for name in targets:
        folder = ROOT / "experiments" / name
        if any((folder / item).exists() for item in ["data/tasks.jsonl", "data/pairs.csv", "prepared/manifest.json", "prepared/models.lock.json", "results"]):
            raise FileExistsError(f"Use a clean experiment copy; refusing to overwrite {name}")
    for name in targets:
        folder = ROOT / "experiments" / name
        data = folder / "data"
        data.mkdir(exist_ok=True)
        (data / "tasks.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        with (data / "pairs.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["pair_id", "task_id_a", "task_id_b"])
            writer.writeheader()
            writer.writerows({"pair_id": f"pair-{n:06d}", "task_id_a": left, "task_id_b": right} for n, (left, right) in enumerate(pairs))
        config = json.loads((folder / "config.json").read_text())
        config.update(expected_tasks=len(typed), expected_pairs=len(pairs), expected_tied_tasks=sum(len(task.optima) > 1 for task in typed))
        write_json(folder / "config.json", config)
        hashes = {file: hashlib.sha256((data / file).read_bytes()).hexdigest() for file in ["tasks.jsonl", "pairs.csv"]}
        write_json(data / "manifest.json", {"origin": "user-supplied local inputs; not paper data", "sha256": hashes})
        print(json.dumps({"experiment": name, "tasks": len(typed), "pairs": len(pairs), "tied_tasks": config["expected_tied_tasks"]}))


if __name__ == "__main__":
    main()

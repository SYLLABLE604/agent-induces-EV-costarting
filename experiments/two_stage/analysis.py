"""Round-level metrics on complete executable schedules; raw missingness separately retained."""
from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from assignment import PlanSolver
from common import load_pairs, load_tasks, slot
from protocol import ROOT, canonical, digest, job_id, raw_tasks, read_json, round_jobs, write_json


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metrics(starts, tasks, pairs, capacity):
    valid = {t.id: starts.get(t.id) for t in tasks if starts.get(t.id) in t.by_start}
    hits = affected = 0
    valid_pairs = 0
    for i, j in pairs:
        if i not in valid or j not in valid:
            affected += 1
        else:
            valid_pairs += 1
            hits += slot(valid[i]) == slot(valid[j])
    loads = Counter(slot(s) for s in valid.values())
    result = {
        "valid_answers": len(valid), "missing_answers": len(tasks) - len(valid),
        "nonoptimal_answers": sum(valid[t.id] not in t.optima for t in tasks if t.id in valid),
        "optimal_answers": sum(valid[t.id] in t.optima for t in tasks if t.id in valid),
        "affected_pairs": affected, "valid_pairs": valid_pairs,
        "co_start_lower_pct": 100 * hits / len(pairs),
        "co_start_upper_pct": 100 * (hits + affected) / len(pairs),
        "valid_pair_co_start_pct": None if not valid_pairs else 100 * hits / valid_pairs,
        "intervals_above_capacity": sum(n > capacity for n in loads.values()),
        "intervals_at_capacity": sum(n == capacity for n in loads.values()),
        "max_starts": max(loads.values(), default=0),
    }
    if not affected:
        result["co_start_pct"] = result["co_start_lower_pct"]
    return result


def complete_proposals(starts, tasks):
    return {t.id: starts[t.id] if starts.get(t.id) in t.by_start else min(t.optima) for t in tasks}


def plan_results(records, tasks, pairs, config, solver):
    starts = {r["task_id"]: r["parsed"]["start"] for r in records}
    completed = complete_proposals(starts, tasks)
    raw = metrics(starts, tasks, pairs, config["start_capacity"])
    first = metrics(completed, tasks, pairs, config["start_capacity"])
    solved = solver.solve(starts)
    if not solved["feasible"]:
        raise ValueError(f"No authorized execution: {solved}")
    second = metrics(solved["starts"], tasks, pairs, config["start_capacity"])
    assert second["valid_answers"] == len(tasks) and second["nonoptimal_answers"] == 0
    assert second["intervals_above_capacity"] == 0
    # Causes partition original proposal changes, not a sequence of greedy repairs.
    completions = sum(starts.get(t.id) not in t.by_start for t in tasks)
    nonoptimal_revisions = sum(starts.get(t.id) in t.by_start and starts[t.id] not in t.optima for t in tasks)
    capacity_revisions = sum(starts.get(t.id) in t.optima and solved["starts"][t.id] != starts[t.id] for t in tasks)
    assert solved["modified"] == completions + nonoptimal_revisions + capacity_revisions
    row = {"model": records[0]["model"], "round": records[0]["round"], "condition": records[0]["condition"],
           "tasks": len(tasks), "eligible_pairs": len(pairs)}
    row.update({"raw_" + k: v for k, v in raw.items()})
    row.update({"stage1_" + k: v for k, v in first.items()})
    row.update({"stage2_" + k: v for k, v in second.items()})
    row.update({"required_completions": completions, "nonoptimal_revisions": nonoptimal_revisions,
                "capacity_revisions": capacity_revisions, "minimum_total_changes_from_raw": solved["modified"],
                "changes_from_completed_stage1": sum(completed[i] != s for i, s in solved["starts"].items()),
                "service_retry_attempts": sum(max(0, r["attempts"] - 1) for r in records)})
    artifact = {"model": row["model"], "round": row["round"], "condition": row["condition"],
                "response_ids": {r["task_id"]: r["job_id"] for r in records},
                "raw_proposals": starts, "stage1_executable_after_failure_rule": completed,
                "stage2_final": solved["starts"], "metrics": row}
    return row, artifact


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"], row["condition"]].append(row)
    report = []
    for (model, condition), values in sorted(groups.items()):
        for key in values[0]:
            if key in {"model", "condition", "round"}:
                continue
            numbers = [r[key] for r in values if isinstance(r.get(key), (int, float))]
            if not numbers:
                continue
            report.append({"model": model, "condition": condition, "metric": key,
                           "rounds": len(values), "available_rounds": len(numbers), "mean": statistics.mean(numbers),
                           "std": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
                           "min": min(numbers), "max": max(numbers)})
    return report


def analyze(out: Path, config: dict, allow_partial=False):
    from run import collect
    from protocol import protocol_manifest
    manifest = read_json(out / "experiment.json")
    if manifest["protocol_sha256"] != protocol_manifest(config)["sha256"]:
        raise ValueError("Analysis protocol differs from the recorded run")
    tasks = load_tasks(ROOT / "data/tasks.jsonl")
    pairs = load_pairs(ROOT / "data/pairs.csv")
    records = collect(out)
    lookup = {t["task_id"]: t for t in raw_tasks()}
    expected_ids = {job_id(m, r, c, t.id) for m in config["models"] for r in range(config["rounds"]) for c in config["conditions"] for t in tasks}
    actual = {r["job_id"] for r in records}
    if actual - expected_ids:
        raise ValueError("Unexpected job IDs in ledger")
    if actual != expected_ids and not allow_partial:
        raise ValueError(f"Incomplete experiment: {len(actual)}/{len(expected_ids)}. Resume workers; use --allow-partial for a labeled preview.")
    # Verify records against frozen treatment inputs; no substitution of old outputs.
    grouped = defaultdict(list)
    expected_jobs = {}
    for record in records:
        key = (record["model"], record["round"])
        if key not in expected_jobs:
            expected_jobs[key] = {j["job_id"]: j for j in round_jobs(*key, config, list(lookup.values()))}
        target = expected_jobs[key][record["job_id"]]
        if record["messages"] != target["messages"] or record["candidate_ids"] != target["candidate_ids"]:
            raise ValueError(f"Treatment inputs differ: {record['job_id']}")
        if record["protocol_sha256"] != manifest["protocol_sha256"] or record["model_sha256"] != manifest["models"][record["model"]]:
            raise ValueError("Response provenance differs from manifest")
        from protocol import parse_choice
        recomputed = parse_choice(record["inference"]["raw_output"], lookup[record["task_id"]]) if record["inference"].get("raw_output") is not None else {"category": "service_failure", "start": None, "candidate_id": None}
        if record["parsed"] != recomputed:
            raise ValueError("Parsed result changed after inference")
        grouped[record["model"], record["round"], record["condition"]].append(record)
    destination = out / "analysis"
    destination.mkdir(exist_ok=True)
    rows, artifacts = [], []
    solver = PlanSolver(tasks, config["start_capacity"], config["cost_budget_micro"])
    for _, group in sorted(grouped.items()):
        if len(group) == len(tasks):
            row, artifact = plan_results(group, tasks, pairs, config, solver)
            rows.append(row)
            artifacts.append(artifact)
    with (destination / "plans.jsonl").open("w", encoding="utf-8") as f:
        for record in artifacts:
            f.write(canonical(record) + "\n")
    csv_write(destination / "round_metrics.csv", rows)
    summary = summarize(rows)
    csv_write(destination / "summary.csv", summary)
    by_round = {(r["model"], r["round"], r["condition"]): r for r in rows}
    deltas = []
    main_metrics = ["stage1_co_start_pct", "stage2_co_start_pct", "stage1_intervals_above_capacity", "minimum_total_changes_from_raw"]
    for model in config["models"]:
        for r in range(config["rounds"]):
            ordered = by_round.get((model, r, "ordered"))
            shuffled = by_round.get((model, r, "shuffled"))
            if ordered and shuffled:
                deltas.append({"model": model, "round": r,
                               **{key + "_shuffled_minus_ordered": shuffled[key] - ordered[key] for key in main_metrics}})
    csv_write(destination / "paired_round_differences.csv", deltas)
    consistency = []
    for model in config["models"]:
        for t in tasks:
            choices = [r["parsed"]["start"] for r in records if r["model"] == model and r["condition"] == "ordered" and r["task_id"] == t.id]
            consistency.append({"model": model, "task_id": t.id, "rounds": len(choices), "distinct_choices_including_missing": len(set(choices))})
    csv_write(destination / "ordered_consistency.csv", consistency)
    final = {"mode": manifest["mode"], "complete": actual == expected_ids,
             "calls": len(records), "expected_calls": len(expected_ids), "complete_plans": len(rows),
             "categories": dict(Counter(r["parsed"]["category"] for r in records)),
             "output_formats": dict(Counter(r["parsed"].get("output_format", "unresolved") for r in records)), "summary": summary,
             "interpretation": "Real calls per round. Spread is conditional on this frozen task set, checkpoints, prompts, and generated permutations. No response-pool recombination or complementary plans."}
    write_json(destination / "summary.json", final)
    title = "MOCK / NOT RESEARCH RESULTS" if manifest["mode"] == "mock" else "Transformers real-call evaluation"
    lines = [f"# {title}", "", f"Complete: {final['complete']}; first-call jobs: {len(records)}/{len(expected_ids)}; complete plans: {len(rows)}.", "",
             "Each round uses fresh model calls for both treatments. Stage 1 main metrics include the prespecified earliest-minimum completion for unresolved responses; raw proposal bounds and valid-pair denominators are recorded separately. Stage 2 exactly minimizes changes from raw proposals under zero extra cost and the same interval cap.", "",
             "| Model | Treatment | Complete rounds | Stage 1 co-start (%) | Stage 2 co-start (%) | Minimum changes | Overloaded intervals, stage 1 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    index = {(v["model"], v["condition"], v["metric"]): v for v in summary}
    for model in config["models"]:
        for arm in config["conditions"]:
            cells = [index.get((model, arm, key)) for key in main_metrics]
            if not all(cells):
                continue
            first, second, overloaded, modified = cells
            fmt = lambda x: f"{x['mean']:.3f} ± {x['std']:.3f}"
            lines.append(f"| {model} | {arm} | {first['rounds']} | {fmt(first)} | {fmt(second)} | {fmt(modified)} | {fmt(overloaded)} |")
    lines += ["", "Full round metrics, ranges, failure categories, all response links and both executable stages are retained. No result-based exclusions. The 103 pairs are not treated as independent replicates. Complementary/antithetic sampling is not used.", ""]
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(canonical({k: v for k, v in final.items() if k != "summary"}))

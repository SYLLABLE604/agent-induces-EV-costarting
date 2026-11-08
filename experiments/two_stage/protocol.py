"""Frozen randomized full-list protocol. No inference and no output-based selection."""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from common import load_pairs, load_tasks, micro, slot

ROOT = Path(__file__).resolve().parent


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def raw_tasks() -> list[dict]:
    return sorted((json.loads(line) for line in (ROOT / "data/tasks.jsonl").read_text().splitlines() if line.strip()), key=lambda t: t["task_id"])


def options(task: dict) -> list[dict]:
    return sorted(task["candidate_options"], key=lambda c: c["start_time_utc"])


def candidate_map(task: dict) -> dict[str, dict]:
    return {f"c{n:04d}": opt for n, opt in enumerate(options(task))}


def permutation(task: dict, condition: str, round_id: int, config: dict) -> tuple[list[str], int]:
    if condition not in config["conditions"]:
        raise ValueError(condition)
    ids = list(candidate_map(task))
    seed = int(digest([config["shuffle_seed"], task["task_id"], round_id]), 16) % (2**63)
    if condition == "shuffled":
        random.Random(seed).shuffle(ids)
    return ids, seed


def payload(task: dict, ids: list[str]) -> dict:
    mapping = candidate_map(task)
    return {
        "planning_time_utc": task["release_time_utc"],
        "task": {
            "energy_required_kwh": task["energy_required_kwh"],
            "charger_power_kw": task["charger_power_kw"],
            "continuous_duration_hours": task["continuous_duration_hours"],
            "availability_end_utc": task["availability_end_utc"],
            "latest_start_utc": task["latest_start_utc"],
        },
        "release_snapshot": task.get("release_context", {}),
        "options": [{"candidate_id": key, **mapping[key]} for key in ids],
    }


def job_id(model: str, round_id: int, condition: str, task_id: str) -> str:
    return f"{model}/round_{round_id:02d}/{condition}/{task_id}"


def round_jobs(model: str, round_id: int, config: dict, tasks: list[dict]) -> list[dict]:
    # Both treatments run on the same worker/GPU in each round, interleaved by task.
    shuffled_tasks = sorted(tasks, key=lambda t: t["task_id"])
    random.Random(int(digest([config["schedule_seed"], round_id]), 16)).shuffle(shuffled_tasks)
    prompt = (ROOT / "prompts/one_shot.txt").read_text()
    jobs = []
    for task in shuffled_tasks:
        arms = list(config["conditions"])
        random.Random(int(digest([config["schedule_seed"], round_id, task["task_id"]]), 16)).shuffle(arms)
        for arm in arms:
            ids, seed = permutation(task, arm, round_id, config)
            value = payload(task, ids)
            jobs.append({
                "job_id": job_id(model, round_id, arm, task["task_id"]),
                "model": model, "round": round_id, "condition": arm,
                "task_id": task["task_id"], "household_id": task["household_id"],
                "candidate_ids": ids, "permutation_seed": seed if arm == "shuffled" else None,
                "prompt_id": file_hash(ROOT / "prompts/one_shot.txt"),
                "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": canonical(value)}],
            })
    return jobs


def parse_choice(raw: str, task: dict) -> dict:
    def strict_object(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate_json_key")
            out[key] = value
        return out
    output_format = "plain_json"
    text = raw.strip() if isinstance(raw, str) else raw
    if isinstance(text, str) and text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```[ \t]*", text)
        if not match:
            return {"category": "invalid_json", "start": None, "candidate_id": None}
        text = match.group(1)
        output_format = "single_json_fence"
    try:
        obj = json.loads(text, object_pairs_hook=strict_object)
    except (ValueError, TypeError):
        return {"category": "invalid_json", "start": None, "candidate_id": None}
    if not isinstance(obj, dict) or set(obj) != {"candidate_id"}:
        return {"category": "invalid_schema", "start": None, "candidate_id": None}
    key = obj["candidate_id"]
    mapping = candidate_map(task)
    if not isinstance(key, str) or key not in mapping:
        return {"category": "unknown_candidate", "start": None, "candidate_id": None}
    selected = mapping[key]
    minimum = min(micro(c["projected_full_session_tariff_cost"]) for c in mapping.values())
    is_optimum = micro(selected["projected_full_session_tariff_cost"]) == minimum
    return {"category": "valid_optimal" if is_optimum else "valid_nonoptimal", "start": selected["start_time_utc"], "candidate_id": key, "output_format": output_format}


def validate_inputs(config: dict) -> dict:
    tasks = raw_tasks()
    typed = load_tasks(ROOT / "data/tasks.jsonl")
    pairs = load_pairs(ROOT / "data/pairs.csv")
    if config["rounds"] < 20:
        raise ValueError("The formal protocol requires at least 20 rounds per treatment")
    if config["conditions"] != ["ordered", "shuffled"] or config["generation"]["do_sample"]:
        raise ValueError("This protocol fixes ordered/shuffled greedy generation")
    if config["cost_budget_micro"] != 0 or config["failure_rule"] != "earliest_exact_minimum":
        raise ValueError("Formal protocol uses zero budget and earliest-exact-minimum completion")
    actual_pairs = set()
    bins = {t.id: {c.bin for c in t.candidates} for t in typed}
    for n, left in enumerate(typed):
        for right in typed[n + 1:]:
            if left.household != right.household and bins[left.id] & bins[right.id]:
                actual_pairs.add(tuple(sorted((left.id, right.id))))
    if len(pairs) != len(set(tuple(sorted(p)) for p in pairs)) or actual_pairs != {tuple(sorted(p)) for p in pairs}:
        raise ValueError("Frozen eligible pairs do not match cross-household feasible-bin overlap")
    tied = sum(len(t.optima) > 1 for t in typed)
    expected = (config["expected_tasks"], config["expected_pairs"], config["expected_tied_tasks"])
    if (len(tasks), len(pairs), tied) != expected:
        raise ValueError(f"Frozen counts differ: {(len(tasks), len(pairs), tied)} != {expected}")
    return {"tasks": len(tasks), "pairs": len(pairs), "tied_tasks": tied,
            "expected_first_calls": len(tasks) * 2 * config["rounds"] * len(config["models"])}


def protocol_manifest(config: dict) -> dict:
    paths = sorted(list(ROOT.glob("*.py")) + [ROOT / "config.json", ROOT / "prompts/one_shot.txt", ROOT / "data/tasks.jsonl", ROOT / "data/pairs.csv"])
    files = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}
    return {"protocol": config, "files": files, "sha256": digest(files)}

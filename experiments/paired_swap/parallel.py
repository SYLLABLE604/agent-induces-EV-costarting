"""Deterministic four-worker partitioning and manifest validation."""
from __future__ import annotations

import json
from pathlib import Path

from design import conditions, payload, select_tasks
from protocol import canonical, digest
from transport import atomic, read_json


def task_weight(task: dict, config: dict) -> int:
    """Pre-output prompt-byte proxy across all experimental conditions."""
    return sum(len(canonical(payload(task, arm, config))) for arm in conditions(config))


def balanced_halves(tasks: list[dict], config: dict) -> tuple[list[str], list[str]]:
    bins: list[list[str]] = [[], []]
    totals = [0, 0]
    weighted = sorted(((task_weight(task, config), task["task_id"]) for task in tasks),
                      key=lambda row: (-row[0], row[1]))
    for weight, task_id in weighted:
        target = min(range(2), key=lambda index: (totals[index], index))
        bins[target].append(task_id)
        totals[target] += weight
    return tuple(sorted(group) for group in bins)


def build_worker_manifests(config: dict, all_tasks: list[dict], scope: str,
                           gpu_ids: list[int], model_paths: dict[str, Path],
                           model_locks: dict[str, dict]) -> list[dict]:
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4 or any(gpu < 0 for gpu in gpu_ids):
        raise ValueError("Exactly four distinct nonnegative GPU indices are required")
    if set(model_paths) != set(config["models"]) or set(model_locks) != set(config["models"]):
        raise ValueError("Model paths and locks must cover every configured model")
    tasks = select_tasks(all_tasks, scope)
    halves = balanced_halves(tasks, config)
    models = list(config["models"])
    if len(models) != 2:
        raise ValueError("The four-GPU layout requires exactly the two frozen models")
    manifests = []
    for worker in range(4):
        model_index, half = divmod(worker, 2)
        task_ids = halves[half]
        manifests.append({
            "schema_version": "presentation_control_worker_v2_transformers",
            "worker_index": worker,
            "scope": scope,
            "gpu_index": gpu_ids[worker],
            "model": models[model_index],
            "model_path": str(model_paths[models[model_index]].resolve()),
            "model_content_sha256": model_locks[models[model_index]]["content_sha256"],
            "task_ids": task_ids,
            "task_weight": sum(task_weight(task, config) for task in tasks
                               if task["task_id"] in set(task_ids)),
            "expected_outcomes": len(task_ids) * len(conditions(config)),
        })
    validate_partition(manifests, tasks, config)
    return manifests


def validate_partition(manifests: list[dict], tasks: list[dict], config: dict) -> None:
    if len(manifests) != 4 or {m["worker_index"] for m in manifests} != set(range(4)):
        raise ValueError("Worker indices must be exactly 0,1,2,3")
    expected_ids = {task["task_id"] for task in tasks}
    for model in config["models"]:
        groups = [set(m["task_ids"]) for m in manifests if m["model"] == model]
        if len(groups) != 2 or groups[0] & groups[1] or groups[0] | groups[1] != expected_ids:
            raise ValueError(f"Incomplete or overlapping task partition for {model}")
    if len({m["gpu_index"] for m in manifests}) != 4:
        raise ValueError("GPU assignments overlap")
    for manifest in manifests:
        if manifest["model"] not in config["models"]:
            raise ValueError("Unknown model in worker manifest")
        if manifest["expected_outcomes"] != len(manifest["task_ids"]) * len(conditions(config)):
            raise ValueError("Worker outcome count differs")


def attach_worker(output: Path, manifest: dict, study_fingerprint: str) -> dict:
    frozen = {**manifest, "study_fingerprint": study_fingerprint}
    path = output / "worker.json"
    if path.exists() and read_json(path) != frozen:
        raise ValueError("Worker manifest changed; use a new output root")
    if not path.exists():
        atomic(path, frozen)
    return frozen


def load_worker(path: Path, scope: str, tasks: list[dict], config: dict,
                study_fingerprint: str) -> tuple[dict, list[dict]]:
    manifest = read_json(path)
    required = {"schema_version", "worker_index", "scope", "gpu_index", "model",
                "model_path", "model_content_sha256", "task_ids", "task_weight",
                "expected_outcomes", "study_fingerprint"}
    if (set(manifest) != required
            or manifest["schema_version"] != "presentation_control_worker_v2_transformers"):
        raise ValueError("Invalid worker manifest schema")
    if manifest["scope"] != scope or manifest["study_fingerprint"] != study_fingerprint:
        raise ValueError("Worker manifest belongs to another study")
    task_map = {task["task_id"]: task for task in tasks}
    if len(manifest["task_ids"]) != len(set(manifest["task_ids"])):
        raise ValueError("Duplicate task in worker manifest")
    if any(task_id not in task_map for task_id in manifest["task_ids"]):
        raise ValueError("Worker task falls outside selected scope")
    selected = [task_map[task_id] for task_id in manifest["task_ids"]]
    if manifest["model"] not in config["models"]:
        raise ValueError("Worker model falls outside frozen configuration")
    if manifest["expected_outcomes"] != len(selected) * len(conditions(config)):
        raise ValueError("Worker expected count differs")
    return manifest, selected


def write_parallel_manifest(root: Path, study_fingerprint: str, scope: str,
                            manifests: list[dict]) -> dict:
    value = {"schema_version": "presentation_control_parallel_v2_transformers",
             "study_fingerprint": study_fingerprint, "scope": scope,
             "workers": [{"worker_index": m["worker_index"],
                          "sha256": digest(m)} for m in manifests],
             "expected_outcomes": sum(m["expected_outcomes"] for m in manifests)}
    path = root / "parallel_manifest.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("Parallel manifest changed; use a new output root")
    if not path.exists():
        atomic(path, value)
    return value

#!/usr/bin/env python3
"""Presentation-response experiment using local PyTorch/Transformers models."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path

from design import conditions, indices, paired_group, payload, rng, select_tasks
from hf_backend import (TransformersBackend, build_model_locks, inspect_runtime,
                        load_model_paths, verify_model_lock)
from parallel import attach_worker, load_worker
from protocol import digest, measure, minima, parse, request_body
from transport import Client, atomic, file_hash, load_inputs, load_outcomes, read_json

ROOT = Path(__file__).resolve().parent


def prepare(config_path: Path, output: Path, scope: str,
            model_locks: dict[str, dict] | None = None):
    config, all_tasks, all_pairs, prompts = load_inputs(config_path)
    for key in ("calibration_repeats", "test_repeats", "independent_plan_repeats"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    if config["authorization"] != "all_frozen_exact_minima":
        raise ValueError("This package fixes authorization to all exact minima")
    if (not config["capacity_levels"]
            or any(type(value) is not int or value < 1 for value in config["capacity_levels"])):
        raise ValueError("Capacities must be positive integers")

    output = output.resolve()
    protocol_path = output / "protocol.json"
    if model_locks is None and protocol_path.exists():
        model_locks = read_json(protocol_path)["identity"]["model_locks"]
    if model_locks is None or set(model_locks) != set(config["models"]):
        raise ValueError("Frozen local model identities are required for a new protocol")
    for name, lock in model_locks.items():
        if lock.get("logical_name") != name or not lock.get("content_sha256"):
            raise ValueError(f"Invalid frozen model identity: {name}")
        base_identity = {
            key: lock[key] for key in ("logical_name", "source", "model_type", "files")
        }
        if digest(base_identity) != lock["content_sha256"]:
            raise ValueError(f"Inconsistent frozen model identity: {name}")

    tasks = select_tasks(all_tasks, scope)
    task_ids = {task["task_id"] for task in tasks}
    pairs = [pair for pair in all_pairs
             if pair["task_id_a"] in task_ids and pair["task_id_b"] in task_ids]
    orders = {
        task["task_id"]: {arm: indices(task, arm, config) for arm in conditions(config)}
        for task in tasks
    }
    identity = {
        "config": config,
        "scope": scope,
        "prompts": prompts,
        "data": read_json(ROOT / "data/manifest.json"),
        "task_ids": sorted(task_ids),
        "pairs": pairs,
        "orders_sha256": digest(orders),
        "model_locks": model_locks,
        "code_sha256": {
            name: file_hash(ROOT / name)
            for name in (
                "run.py", "design.py", "protocol.py", "transport.py", "hf_backend.py",
                "parallel.py", "analyze.py", "optimization.py",
            )
        },
    }
    fingerprint = digest(identity)
    protocol = {
        "fingerprint": fingerprint,
        "identity": identity,
        "expected_outcomes": len(tasks) * len(config["models"]) * len(conditions(config)),
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if protocol_path.exists():
            if read_json(protocol_path) != protocol or read_json(output / "orders.json") != orders:
                raise ValueError("Resume rejected: inputs, code, model, or scope changed")
        else:
            if any(path.name != "prepare.lock" for path in output.iterdir()):
                raise ValueError("New experiment needs an empty directory")
            atomic(output / "orders.json", orders)
            atomic(protocol_path, protocol)
    return config, tasks, pairs, prompts, fingerprint


def execute(task, model, arm, config, prompt, call):
    data = payload(task, arm, config)
    context = {"model": model, "arm": arm, "task_id": task["task_id"], "repair": False}
    traces = []
    response, key, hit = call(request_body(model, prompt, data, config, "B"), context)
    raw = response.get("message", {}).get("content", "")
    valid, selected, error = parse(raw, task, "B", config["timezone"])
    first = selected if valid else None
    traces.append({"request_hash": key, "cache_hit": hit, "repair": False, "error": error})
    if not valid:
        response, key, hit = call(
            request_body(model, prompt, data, config, "B", raw),
            {**context, "repair": True},
        )
        valid, selected, error = parse(
            response.get("message", {}).get("content", ""), task, "B", config["timezone"]
        )
        traces.append({"request_hash": key, "cache_hit": hit, "repair": True, "error": error})
    selected = selected if valid else None
    metrics = measure(task, selected, "B", config)
    presented = [task["candidate_options"][index]["start_time_utc"]
                 for index in indices(task, arm, config)]
    optimal = {option["start_time_utc"] for option in minima(task)}
    first_presented_minimum = next(start for start in presented if start in optimal)
    metrics["first_presented_minimum"] = selected == first_presented_minimum
    metrics["first_presented_option"] = selected == presented[0]
    metrics["discriminating_order"] = first_presented_minimum != minima(task)[0]["start_time_utc"]
    return {
        "model": model,
        "task_id": task["task_id"],
        "arm": arm,
        "group": paired_group(task),
        "system_start_utc": selected,
        "first_start_utc": first,
        "repair_calls": len(traces) - 1,
        "candidate_order": indices(task, arm, config),
        "metrics": metrics,
        "request_traces": traces,
    }


def _model_inputs(args, output: Path, config: dict) -> tuple[dict[str, Path], dict[str, dict]]:
    if args.worker_manifest:
        worker = read_json(args.worker_manifest.resolve())
        paths = {worker["model"]: Path(worker["model_path"]).resolve()}
    elif args.models:
        paths = load_model_paths(args.models, config)
    else:
        paths = {}

    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        locks = read_json(protocol_path)["identity"]["model_locks"]
    elif paths and set(paths) == set(config["models"]):
        locks = build_model_locks(paths, config)
    else:
        raise ValueError("A new run requires --models with both local model paths")
    return paths, locks


def main() -> None:
    base_config = read_json(ROOT / "config.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["prepare", "check", "preflight", "run", "status", "audit", "analyze"]
    )
    parser.add_argument("--scope", choices=["pilot", "full"], default="full")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--models", type=Path, help="JSON mapping logical model names to local paths")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", choices=list(base_config["models"]))
    parser.add_argument("--worker-manifest", type=Path,
                        help="Frozen four-GPU worker manifest; restricts model and tasks")
    parser.add_argument("--phase", choices=["all", "calibration", "test"], default="all")
    parser.add_argument("--max-outcomes", type=int)
    args = parser.parse_args()
    output = (args.output or ROOT / "results" / args.scope).resolve()

    if args.mode == "status":
        protocol = read_json(output / "protocol.json")
        records = load_outcomes(output, protocol["fingerprint"])
        worker = read_json(output / "worker.json") if (output / "worker.json").exists() else None
        expected = worker["expected_outcomes"] if worker else protocol["expected_outcomes"]
        print(json.dumps({
            "completed": len(records),
            "expected": expected,
            "worker": None if worker is None else worker["worker_index"],
            "last_status": (read_json(output / "status.json")
                            if (output / "status.json").exists() else "prepared"),
        }, ensure_ascii=False, indent=2))
        return

    config = load_inputs(args.config)[0]
    paths, model_locks = _model_inputs(args, output, config)
    config, tasks, pairs, prompts, fingerprint = prepare(
        args.config, output, args.scope, model_locks
    )
    worker = None
    if args.worker_manifest:
        if args.model or args.models:
            raise ValueError("A worker manifest cannot be combined with --model or --models")
        worker, tasks = load_worker(
            args.worker_manifest.resolve(), args.scope, tasks, config, fingerprint
        )
        worker = attach_worker(output, worker, fingerprint)
        if worker["model_content_sha256"] != model_locks[worker["model"]]["content_sha256"]:
            raise ValueError("Worker model identity differs from the scientific protocol")
        verify_model_lock(paths[worker["model"]], model_locks[worker["model"]], full_hash=False)
        worker_task_ids = {task["task_id"] for task in tasks}
        pairs = [pair for pair in pairs if pair["task_id_a"] in worker_task_ids
                 and pair["task_id_b"] in worker_task_ids]
        models = [worker["model"]]
        expected_outcomes = worker["expected_outcomes"]
    else:
        models = [args.model] if args.model else list(config["models"])
        expected_outcomes = read_json(output / "protocol.json")["expected_outcomes"]

    if args.mode == "prepare":
        print(json.dumps({
            "scope": args.scope,
            "tasks": len(tasks),
            "pairs": len(pairs),
            "expected": expected_outcomes,
            "worker": None if worker is None else worker["worker_index"],
            "fingerprint": fingerprint,
            "model_identities": {
                name: model_locks[name]["content_sha256"] for name in model_locks
            },
        }, indent=2))
        return

    if args.mode == "analyze":
        if worker is not None:
            raise ValueError("Analyze the merged study, not one worker shard")
        from analyze import analyze
        analyze(output)
        return

    if args.mode == "audit":
        records = load_outcomes(output, fingerprint)
        task_map = {task["task_id"]: task for task in tasks}
        for (model, task_id, arm), stored in records.items():
            if model not in models or task_id not in task_map or arm not in conditions(config):
                raise ValueError("Unexpected outcome identity")
            traces = iter(stored["request_traces"])

            def cached(body, context):
                trace = next(traces)
                key = trace["request_hash"]
                record = read_json(output / "cache" / f"{key}.json")
                if record["request"] != body or record["identity"]["context"] != context:
                    raise ValueError("Cached request mismatch")
                if (record["identity"]["protocol"] != fingerprint
                        or digest(record["response"]) != record["response_sha256"]):
                    raise ValueError("Cached identity or response mismatch")
                if digest({"identity": record["identity"], "body": body}) != key:
                    raise ValueError("Cache key mismatch")
                return record["response"], key, trace["cache_hit"]

            replayed = execute(task_map[task_id], model, arm, config, prompts["one_shot"], cached)
            if replayed != stored or next(traces, None) is not None:
                raise ValueError("Offline response reconstruction differs")
        print(json.dumps({
            "audited": len(records),
            "expected": expected_outcomes,
            "worker": None if worker is None else worker["worker_index"],
            "real_model_calls": 0,
        }))
        return

    missing_paths = [name for name in models if name not in paths]
    if missing_paths:
        raise ValueError(f"Local paths are missing for: {', '.join(missing_paths)}")
    selected_paths = {name: paths[name] for name in models}
    runtime = inspect_runtime(config, selected_paths, model_locks, models)
    runtime["execution"] = {
        "worker_index": None if worker is None else worker["worker_index"],
        "declared_gpu_index": None if worker is None else worker["gpu_index"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if args.mode == "check":
        print(json.dumps(runtime, indent=2))
        return

    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime_path = output / "runtime.json"
        if runtime_path.exists() and read_json(runtime_path) != runtime:
            raise ValueError("Runtime changed; resume refused")
        atomic(runtime_path, runtime)
        records = load_outcomes(output, fingerprint)
        client = Client(output, config, fingerprint, runtime, None, expected_outcomes)
        client.completed = len(records)
        chosen = {
            min(tasks, key=lambda task: len(task["candidate_options"]))["task_id"],
            max(tasks, key=lambda task: len(task["candidate_options"]))["task_id"],
        }
        newly_completed = 0
        try:
            for model in models:
                backend = TransformersBackend(
                    model, paths[model], config, model_locks[model]
                )
                client.generate = backend.generate
                try:
                    for task in tasks:
                        if args.mode == "preflight" and task["task_id"] not in chosen:
                            continue
                        arms = conditions(config)
                        if args.phase == "calibration":
                            arms = [arm for arm in arms if not arm.startswith("test")]
                        elif args.phase == "test":
                            arms = [arm for arm in arms if arm.startswith("test")]
                        if args.mode == "preflight":
                            arms = [arm for arm in arms
                                    if arm in ("baseline", "cal_00_plus", "cal_00_minus")]
                        rng(config["shuffle_seed"], "execution", task["task_id"]).shuffle(arms)
                        for arm in arms:
                            key = (model, task["task_id"], arm)
                            if key in records:
                                continue
                            if (args.max_outcomes is not None
                                    and newly_completed >= args.max_outcomes):
                                client.status("paused", reason="max-outcomes")
                                return
                            row = execute(
                                task, model, arm, config, prompts["one_shot"], client.call
                            )
                            atomic(output / "outcomes" / f"{digest(key)}.json", {
                                "protocol": fingerprint,
                                "outcome": row,
                                "sha256": digest(row),
                            })
                            records[key] = row
                            newly_completed += 1
                            client.completed = len(records)
                            client.status("running", model=model,
                                          task_id=task["task_id"], arm=arm)
                finally:
                    client.generate = None
                    backend.close()
            final_runtime = inspect_runtime(config, selected_paths, model_locks, models)
            final_runtime["execution"] = runtime["execution"]
            if final_runtime != runtime:
                raise ValueError("Runtime identity changed during execution")
            state = "complete" if len(records) == expected_outcomes else "paused"
            reason = "preflight_finished" if args.mode == "preflight" else "selected_models_finished"
            client.status(state, reason=reason)
        except BaseException as exc:
            client.status("stopped", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()

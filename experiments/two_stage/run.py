#!/usr/bin/env python3
"""Prepare, inspect, and run actual ordered/shuffled calls. No response reuse across jobs."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import sys
import time
from collections import Counter
from pathlib import Path

from protocol import (ROOT, canonical, digest, file_hash, job_id, payload, permutation,
                      protocol_manifest, raw_tasks, read_json, round_jobs, validate_inputs, write_json)


def emit(value):
    print(canonical(value), flush=True)


def append(path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_records(path):
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    out = []
    for n, line in enumerate(lines):
        if not line.endswith(b"\n"):
            if n != len(lines) - 1:
                raise ValueError(f"Broken ledger inside {path}")
            break
        out.append(json.loads(line))
    return out


def repair_tail(path):
    if path.exists():
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            # Only a torn final write may be removed; complete records are immutable.
            end = data.rfind(b"\n") + 1
            with path.open("r+b") as f:
                f.truncate(end)


def prepared(config):
    current = protocol_manifest(config)
    path = ROOT / "prepared/manifest.json"
    if not path.exists() or read_json(path) != current:
        raise ValueError("Missing/stale protocol lock; run prepare before formal calls. Never edit a frozen experiment in place.")
    return current


def prepare(config):
    counts = validate_inputs(config)
    manifest = protocol_manifest(config)
    directory = ROOT / "prepared"
    path = directory / "manifest.json"
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError("Protocol changed after prepare; use a fresh package/output directory")
        emit({"status": "already_prepared", **counts, "protocol": manifest["sha256"]})
        return
    directory.mkdir(exist_ok=True)
    rows = []
    for r in range(config["rounds"]):
        for task in raw_tasks():
            for arm in config["conditions"]:
                ids, seed = permutation(task, arm, r, config)
                rows.append({"task_id": task["task_id"], "round": r, "condition": arm,
                             "seed": seed if arm == "shuffled" else None, "candidate_ids": ids})
    with (directory / "permutations.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(canonical(row) + "\n")
    # Full task ledger, including source IDs and exact optimum membership.
    from protocol import candidate_map
    from common import micro
    with (directory / "task_ledger.jsonl").open("w", encoding="utf-8") as f:
        for task in raw_tasks():
            mapping = candidate_map(task)
            minimum = min(micro(c["projected_full_session_tariff_cost"]) for c in mapping.values())
            row = {**task, "stable_candidates": [{"candidate_id": key, **c,
                   "is_exact_minimum": micro(c["projected_full_session_tariff_cost"]) == minimum} for key, c in mapping.items()]}
            f.write(canonical(row) + "\n")
    write_json(path, manifest)
    emit({"status": "prepared", **counts, "protocol": manifest["sha256"]})


def model_paths(args, config):
    data = read_json(args.models)
    if set(data) != set(config["models"]):
        raise ValueError("models.local.json must map qwen and gemma to different local checkpoint folders")
    paths = {name: Path(value).expanduser().resolve() for name, value in data.items()}
    if any(not p.is_dir() for p in paths.values()) or len(set(paths.values())) != len(paths):
        raise ValueError("Model folders are missing or identical")
    return paths


def frozen_models(args, config, paths):
    from backend import verify_identity
    lock_path = ROOT / "prepared/models.lock.json"
    if not lock_path.exists():
        raise ValueError("Run freeze-models first; this hashes the actual local weight files")
    locks = read_json(lock_path)
    for name in config["models"]:
        verify_identity(paths[name], locks[name])
    return locks


def bind_output(out, manifest, mode, locks):
    out.mkdir(parents=True, exist_ok=True)
    value = {"protocol_sha256": manifest["sha256"], "mode": mode,
             "models": {name: lock["sha256"] for name, lock in locks.items()}}
    lock = out / "experiment.json"
    # Atomic exclusive creation, safe when four workers start together.
    try:
        with lock.open("x", encoding="utf-8") as f:
            f.write(canonical(value) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except FileExistsError:
        for _ in range(10):
            try:
                previous = read_json(lock)
                break
            except json.JSONDecodeError:
                time.sleep(0.1)
        else:
            raise ValueError("Incomplete experiment manifest")
        if previous != value:
            raise ValueError("Output belongs to another protocol/model/runtime mode")


def mock_generate(jobs, tasks):
    from protocol import candidate_map
    from common import micro
    results = []
    for job in jobs:
        mapping = candidate_map(tasks[job["task_id"]])
        best = min(micro(c["projected_full_session_tariff_cost"]) for c in mapping.values())
        selected = next(key for key in job["candidate_ids"] if micro(mapping[key]["projected_full_session_tariff_cost"]) == best)
        results.append({"raw_output": canonical({"candidate_id": selected}), "error": None,
                        "mock": True, "input_tokens": None, "output_tokens": None})
    return results


def run(args, config):
    from protocol import parse_choice
    def stop_worker(signum, frame):
        raise KeyboardInterrupt("Worker stopped; completed jobs remain resumable")
    signal.signal(signal.SIGTERM, stop_worker)
    manifest = prepared(config)
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Invalid shard")
    tasks = {t["task_id"]: t for t in raw_tasks()}
    selected_rounds = [r for r in range(config["rounds"]) if r % args.shards == args.shard]
    if args.round_limit:
        selected_rounds = selected_rounds[:args.round_limit]
    if args.mock:
        locks = {n: {"sha256": "MOCK-NO-MODEL"} for n in config["models"]}
        paths = {}
    else:
        paths = model_paths(args, config)
        locks = frozen_models(args, config, paths)
    bind_output(args.out, manifest, "mock" if args.mock else "transformers", locks)
    for existing in args.out.glob(f"{args.model}_shard*of*"):
        if not existing.name.endswith(f"of{args.shards:02d}"):
            raise ValueError("Do not change the shard count when resuming an output directory")
    worker = args.out / f"{args.model}_shard{args.shard:02d}of{args.shards:02d}"
    worker.mkdir(exist_ok=True)
    with (worker / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = worker / "responses.jsonl"
        attempts = worker / "attempts.jsonl"
        repair_tail(ledger)
        repair_tail(attempts)
        completed = {r["job_id"]: r for r in read_records(ledger)}
        attempt_count = Counter()
        for row in read_records(attempts):
            if row["event"] == "begin":
                attempt_count[row["job_id"]] += 1
        queue_jobs = [j for r in selected_rounds for j in round_jobs(args.model, r, config, list(tasks.values())) if j["job_id"] not in completed]
        if args.limit:
            queue_jobs = queue_jobs[:args.limit]
        if not queue_jobs:
            emit({"status": "complete", "worker": str(worker), "new_calls": 0})
            return
        session = None
        try:
            runtime = {"engine": "mock", "not_research_results": True}
            if not args.mock:
                from backend import InferenceSession
                session = InferenceSession(args.model, paths[args.model], config)
                runtime = session.runtime
            runtime_path = worker / "runtime.json"
            runtime_record = {"runtime": runtime, "model_sha256": locks[args.model]["sha256"]}
            if runtime_path.exists() and read_json(runtime_path) != runtime_record:
                raise ValueError("Runtime changed since this worker's previous calls")
            write_json(runtime_path, runtime_record)
            for offset in range(0, len(queue_jobs), config["batch_size"]):
                batch = queue_jobs[offset:offset + config["batch_size"]]
                pending = list(batch)
                final = {}
                while pending:
                    eligible = []
                    for job in pending:
                        if attempt_count[job["job_id"]] >= config["service_retries"] + 1:
                            final[job["job_id"]] = {"raw_output": None, "error": "service_attempts_exhausted"}
                        else:
                            attempt_count[job["job_id"]] += 1
                            append(attempts, {"event": "begin", "job_id": job["job_id"],
                                              "attempt": attempt_count[job["job_id"]], "utc_unix": time.time(),
                                              "input_sha256": digest(job["messages"])})
                            eligible.append(job)
                    if not eligible:
                        break
                    started = time.monotonic()
                    try:
                        answers = mock_generate(eligible, tasks) if args.mock else session.generate(eligible)
                    except Exception as exc:
                        answers = [{"raw_output": None, "error": f"{type(exc).__name__}: {exc}"} for _ in eligible]
                    if len(answers) != len(eligible):
                        raise RuntimeError("Inference output count mismatch")
                    retry = []
                    for job, answer in zip(eligible, answers):
                        append(attempts, {"event": "end", "job_id": job["job_id"],
                                          "attempt": attempt_count[job["job_id"]], "wall_seconds": time.monotonic() - started,
                                          **answer})
                        if answer.get("error") and answer.get("retryable", True):
                            retry.append(job)
                        else:
                            final[job["job_id"]] = answer
                    pending = retry
                for job in batch:
                    answer = final[job["job_id"]]
                    parsed = parse_choice(answer["raw_output"], tasks[job["task_id"]]) if answer.get("raw_output") is not None else {"category": "service_failure", "start": None, "candidate_id": None}
                    record = {**job, "protocol_sha256": manifest["sha256"], "model_sha256": locks[args.model]["sha256"],
                              "runtime_sha256": digest(runtime), "generation_seed": config["generation_seed"],
                              "attempts": attempt_count[job["job_id"]], "inference": answer, "parsed": parsed}
                    append(ledger, record)
                emit({"worker": worker.name, "completed_new": min(offset + len(batch), len(queue_jobs)),
                      "pending_initial": len(queue_jobs), "last_round": batch[-1]["round"],
                      "categories": dict(Counter(final[j["job_id"]].get("error") or "generated" for j in batch))})
        finally:
            if session:
                session.close()


def status(args, config):
    records = collect(args.out)
    counts = Counter((r["model"], r["condition"]) for r in records)
    for model in config["models"]:
        for arm in config["conditions"]:
            emit({"model": model, "condition": arm, "completed": counts[model, arm],
                  "expected": config["rounds"] * config["expected_tasks"]})
    emit({"total": len(records), "expected": len(config["models"]) * 2 * config["rounds"] * config["expected_tasks"],
          "categories": dict(Counter(r["parsed"]["category"] for r in records))})


def collect(out):
    result = []
    ids = set()
    for path in sorted(out.glob("*_shard*of*/responses.jsonl")):
        for row in read_records(path):
            if row["job_id"] in ids:
                raise ValueError(f"Duplicate job across shards: {row['job_id']}")
            ids.add(row["job_id"])
            result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "check", "freeze-models", "preflight", "run", "status", "analyze"])
    parser.add_argument("--models", type=Path, default=ROOT / "models.local.json")
    parser.add_argument("--model", choices=["qwen", "gemma"], default="qwen")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--round-limit", type=int, help="debug only; formal completeness will fail")
    parser.add_argument("--limit", type=int, help="debug only; formal completeness will fail")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    config = read_json(ROOT / "config.json")
    if args.command == "prepare":
        prepare(config)
    elif args.command == "check":
        manifest = prepared(config)
        counts = validate_inputs(config)
        from assignment import PlanSolver
        from common import load_tasks
        solver = PlanSolver(load_tasks(ROOT / "data/tasks.jsonl"), config["start_capacity"], config["cost_budget_micro"])
        if not solver.validation["feasible"]:
            raise ValueError(solver.validation)
        emit({"status": "ok", "protocol": manifest["sha256"], **counts, "zero_budget_capacity_feasible": True})
    elif args.command == "freeze-models":
        prepared(config)
        from backend import model_identity
        paths = model_paths(args, config)
        locks = {name: model_identity(paths[name], spec) for name, spec in config["models"].items()}
        target = ROOT / "prepared/models.lock.json"
        if target.exists() and read_json(target) != locks:
            raise ValueError("Model lock changed; use a new experiment directory")
        write_json(target, locks)
        emit({"status": "models_frozen", "models": {n: l["sha256"] for n, l in locks.items()}})
    elif args.command == "preflight":
        prepared(config)
        paths = model_paths(args, config)
        locks = frozen_models(args, config, paths)
        from backend import InferenceSession
        from protocol import parse_choice
        tasks = raw_tasks()
        small, large = min(tasks, key=lambda t: len(t["candidate_options"])), max(tasks, key=lambda t: len(t["candidate_options"]))
        jobs = [j for j in round_jobs(args.model, 0, config, [small, large])]
        session = InferenceSession(args.model, paths[args.model], config)
        try:
            answers = session.generate(jobs)
            lookup = {t["task_id"]: t for t in tasks}
            records = [{"job_id": j["job_id"], "answer": a, "parsed": parse_choice(a.get("raw_output"), lookup[j["task_id"]])} for j, a in zip(jobs, answers)]
            write_json(ROOT / f"prepared/preflight_{args.model}.json", {"runtime": session.runtime, "records": records})
            emit({"status": "preflight", "model": args.model, "categories": dict(Counter(r["parsed"]["category"] for r in records))})
            if any(a.get("error") for a in answers) or not any(r["parsed"]["start"] for r in records):
                raise RuntimeError("Preflight failed; inspect prepared/preflight JSON before formal calls")
        finally:
            session.close()
    elif args.command == "run":
        run(args, config)
    elif args.command == "status":
        status(args, config)
    else:
        prepared(config)
        from analysis import analyze
        analyze(args.out, config, args.allow_partial)


if __name__ == "__main__":
    main()

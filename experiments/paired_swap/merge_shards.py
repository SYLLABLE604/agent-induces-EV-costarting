#!/usr/bin/env python3
"""Strictly merge four complete worker shards, then audit and analyze offline."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from design import conditions, select_tasks
from parallel import validate_partition
from protocol import digest
from run import ROOT, prepare
from transport import atomic, load_inputs, load_outcomes, read_json


def validate_cache(path: Path, fingerprint: str, request_hash: str) -> None:
    record = read_json(path)
    if record["request_hash"] != request_hash:
        raise ValueError(f"Cache request hash differs: {path}")
    if record["identity"]["protocol"] != fingerprint:
        raise ValueError(f"Cache protocol differs: {path}")
    if digest(record["response"]) != record["response_sha256"]:
        raise ValueError(f"Cache response checksum differs: {path}")
    if digest({"identity": record["identity"], "body": record["request"]}) != request_hash:
        raise ValueError(f"Cache key differs: {path}")


def merge(root: Path, scope: str, config_path: Path, analyze: bool = True) -> Path:
    root = root.resolve()
    parallel = read_json(root / "parallel_manifest.json")
    if (parallel["schema_version"] != "presentation_control_parallel_v2_transformers"
            or parallel["scope"] != scope):
        raise ValueError("Parallel manifest scope/schema differs")
    config, all_tasks, _, _ = load_inputs(config_path)
    tasks = select_tasks(all_tasks, scope)
    manifests = [read_json(root / "manifests" / f"worker_{i}.json") for i in range(4)]
    validate_partition(manifests, tasks, config)
    hashes = {row["worker_index"]: row["sha256"] for row in parallel["workers"]}
    if any(digest(m) != hashes.get(m["worker_index"]) for m in manifests):
        raise ValueError("Worker manifest checksum differs")
    if any(m["study_fingerprint"] != parallel["study_fingerprint"] for m in manifests):
        raise ValueError("Worker study fingerprint differs")
    full_expected = {(model, task["task_id"], arm)
                     for model in config["models"] for task in tasks
                     for arm in conditions(config)}
    all_records = {}
    runtimes = []
    cache_sources = {}
    for manifest in manifests:
        worker = root / f"worker_{manifest['worker_index']}"
        if read_json(worker / "worker.json") != manifest:
            raise ValueError(f"Worker {manifest['worker_index']} frozen manifest differs")
        protocol = read_json(worker / "protocol.json")
        if protocol["fingerprint"] != parallel["study_fingerprint"]:
            raise ValueError("Worker scientific protocol differs")
        model_lock = protocol["identity"]["model_locks"][manifest["model"]]
        if manifest["model_content_sha256"] != model_lock["content_sha256"]:
            raise ValueError("Worker model identity differs from protocol")
        records = load_outcomes(worker, protocol["fingerprint"])
        expected = {(manifest["model"], task_id, arm)
                    for task_id in manifest["task_ids"] for arm in conditions(config)}
        if set(records) != expected:
            missing = len(expected - set(records))
            extra = len(set(records) - expected)
            raise ValueError(f"Worker {manifest['worker_index']} incomplete: missing={missing}, extra={extra}")
        overlap = set(all_records) & set(records)
        if overlap:
            raise ValueError(f"Duplicate outcome across workers: {next(iter(overlap))}")
        all_records.update(records)
        runtime = read_json(worker / "runtime.json")
        execution = runtime.get("execution", {})
        if execution.get("worker_index") != manifest["worker_index"]:
            raise ValueError("Worker runtime index differs")
        if str(execution.get("cuda_visible_devices")) != str(manifest["gpu_index"]):
            raise ValueError("Worker CUDA_VISIBLE_DEVICES differs from manifest")
        if set(runtime["models"]) != {manifest["model"]}:
            raise ValueError("Worker inspected the wrong model")
        runtimes.append(runtime)
        for key, outcome in records.items():
            for trace in outcome["request_traces"]:
                request_hash = trace["request_hash"]
                source = worker / "cache" / f"{request_hash}.json"
                if not source.exists():
                    raise ValueError(f"Missing cache for {key}: {request_hash}")
                validate_cache(source, protocol["fingerprint"], request_hash)
                if request_hash in cache_sources:
                    raise ValueError(f"Duplicate cache across workers: {request_hash}")
                cache_sources[request_hash] = source
    if set(all_records) != full_expected:
        raise ValueError(f"Merged coverage differs: missing={len(full_expected-set(all_records))}, extra={len(set(all_records)-full_expected)}")
    common_runtimes = [
        {key: value for key, value in runtime.items() if key not in {"models", "execution"}}
        for runtime in runtimes
    ]
    if any(runtime != common_runtimes[0] for runtime in common_runtimes[1:]):
        raise ValueError("Workers used different PyTorch/Transformers runtime environments")
    for model in config["models"]:
        records = [runtime["models"][model] for runtime in runtimes if model in runtime["models"]]
        if len(records) != 2 or records[0] != records[1]:
            raise ValueError(f"Two workers disagree on model identity: {model}")
    target = root / "merged"
    if target.exists():
        raise ValueError(f"Merged directory already exists: {target}; keep it or move it before rebuilding")
    temporary = root / f".merged.tmp.{os.getpid()}"
    if temporary.exists():
        raise ValueError(f"Temporary merge directory already exists: {temporary}")
    try:
        first_protocol = read_json(root / "worker_0" / "protocol.json")
        model_locks = first_protocol["identity"]["model_locks"]
        *_, fingerprint = prepare(config_path, temporary, scope, model_locks)
        if fingerprint != parallel["study_fingerprint"]:
            raise ValueError("Current source no longer matches worker protocol")
        for worker in range(4):
            source_dir = root / f"worker_{worker}" / "outcomes"
            for source in source_dir.glob("*.json"):
                destination = temporary / "outcomes" / source.name
                if destination.exists():
                    raise ValueError(f"Duplicate outcome filename: {source.name}")
                destination.parent.mkdir(exist_ok=True)
                shutil.copy2(source, destination)
        for request_hash, source in cache_sources.items():
            destination = temporary / "cache" / f"{request_hash}.json"
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(source, destination)
        atomic(temporary / "runtime.json", {"parallel_workers": runtimes,
                                              "parallel_manifest": parallel})
        atomic(temporary / "status.json", {"status": "complete", "completed": len(all_records),
                                             "expected": len(full_expected), "merged_from_workers": 4})
        atomic(temporary / "worker_sources.json", {"workers": manifests})
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    if analyze:
        subprocess.run([sys.executable, str(ROOT / "run.py"), "audit", "--scope", scope,
                        "--config", str(config_path), "--output", str(target)], check=True)
        subprocess.run([sys.executable, str(ROOT / "run.py"), "analyze", "--scope", scope,
                        "--config", str(config_path), "--output", str(target)], check=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["pilot", "full"], default="full")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--no-analyze", action="store_true")
    args = parser.parse_args()
    root = args.root or ROOT / "results" / f"{args.scope}_4gpu"
    target = merge(root, args.scope, args.config.resolve(), not args.no_analyze)
    print(json.dumps({"status": "merged", "output": str(target),
                      "real_model_calls": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

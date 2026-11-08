#!/usr/bin/env python3
"""Freeze four local-Transformers workers without loading model tensors."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from parallel import (attach_worker, build_worker_manifests,
                      write_parallel_manifest)
from run import ROOT, prepare
from transport import atomic, load_inputs, read_json
from hf_backend import build_model_locks, load_model_paths, verify_model_lock


def parse_gpus(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("GPU IDs must be comma-separated integers") from exc
    if len(result) != 4 or len(set(result)) != 4 or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("Specify exactly four distinct nonnegative GPU IDs")
    return result


def same_or_write(path: Path, value: dict) -> None:
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Frozen file changed: {path}; choose a new root")
    if not path.exists():
        atomic(path, value)


def prepare_cluster(scope: str, root: Path, config_path: Path,
                    gpu_ids: list[int], models_path: Path) -> dict:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config, all_tasks, _, _ = load_inputs(config_path)
    model_paths = load_model_paths(models_path, config)
    locks_path = root / "model_locks.json"
    if locks_path.exists():
        model_locks = read_json(locks_path)
        if set(model_locks) != set(config["models"]):
            raise ValueError("Frozen model lock set differs from configuration")
        for name, path in model_paths.items():
            verify_model_lock(path, model_locks[name], full_hash=False)
    else:
        model_locks = build_model_locks(model_paths, config)
        atomic(locks_path, model_locks)
    manifests = build_worker_manifests(
        config, all_tasks, scope, gpu_ids, model_paths, model_locks
    )
    fingerprints = []
    protocols = []
    for manifest in manifests:
        output = root / f"worker_{manifest['worker_index']}"
        *_, fingerprint = prepare(config_path, output, scope, model_locks)
        fingerprints.append(fingerprint)
        protocols.append(read_json(output / "protocol.json"))
    if len(set(fingerprints)) != 1 or len({json.dumps(p, sort_keys=True) for p in protocols}) != 1:
        raise ValueError("Workers did not freeze the same scientific protocol")
    fingerprint = fingerprints[0]
    frozen = []
    manifests_dir = root / "manifests"
    manifests_dir.mkdir(exist_ok=True)
    for manifest in manifests:
        value = {**manifest, "study_fingerprint": fingerprint}
        path = manifests_dir / f"worker_{manifest['worker_index']}.json"
        same_or_write(path, value)
        attach_worker(root / f"worker_{manifest['worker_index']}", manifest, fingerprint)
        frozen.append(value)
    parallel = write_parallel_manifest(root, fingerprint, scope, frozen)
    if parallel["expected_outcomes"] != protocols[0]["expected_outcomes"]:
        raise ValueError("Shard counts do not equal full study count")
    rows = [[m["worker_index"], m["gpu_index"], m["model"], m["model_path"],
             m["model_content_sha256"],
             str((manifests_dir / f"worker_{m['worker_index']}.json").resolve()),
             str((root / f"worker_{m['worker_index']}").resolve()), m["expected_outcomes"]]
            for m in frozen]
    tsv = "worker\tgpu\tmodel\tmodel_path\tmodel_identity\tmanifest\toutput\texpected\n" + "\n".join(
        "\t".join(map(str, row)) for row in rows) + "\n"
    path = root / "launch.tsv"
    if path.exists() and path.read_text(encoding="utf-8") != tsv:
        raise ValueError("Launch table changed; choose a new root")
    if not path.exists():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(tsv, encoding="utf-8")
        os.replace(temporary, path)
    result = {"scope": scope, "root": str(root), "study_fingerprint": fingerprint,
              "expected_outcomes": parallel["expected_outcomes"],
              "model_identities": {name: lock["content_sha256"]
                                   for name, lock in model_locks.items()},
              "workers": [{k: m[k] for k in ("worker_index", "gpu_index", "model",
                                                "expected_outcomes", "task_weight")}
                          for m in frozen]}
    atomic(root / "cluster.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["pilot", "full"], default="full")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--models", type=Path, required=True,
                        help="JSON mapping logical model names to local directories")
    parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2,3"))
    args = parser.parse_args()
    root = args.root or ROOT / "results" / f"{args.scope}_4gpu"
    print(json.dumps(prepare_cluster(args.scope, root, args.config.resolve(),
                                     args.gpus, args.models.resolve()),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

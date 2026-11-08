#!/usr/bin/env python3
"""Filesystem, cache, and frozen-input utilities for local inference."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from protocol import canonical, digest, minima, timestamp

ROOT = Path(__file__).resolve().parent


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash large model shards without reading them into memory at once."""
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical(value).decode() + "\n")
        handle.flush()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_inputs(config_path: Path) -> tuple[dict, list[dict], list[dict], dict]:
    config = read_json(config_path)
    if config.get("backend") != "transformers_local":
        raise ValueError("This package requires backend=transformers_local")
    if config.get("dtype") != "bfloat16" or config.get("local_files_only") is not True:
        raise ValueError("This protocol requires offline BF16 model loading")
    generation = config.get("generation", {})
    if generation.get("do_sample") is not False:
        raise ValueError("This protocol requires deterministic greedy decoding")
    if type(generation.get("max_new_tokens")) is not int or generation["max_new_tokens"] < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if type(generation.get("max_input_tokens")) is not int or generation["max_input_tokens"] < 1:
        raise ValueError("max_input_tokens must be a positive integer")
    if not isinstance(config.get("models"), dict) or len(config["models"]) != 2:
        raise ValueError("Exactly two logical model specifications are required")
    for name, specification in config["models"].items():
        if not name or set(specification) != {"source", "model_type"}:
            raise ValueError(f"Invalid model specification: {name}")

    manifest = read_json(ROOT / "data/manifest.json")
    for name, expected in manifest["sha256"].items():
        if file_hash(ROOT / "data" / name) != expected:
            raise ValueError(f"Frozen data hash mismatch: {name}")
    tasks = [json.loads(line) for line in
             (ROOT / "data/tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    with (ROOT / "data/pairs.csv").open(encoding="utf-8") as handle:
        pairs = list(csv.DictReader(handle))
    task_map = {task["task_id"]: task for task in tasks}
    if len(task_map) != len(tasks) or len(tasks) != config["expected_tasks"]:
        raise ValueError("Task cardinality differs")
    if sum(len(minima(task)) > 1 for task in tasks) != config["expected_tied_tasks"]:
        raise ValueError("Tied-task cardinality differs")
    if len(pairs) != config["expected_pairs"] or len({p["pair_id"] for p in pairs}) != len(pairs):
        raise ValueError("Pair cardinality differs")
    for pair in pairs:
        first, second = task_map[pair["task_id_a"]], task_map[pair["task_id_b"]]
        if first["household_id"] == second["household_id"]:
            raise ValueError("Not a cross-household pair")
    for task in tasks:
        options = task["candidate_options"]
        starts = [timestamp(option["start_time_utc"]) for option in options]
        if starts != sorted(set(starts)) or not starts:
            raise ValueError("Candidates must be nonempty, unique, and chronological")
        if any(option["cost_rank"] != 1 for option in minima(task)):
            raise ValueError("Minimum candidates must share rank 1")
        if any(timestamp(option["finish_time_utc"]) > timestamp(task["availability_end_utc"])
               for option in options):
            raise ValueError("Infeasible candidate in frozen input")
    prompts = {
        "A": (ROOT / "prompts/sequential.txt").read_text(encoding="utf-8"),
        "one_shot": (ROOT / "prompts/one_shot.txt").read_text(encoding="utf-8"),
    }
    return config, tasks, pairs, prompts


class Client:
    """Atomic response cache around one in-process model generator."""

    def __init__(self, output: Path, config: dict, fingerprint: str, runtime: dict,
                 generate: Callable[[dict], dict] | None,
                 expected: int | None = None):
        self.output = output
        self.config = config
        self.fingerprint = fingerprint
        self.runtime = runtime
        self.generate = generate
        (output / "cache").mkdir(exist_ok=True)
        self.log = output / "api_events.jsonl"
        self.attempts = 0
        if self.log.exists():
            self.attempts = sum('"event":"attempt"' in line
                                for line in self.log.read_text(encoding="utf-8").splitlines())
        self.completed = 0
        self.expected = (read_json(output / "protocol.json")["expected_outcomes"]
                         if expected is None else expected)

    def status(self, state: str, **fields: Any) -> None:
        record = {
            "status": state,
            "utc": now(),
            "completed": self.completed,
            "expected": self.expected,
            "actual_generation_attempts": self.attempts,
            **fields,
        }
        atomic(self.output / "status.json", record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def call(self, body: dict, context: dict) -> tuple[dict, str, bool]:
        identity = {"protocol": self.fingerprint, "runtime": self.runtime, "context": context}
        key = digest({"identity": identity, "body": body})
        path = self.output / "cache" / f"{key}.json"
        if path.exists():
            cached = read_json(path)
            if cached["request"] != body or cached["identity"] != identity:
                raise ValueError("Cache identity mismatch")
            if cached["response_sha256"] != digest(cached["response"]):
                raise ValueError("Cache response checksum mismatch")
            response, hit = cached["response"], True
        else:
            if self.generate is None:
                raise RuntimeError("A cache miss requires a loaded local model")
            self.attempts += 1
            append(self.log, {
                "event": "attempt", "utc": now(), "request_hash": key,
                "context": context, "attempt": 1,
            })
            self.status("calling", request_hash=key, **context)
            started = time.monotonic()
            try:
                response = self.generate(body)
            except BaseException as exc:
                append(self.log, {
                    "event": "error", "utc": now(), "request_hash": key,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                raise
            atomic(path, {
                "identity": identity,
                "request_hash": key,
                "request": body,
                "response": response,
                "response_sha256": digest(response),
                "saved_utc": now(),
                "wall_seconds": time.monotonic() - started,
            })
            append(self.log, {"event": "success", "utc": now(), "request_hash": key})
            hit = False
        if response.get("done_reason") == "length":
            raise ValueError("Generation truncated; increase max_new_tokens for every arm in a new run")
        if response.get("prompt_eval_count", 0) > self.config["generation"]["max_input_tokens"]:
            raise ValueError("Context limit exceeded")
        if response.get("done") is not True:
            raise ValueError("Incomplete local generation response")
        return response, key, hit


def load_outcomes(output: Path, fingerprint: str) -> dict:
    result = {}
    outcomes = output / "outcomes"
    if not outcomes.exists():
        return result
    for path in sorted(outcomes.glob("*.json")):
        record = read_json(path)
        if record["protocol"] != fingerprint or digest(record["outcome"]) != record["sha256"]:
            raise ValueError(f"Outcome checksum or identity differs: {path.name}")
        row = record["outcome"]
        key = (row["model"], row["task_id"], row["arm"])
        if key in result:
            raise ValueError(f"Duplicate outcome: {key}")
        result[key] = row
    return result

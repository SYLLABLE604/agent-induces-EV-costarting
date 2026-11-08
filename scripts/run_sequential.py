#!/usr/bin/env python3
"""Fresh Transformers start/wait run; not a recreation of historical Ollama outputs."""
import argparse
import importlib.util
import json
from pathlib import Path

from _shared import ROOT, read_rows, write_json
from backend import InferenceSession, model_identity

spec = importlib.util.spec_from_file_location("sequential_protocol", ROOT / "experiments/paired_swap/protocol.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--model", choices=["qwen", "gemma"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--timezone", default="Europe/Dublin")
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error("Require 0 <= shard < shards")
    tasks = sorted(read_rows(args.tasks), key=lambda task: task["task_id"])[args.shard::args.shards]
    if not tasks:
        raise ValueError("Empty task shard")
    if args.out.exists():
        raise FileExistsError("Choose a new output directory for a fresh run")
    config = json.loads((ROOT / "experiments/two_stage/config.json").read_text())
    model_path = Path(json.loads(args.models.read_text())[args.model]).expanduser().resolve()
    lock = model_identity(model_path, config["models"][args.model])
    prompt = (ROOT / "experiments/paired_swap/prompts/sequential.txt").read_text()
    args.out.mkdir(parents=True)
    write_json(args.out / "protocol.json", {"protocol": "sequential_transformers_public_port_v1", "historical_backend": "Ollama; original outputs not included", "model": args.model, "model_identity": lock, "config": config, "prompt": prompt, "timezone": args.timezone, "shard": args.shard, "shards": args.shards, "input_sha256": __import__("hashlib").sha256(args.tasks.read_bytes()).hexdigest(), "failure_rule": "earliest_minimum_for_unresolved_or_all_wait"})
    session = InferenceSession(args.model, model_path, config)
    starts = {}
    unresolved = []
    try:
        write_json(args.out / "runtime.json", session.runtime)
        with (args.out / "responses.jsonl").open("w", encoding="utf-8") as stream:
            for task in tasks:
                selected = None
                for index, candidate in enumerate(task["candidate_options"]):
                    payload = legacy.sequential_payload(task, index, args.timezone)
                    job = {"job_id": f"{args.model}/{task['task_id']}/{index}", "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(payload)}]}
                    answer = session.generate([job])[0]
                    valid, start_now, error = legacy.parse(answer.get("raw_output"), task, "A", args.timezone)
                    if answer.get("error"):
                        valid, error = False, answer["error"]
                    stream.write(json.dumps({"task_id": task["task_id"], "index": index, "job": job, "answer": answer, "parsed_valid": valid, "start_now": start_now if valid else None, "parse_error": error}) + "\n")
                    stream.flush()
                    if not valid:
                        break
                    if start_now:
                        selected = candidate["start_time_utc"]
                        break
                if selected is None:
                    unresolved.append(task["task_id"])
                    selected = min(c["start_time_utc"] for c in legacy.minima(task))
                starts[task["task_id"]] = selected
    finally:
        session.close()
    write_json(args.out / "proposals.json", {args.model: starts})
    write_json(args.out / "completion_summary.json", {"tasks": len(tasks), "unresolved_or_all_wait": len(unresolved), "affected_task_ids": unresolved, "complete": True})


if __name__ == "__main__":
    main()

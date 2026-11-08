"""Shared exact-cost and frozen-input helpers. Standard library only."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

SCALE = 1_000_000


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def micro(value):
    scaled = Decimal(str(value)) * SCALE
    if scaled != scaled.to_integral_value():
        raise ValueError(f"Cost needs more than six decimals: {value}")
    return int(scaled)


def slot(start):
    time = datetime.fromisoformat(start.replace("Z", "+00:00"))
    if time.tzinfo is None:
        raise ValueError("Candidate times must contain a timezone")
    return int(time.timestamp() // 1800)


@dataclass(frozen=True)
class Candidate:
    start: str
    bin: int
    cost: int
    regret: int


@dataclass
class Task:
    id: str
    household: str
    minimum: int
    candidates: tuple[Candidate, ...]

    @property
    def by_start(self):
        return {c.start: c for c in self.candidates}

    @property
    def optima(self):
        return {c.start for c in self.candidates if c.regret == 0}


def load_tasks(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    tasks = []
    for row in rows:
        options = row["candidate_options"]
        minimum = min(micro(o["projected_full_session_tariff_cost"]) for o in options)
        candidates = tuple(Candidate(
            o["start_time_utc"], slot(o["start_time_utc"]),
            micro(o["projected_full_session_tariff_cost"]),
            micro(o["projected_full_session_tariff_cost"]) - minimum,
        ) for o in options)
        if len({c.start for c in candidates}) != len(candidates):
            raise ValueError("Repeated exact candidate time")
        tasks.append(Task(row["task_id"], row["household_id"], minimum, candidates))
    if len({t.id for t in tasks}) != len(tasks):
        raise ValueError("Repeated task ID")
    return tasks


def load_pairs(path):
    import csv
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return [(r["task_id_a"], r["task_id_b"]) for r in csv.DictReader(stream)]

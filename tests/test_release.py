"""Executable checks using invented tasks only; no author records or GPU access."""
import importlib.util
import itertools
import json
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _shared import eligible_pairs, load_tasks, slot
from analyze_replay import summarize
from assignment import PlanSolver
from common import Candidate, Task
from protocol import parse_choice


def synthetic_tasks():
    """Invented year-2100 fixtures generated in code, not anonymized research data."""
    starts = [f"2100-01-01T{hour:02d}:{minute:02d}:00Z" for hour, minute in [(0, 0), (0, 30), (1, 0), (1, 30)]]
    shapes = [("fiction-a", "fiction-home-1", [(0, 0), (1, 0), (2, 1)]),
              ("fiction-b", "fiction-home-2", [(0, 0), (1, 1), (2, 0)]),
              ("fiction-c", "fiction-home-1", [(0, 0), (2, 0), (3, 1)])]
    rows = []
    for task_id, home, choices in shapes:
        rows.append({"task_id": task_id, "household_id": home, "release_time_utc": starts[0],
                     "availability_end_utc": "2100-01-01T03:00:00Z", "latest_start_utc": starts[3],
                     "energy_required_kwh": 1.85, "charger_power_kw": 7.4, "continuous_duration_hours": 0.25,
                     "release_context": {"household_import_kwh_30min": None, "solar_generation_kwh_30min": None,
                                         "home_battery_soc_percent": None, "heat_pump_kwh_30min": None, "missing": []},
                     "candidate_options": [{"start_time_utc": starts[index], "finish_time_utc": (datetime.fromisoformat(starts[index].replace("Z", "+00:00")) + timedelta(minutes=15)).isoformat().replace("+00:00", "Z"),
                                            "projected_full_session_tariff_cost": cost, "cost_rank": cost + 1,
                                            "cost_delta_from_min": cost, "normalized_cost_0_1": cost} for index, cost in choices]})
    return rows


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.tasks_file = self.folder / "synthetic.jsonl"
        self.rows = synthetic_tasks()
        self.tasks_file.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        self.tasks = load_tasks(self.tasks_file)
        self.pairs = eligible_pairs(self.tasks)

    def test_analytical_rates_and_tie_strata(self):
        value = summarize(self.tasks, self.pairs)
        self.assertEqual(len(self.pairs), 2)
        self.assertAlmostEqual(value["overall"]["independent_minimum_cost"], 0.375)
        self.assertAlmostEqual(value["overall"]["independent_feasible"], 5 / 18)
        self.assertEqual(value["overall"]["common_earliest_minimum"], 1)
        self.assertEqual(value["tie_strata"]["2"]["pairs"], 2)

    def test_incomplete_proposals_rejected(self):
        with self.assertRaises(ValueError):
            summarize(self.tasks, self.pairs, {"model": {self.tasks[0].id: min(self.tasks[0].optima)}})

    def test_flow_matches_bruteforce(self):
        for capacity in [1, 2]:
            solver = PlanSolver(self.tasks, capacity, 0)
            for proposal in itertools.product(*[sorted(task.optima) for task in self.tasks]):
                starts = dict(zip([task.id for task in self.tasks], proposal))
                candidates = []
                for final in itertools.product(*[sorted(task.optima) for task in self.tasks]):
                    if max(sum(slot(s) == slot(other) for other in final) for s in final) <= capacity:
                        candidates.append(sum(a != b for a, b in zip(proposal, final)))
                result = solver.solve(starts)
                self.assertTrue(result["feasible"])
                self.assertEqual(result["modified"], min(candidates))
                self.assertEqual(set(result["starts"]), set(starts))

    def test_joint_infeasibility_reported(self):
        start = "2100-01-01T00:00:00Z"
        tasks = [Task(f"invented-{i}", str(i), 0, (Candidate(start, slot(start), 0, 0),)) for i in range(2)]
        result = PlanSolver(tasks, 1, 0).solve({task.id: start for task in tasks})
        self.assertFalse(result["feasible"])

    def test_response_parser(self):
        self.assertEqual(parse_choice('```json\n{"candidate_id":"c0000"}\n```', self.rows[0])["category"], "valid_optimal")
        self.assertEqual(parse_choice('{"candidate_id":"c0000","candidate_id":"c0001"}', self.rows[0])["category"], "invalid_json")
        self.assertEqual(parse_choice('{"candidate_id":"missing"}', self.rows[0])["category"], "unknown_candidate")

    def test_paired_swap_changes_only_two_positions(self):
        base = ROOT / "experiments/paired_swap"
        spec = importlib.util.spec_from_file_location("swap_protocol_fixture", base / "protocol.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.dict(sys.modules, {"protocol": module}):
            spec = importlib.util.spec_from_file_location("swap_design_fixture", base / "design.py")
            design = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(design)
        config = {"shuffle_seed": 123, "calibration_repeats": 5}
        for repeat in range(5):
            plus = design.indices(self.rows[0], f"test_{repeat:02d}_plus", config)
            minus = design.indices(self.rows[0], f"test_{repeat:02d}_minus", config)
            self.assertEqual(sorted(plus), [0, 1, 2])
            changed = [n for n, (a, b) in enumerate(zip(plus, minus)) if a != b]
            self.assertEqual(len(changed), 2)
            self.assertEqual({plus[n] for n in changed}, {0, 1})
            self.assertEqual(plus.index(2), minus.index(2))

    def test_local_import_and_complete_mock_pipeline(self):
        exported = self.folder / "export"
        shutil.copytree(ROOT, exported, ignore=shutil.ignore_patterns("__pycache__", "results", "prepared", "MANIFEST.sha256"))
        def run(*args, cwd=exported):
            value = subprocess.run([sys.executable, *map(str, args)], cwd=cwd, capture_output=True, text=True, timeout=35)
            self.assertEqual(value.returncode, 0, value.stdout + value.stderr)
            return value
        run("scripts/import_inputs.py", "--tasks", self.tasks_file, "--experiment", "both")
        folder = exported / "experiments/two_stage"
        run("run.py", "prepare", cwd=folder)
        run("run.py", "check", cwd=folder)
        for model in ["qwen", "gemma"]:
            for shard in [0, 1]:
                run("run.py", "run", "--mock", "--model", model, "--shard", shard, "--shards", 2, cwd=folder)
        run("run.py", "analyze", cwd=folder)
        summary = json.loads((folder / "results/analysis/summary.json").read_text())
        self.assertEqual(summary["mode"], "mock")
        self.assertTrue(summary["complete"])
        plans = (folder / "results/analysis/plans.jsonl").read_text().splitlines()
        self.assertEqual(len(plans), 80)
        # Re-running a completed worker must not add model calls or duplicate records.
        before = (folder / "results/qwen_shard00of02/responses.jsonl").read_bytes()
        run("run.py", "run", "--mock", "--model", "qwen", "--shard", 0, "--shards", 2, cwd=folder)
        self.assertEqual(before, (folder / "results/qwen_shard00of02/responses.jsonl").read_bytes())

    def test_swap_extractor_distinguishes_first_and_repaired_choices(self):
        import hashlib
        exported = self.folder / "export"
        shutil.copytree(ROOT, exported, ignore=shutil.ignore_patterns("__pycache__", "results", "prepared", "MANIFEST.sha256"))
        subprocess.run([sys.executable, "scripts/import_inputs.py", "--tasks", str(self.tasks_file), "--experiment", "paired_swap"], cwd=exported, check=True, capture_output=True)
        run = self.folder / "invented_swap_run"
        (run / "outcomes").mkdir(parents=True)
        config = {"test_repeats": 5, "models": {"qwen3.5-9b": {}}}
        identity = {"config": config, "task_ids": [row["task_id"] for row in self.rows]}
        (run / "protocol.json").write_text(json.dumps({"fingerprint": "synthetic-only", "identity": identity}))
        def digest(obj):
            return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        for task in self.rows:
            optima = sorted(c["start_time_utc"] for c in task["candidate_options"] if c["cost_rank"] == 1)
            for repeat in range(5):
                for index, sign in enumerate(["plus", "minus"]):
                    outcome = {"model": "qwen3.5-9b", "task_id": task["task_id"], "arm": f"test_{repeat:02d}_{sign}", "first_start_utc": optima[index], "system_start_utc": optima[0]}
                    path = run / "outcomes" / f"{task['task_id']}-{repeat}-{sign}.json"
                    path.write_text(json.dumps({"protocol": "synthetic-only", "outcome": outcome, "sha256": digest(outcome)}))
        for choice, expected in [("first", 15), ("final", 0)]:
            out = self.folder / f"{choice}.json"
            done = subprocess.run([sys.executable, "scripts/summarize_swaps.py", "--run", str(run), "--choice", choice, "--out", str(out)], cwd=exported, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            value = json.loads(out.read_text())["models"]["qwen3.5-9b"]
            self.assertEqual(value["paired_comparisons"], 15)
            self.assertEqual(value["changed_known"], expected)
            self.assertEqual(value["both_optimal_changes"], expected)


if __name__ == "__main__":
    unittest.main()

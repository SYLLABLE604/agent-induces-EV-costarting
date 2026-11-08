"""Calibration-only fit; held-out finite-table policy evaluation; no inference."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from pathlib import Path

from design import conditions, indices, rng
from optimization import coordinate, repair
from protocol import bin_id, digest, minima, pair_rate, timestamp
from transport import atomic, load_outcomes, read_json

ROOT = Path(__file__).resolve().parent


def distribution(rows, task, arm_prefix, sign, config):
    n = config["calibration_repeats"] if arm_prefix == "cal" else config["test_repeats"]
    valid = {o["start_time_utc"] for o in minima(task)}
    starts = [rows[f"{arm_prefix}_{r:02d}_{sign}"]["system_start_utc"] for r in range(n)]
    counts = Counter(bin_id(s) for s in starts if s in valid)
    p = {k: Fraction(v, n) for k, v in counts.items()}
    return p, Fraction(sum(s not in valid for s in starts), n)


def response_weights(tasks, pairs, records, config):
    differences, errors = {}, {}
    for task in tasks:
        tid = task["task_id"]
        pp, ep = distribution(records[tid], task, "cal", "plus", config)
        pm, em = distribution(records[tid], task, "cal", "minus", config)
        differences[tid] = {k: pp.get(k, 0) - pm.get(k, 0) for k in pp.keys() | pm.keys()}
        errors[tid] = ep - em
    weights = {}
    for pair in pairs:
        i, j = pair["task_id_a"], pair["task_id_b"]
        di, dj = differences[i], differences[j]
        weights[i, j] = sum(v * dj.get(k, 0) for k, v in di.items()) - errors[i] * errors[j]
    return weights


def plan_metrics(tasks, pairs, starts):
    opt = {t["task_id"]: {o["start_time_utc"] for o in minima(t)} for t in tasks}
    bins = Counter(bin_id(s) for s in starts.values() if s is not None)
    missing = sum(starts.get(t["task_id"]) is None for t in tasks)
    quality_bad = sum(starts.get(i) not in opts for i, opts in opt.items())
    conservative = sum(starts.get(p["task_id_a"]) not in opt[p["task_id_a"]]
                       or starts.get(p["task_id_b"]) not in opt[p["task_id_b"]]
                       or bin_id(starts[p["task_id_a"]]) == bin_id(starts[p["task_id_b"]]) for p in pairs)
    return {"co_start": pair_rate(pairs, starts) if pairs else None,
            "quality_conservative_pair_score": conservative / len(pairs) if pairs else None,
            "missing": missing, "not_authorized_optimal": quality_bad,
            "peak_known": max(bins.values(), default=0),
            "peak_upper": max(bins.values(), default=0) + missing}


def evaluate_model(tasks, pairs, records, config):
    ids = [t["task_id"] for t in tasks]
    weights = response_weights(tasks, pairs, records, config)
    z, components = coordinate(ids, weights, config["exact_component_limit"])
    means, error_means = {}, {}
    for t in tasks:
        i = t["task_id"]
        pp, ep = distribution(records[i], t, "cal", "plus", config)
        pm, em = distribution(records[i], t, "cal", "minus", config)
        means[i] = {k: (pp.get(k, 0) + pm.get(k, 0)) / 2 for k in pp.keys() | pm.keys()}
        error_means[i] = (ep + em) / 2
    base = sum(sum(v * means[j].get(k, 0) for k, v in means[i].items())
               + error_means[i] + error_means[j] - error_means[i] * error_means[j]
               for i, j in weights)
    predictions = ({"independent": float(base / len(pairs)),
                    "shared": float((base + sum(weights.values()) / 4) / len(pairs)),
                    "coordinated": float((base + sum(w * z[i] * z[j] for (i, j), w in weights.items()) / 4) / len(pairs))}
                   if pairs else {})
    responses, swaps = [], []
    for task in tasks:
        tid = task["task_id"]
        early = minima(task)[0]["start_time_utc"]
        row = {"task_id": tid, "optimal_candidates": len(minima(task))}
        cal_orders = {tuple(indices(task, f"cal_{r:02d}_plus", config)) for r in range(config["calibration_repeats"])}
        test_orders = {tuple(indices(task, f"test_{r:02d}_plus", config)) for r in range(config["test_repeats"])}
        row["cal_distinct_paired_backgrounds"] = len(cal_orders)
        row["test_distinct_paired_backgrounds"] = len(test_orders)
        row["test_unseen_paired_backgrounds"] = len(test_orders - cal_orders)
        for split, n in (("cal", config["calibration_repeats"]), ("test", config["test_repeats"])):
            for sign in ("plus", "minus"):
                actions = [records[tid][f"{split}_{r:02d}_{sign}"]["system_start_utc"] for r in range(n)]
                row[f"{split}_{sign}_earliest"] = sum(s == early for s in actions) / n
                row[f"{split}_{sign}_missing"] = sum(s is None for s in actions) / n
            changed = both_optimal = unknown = 0
            opts = {o["start_time_utc"] for o in minima(task)}
            for r in range(n):
                a, b = [records[tid][f"{split}_{r:02d}_{sign}"]["system_start_utc"] for sign in ("plus", "minus")]
                unknown += a is None or b is None
                if a is not None and b is not None and a != b:
                    changed += 1
                    both_optimal += a in opts and b in opts
            swaps.append({"task_id": tid, "split": split, "pairs": n, "changed_known": changed,
                          "both_optimal_changes": both_optimal, "unknown": unknown})
        responses.append(row)
    plans, repair_cache = [], {}

    def evaluate(name, repeat, realization, starts):
        result = {"method": name, "repeat": repeat, "realization": realization,
                  "starts": starts, "metrics": plan_metrics(tasks, pairs, starts), "repair": {}}
        for capacity in config["capacity_levels"]:
            key = digest([capacity, starts])
            if key not in repair_cache:
                repaired = repair(tasks, starts, capacity)
                if repaired["feasible"]:
                    final = repaired["starts"]
                    repaired["final_metrics"] = plan_metrics(tasks, pairs, final)
                    repaired["total_wait_minutes"] = sum((timestamp(final[t["task_id"]]) - timestamp(t["release_time_utc"])).total_seconds() / 60 for t in tasks)
                    assert repaired["final_metrics"]["peak_known"] <= capacity
                    assert repaired["final_metrics"]["not_authorized_optimal"] == 0
                repair_cache[key] = repaired
            result["repair"][str(capacity)] = repair_cache[key]
        plans.append(result)

    evaluate("chronological", -1, 0, {i: records[i]["baseline"]["system_start_utc"] for i in ids})
    for r in range(config["test_repeats"]):
        choices = {i: {s: records[i][f"test_{r:02d}_{s}"]["system_start_utc"] for s in ("plus", "minus")} for i in ids}
        for name, vectors in (
                ("shared", [{i: 1 for i in ids}]),
                ("coordinated", [z]),
                ("independent", [{i: rng(config["shuffle_seed"], "policy", r, b, i).choice((-1, 1)) for i in ids}
                                 for b in range(config["independent_plan_repeats"])])):
            for b, vector in enumerate(vectors):
                # Antithetic pairing: every task uses each template exactly half the time.
                for flip in (1, -1):
                    starts = {i: choices[i]["plus" if flip * vector[i] == 1 else "minus"] for i in ids}
                    evaluate(name, r, 2 * b + int(flip == -1), starts)
    summary = {}
    for name in ("chronological", "shared", "independent", "coordinated"):
        selected = [p for p in plans if p["method"] == name]
        avg = lambda fn: sum(fn(p) for p in selected) / len(selected)
        summary[name] = {"plan_realizations": len(selected),
                         "co_start_lower": avg(lambda p: p["metrics"]["co_start"]["lower"]) if pairs else None,
                         "co_start_upper": avg(lambda p: p["metrics"]["co_start"]["upper"]) if pairs else None,
                         "quality_bad_mean": avg(lambda p: p["metrics"]["not_authorized_optimal"]),
                         "conservative_score": avg(lambda p: p["metrics"]["quality_conservative_pair_score"]) if pairs else None,
                         "peak_known_mean": avg(lambda p: p["metrics"]["peak_known"]),
                         "repair": {}}
        for cap in map(str, config["capacity_levels"]):
            feasible = all(p["repair"][cap]["feasible"] for p in selected)
            summary[name]["repair"][cap] = {"feasible": feasible}
            if feasible:
                summary[name]["repair"][cap].update({k: avg(lambda p: p["repair"][cap][k]) for k in
                                                     ("modified", "mandatory_quality_changes", "additional_capacity_changes", "total_wait_minutes")})
    # First-attempt, fixed-denominator summary keeps format repair distinct.
    return {"responses": responses, "paired_swaps": swaps, "calibration_predicted_conservative_score": predictions,
            "weights": [[i, j, float(w)] for (i, j), w in weights.items()],
            "coordinated_signs": z, "components": components, "summary": summary, "plans": plans}


def analyze(output):
    import json
    protocol = read_json(output / "protocol.json")
    identity = protocol["identity"]
    # Refuse analysis with changed source inputs or analysis implementation.
    from transport import file_hash
    for name, expected in identity["code_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Analysis source changed: {name}")
    config = identity["config"]
    tasks = [json.loads(s) for s in (ROOT / "data/tasks.jsonl").read_text().splitlines()]
    tasks = [t for t in tasks if t["task_id"] in identity["task_ids"]]
    records = load_outcomes(output, protocol["fingerprint"])
    result = {"protocol": protocol["fingerprint"], "scope": identity["scope"], "models": {},
              "interpretation": "Held-out offline recombination of stateless measured potential responses; not a prospective coordinated fleet run. Reused realizations are not independent new tasks."}
    for model in config["models"]:
        keys = [(model, t["task_id"], a) for t in tasks for a in conditions(config)]
        missing = [k for k in keys if k not in records]
        if missing:
            result["models"][model] = {"status": "incomplete", "missing_outcomes": len(missing)}
            continue
        table = {t["task_id"]: {a: records[model, t["task_id"], a] for a in conditions(config)} for t in tasks}
        evaluated = evaluate_model(tasks, identity["pairs"], table, config)
        evaluated["output_quality_by_arm"] = {
            a: {"tasks": len(tasks), "valid": sum(table[t["task_id"]][a]["system_start_utc"] is not None for t in tasks),
                "first_valid": sum(table[t["task_id"]][a]["first_start_utc"] is not None for t in tasks),
                "first_optimal": sum(table[t["task_id"]][a]["first_start_utc"] in {o["start_time_utc"] for o in minima(t)} for t in tasks),
                "optimal": sum(table[t["task_id"]][a]["metrics"]["optimal"] for t in tasks),
                "format_repairs": sum(table[t["task_id"]][a]["repair_calls"] for t in tasks)} for a in conditions(config)}
        result["models"][model] = {"status": "complete", **evaluated}
    atomic(output / "analysis.json", result)
    lines = ["# 候选呈现控制：实验结果", "", f"范围：{identity['scope']}；任务数：{len(tasks)}；固定任务对：{len(identity['pairs'])}。",
             "", "下表为留出响应表的离线策略评价，不是新增联合部署；上下界来自缺失动作，不是置信区间。", ""]
    for model, value in result["models"].items():
        lines += [f"## {model}", ""]
        if value["status"] != "complete":
            lines += [f"未完成，缺少 {value['missing_outcomes']} 条记录；不输出完整策略结论。", ""]
            continue
        lines += ["|方法|共同启动下界|上界|平均不合格任务|已知峰值均值|", "|---|---:|---:|---:|---:|"]
        for method, stats in value["summary"].items():
            lo, hi = stats["co_start_lower"], stats["co_start_upper"]
            rates = "N/A|N/A" if lo is None else f"{lo:.2%}|{hi:.2%}"
            lines.append(f"|{method}|{rates}|{stats['quality_bad_mean']:.2f}|{stats['peak_known_mean']:.2f}|")
        lines += ["", "### 同一授权集合下的执行调整", "", "|方法|容量/辆|可行|平均最少改动|其中必需质量改动|额外容量改动|", "|---|---:|---|---:|---:|---:|"]
        for method, stats in value["summary"].items():
            for cap, repair_stats in stats["repair"].items():
                if repair_stats["feasible"]:
                    cells = f"是|{repair_stats['modified']:.2f}|{repair_stats['mandatory_quality_changes']:.2f}|{repair_stats['additional_capacity_changes']:.2f}"
                else:
                    cells = "否|—|—|—"
                lines.append(f"|{method}|{cap}|{cells}|")
        unseen = sum(r["test_unseen_paired_backgrounds"] > 0 for r in value["responses"])
        lines += ["", f"具有至少一个未见成对背景的任务：{unseen}/{len(tasks)}。有限候选空间发生的复用逐任务记录在 analysis.json。",
                  "", "### 校准预测与测试观察（不合格输出按碰撞计的保守指标）", "",
                  "|方法|校准预测|测试观察|", "|---|---:|---:|"]
        for method, prediction in value["calibration_predicted_conservative_score"].items():
            actual = value["summary"][method]["conservative_score"]
            lines.append(f"|{method}|{prediction:.4f}|{actual:.4f}|")
        lines += ["", "完整首次输出质量、逐任务响应、交换结果、最终计划及不可行证书见 analysis.json。", ""]
    temporary = output / "report.md.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(output / "report.md")
    print(json.dumps({m: {"status": v["status"], "summary": v.get("summary")} for m, v in result["models"].items()}, ensure_ascii=False, indent=2))

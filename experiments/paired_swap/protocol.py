"""Pure, standard-library protocol functions; no model calls or file writes."""
from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Naive timestamp is not permitted")
    return result


def local(value: str, timezone: str) -> str:
    converted = timestamp(value).astimezone(ZoneInfo(timezone))
    # datetime truncates nanoseconds; preserve the frozen pandas timestamp text.
    match = re.search(r"T\d{2}:\d{2}:\d{2}\.(\d+)", value)
    nanos = int((match.group(1) if match else "0").ljust(9, "0"))
    fraction = f".{nanos:09d}" if nanos % 1000 else (f".{nanos // 1000:06d}" if nanos else "")
    base = converted.replace(microsecond=0).isoformat()
    return base[:19] + fraction + base[19:]


def bin_id(value: str) -> int:
    return int(timestamp(value).timestamp() // 1800)


def minima(task: dict) -> list[dict]:
    options = task["candidate_options"]
    minimum = min(o["projected_full_session_tariff_cost"] for o in options)
    return [o for o in options if o["projected_full_session_tariff_cost"] == minimum]


def visible(option: dict, timezone: str) -> dict:
    return {
        "start_time_local": local(option["start_time_utc"], timezone),
        "finish_time_local": local(option["finish_time_utc"], timezone),
        "projected_full_session_tariff_cost": option["projected_full_session_tariff_cost"],
        "cost_rank": option["cost_rank"],
        "cost_delta_from_min": option["cost_delta_from_min"],
        "normalized_cost_0_1": option["normalized_cost_0_1"],
    }


def sequential_payload(task: dict, index: int, timezone: str) -> dict:
    """Byte-equivalent JSON content to the frozen decision_state_v1 builder."""
    options = task["candidate_options"]
    current = timestamp(options[index]["start_time_utc"])
    release = timestamp(task["release_time_utc"])
    latest = timestamp(task["latest_start_utc"])
    context = task["release_context"]
    return {
        "schema_version": "decision_state_v1",
        "decision": {
            "current_time_utc": options[index]["start_time_utc"],
            "current_time_local": local(options[index]["start_time_utc"], timezone),
            "decision_index": index,
            "elapsed_wait_minutes": round((current - release).total_seconds() / 60, 6),
            "is_last_opportunity": index == len(options) - 1,
        },
        "task": {
            "energy_required_kwh": task["energy_required_kwh"],
            "charger_power_kw": task["charger_power_kw"],
            "continuous_duration_hours": task["continuous_duration_hours"],
            "availability_end_local": local(task["availability_end_utc"], timezone),
            "latest_start_local": local(task["latest_start_utc"], timezone),
            "remaining_slack_hours": round(max(0.0, (latest - current).total_seconds() / 3600), 6),
        },
        "release_snapshot": {
            "snapshot_is_frozen": True,
            "household_import_kwh_30min": context["household_import_kwh_30min"],
            "solar_generation_kwh_30min": context["solar_generation_kwh_30min"],
            "home_battery_soc_percent": context["home_battery_soc_percent"],
            "heat_pump_kwh_30min": context["heat_pump_kwh_30min"],
            "missing": context["missing"],
        },
        "current_option": visible(options[index], timezone),
        "future_options": [visible(o, timezone) for o in options[index + 1:]],
    }


def order(task: dict, arm: str, config: dict) -> list[int]:
    indices = list(range(len(task["candidate_options"])))
    if arm.startswith("C"):
        seed = int(digest([config["shuffle_seed"], task["task_id"], arm]), 16)
        random.Random(seed).shuffle(indices)
    return indices


def one_shot_payload(task: dict, arm: str, config: dict) -> dict:
    timezone = config["timezone"]
    base = sequential_payload(task, 0, timezone)
    return {
        "schema_version": "one_shot_state_v1",
        "planning_time_local": local(task["release_time_utc"], timezone),
        # Keep the complete task constraints visible across interfaces.  The
        # prompt and strict parser distinguish selectable options from boundary
        # timestamps; protocol errors are recorded rather than hidden.
        "task": base["task"],
        "release_snapshot": base["release_snapshot"],
        "options": [visible(task["candidate_options"][i], timezone)
                    for i in order(task, arm, config)],
    }


def request_body(model: str, prompt: str, payload: dict, config: dict,
                 arm: str, repair_raw: str | None = None) -> dict:
    sequential = arm == "A"
    key = "start_now" if sequential else "start_time_local"
    messages = [{"role": "system", "content": prompt}, {
        "role": "user", "content": json.dumps(payload, ensure_ascii=False,
                                                separators=(",", ":"), allow_nan=False)}]
    if repair_raw is not None:
        instruction = (
            "The previous response was invalid. Return exactly one JSON "
            "object with the single boolean key start_now and no other text."
        ) if sequential else (
            "The previous response was invalid. Return exactly one JSON object "
            "with the single string key start_time_local and no other text. "
            "Copy the complete start_time_local string from one supplied option."
        )
        messages.extend([{"role": "assistant", "content": repair_raw},
                         {"role": "user", "content": instruction}])
    return {
        "model": model,
        "messages": messages,
        "expected_json": {
            "key": key,
            "type": "boolean" if sequential else "string",
        },
        "generation": config["generation"],
        "chat_template_kwargs": config["chat_template_kwargs"],
    }


def parse(raw: str, task: dict, arm: str, timezone: str) -> tuple[bool, Any, str | None]:
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return False, None, "invalid_json"
    key = "start_now" if arm == "A" else "start_time_local"
    if not isinstance(value, dict) or set(value) != {key}:
        return False, None, "invalid_keys"
    if arm == "A":
        return (True, value[key], None) if type(value[key]) is bool else (False, None, "not_boolean")
    mapping = {local(o["start_time_utc"], timezone): o["start_time_utc"]
               for o in task["candidate_options"]}
    if not isinstance(value[key], str) or value[key] not in mapping:
        return False, None, "not_a_supplied_start"
    return True, mapping[value[key]], None


def arms(config: dict) -> list[str]:
    return ["A", "B"] + [f"C{i}" for i in range(config["shuffle_repeats"])]


def measure(task: dict, selected: str | None, arm: str, config: dict) -> dict:
    options = task["candidate_options"]
    minimum = minima(task)
    min_starts = {o["start_time_utc"] for o in minimum}
    earliest = min(min_starts, key=timestamp)
    presented = [options[i]["start_time_utc"] for i in order(task, arm, config)]
    first_min = next(s for s in presented if s in min_starts)
    return {
        "tied": len(minimum) > 1,
        "feasible": selected in {o["start_time_utc"] for o in options},
        "optimal": selected in min_starts,
        "earliest": selected == earliest,
        "first_presented_minimum": selected == first_min,
        "first_presented_option": selected == presented[0],
        "discriminating_order": first_min != earliest,
        "elapsed_wait_minutes": None if selected is None else
            (timestamp(selected) - timestamp(task["release_time_utc"])).total_seconds() / 60,
    }


def pair_rate(pairs: list[dict], selections: dict[str, str | None]) -> dict:
    collisions = unknown = 0
    for pair in pairs:
        a, b = selections.get(pair["task_id_a"]), selections.get(pair["task_id_b"])
        if a is None or b is None:
            unknown += 1
        else:
            collisions += bin_id(a) == bin_id(b)
    return {"pairs": len(pairs), "known_collisions": collisions, "unknown_pairs": unknown,
            "lower": collisions / len(pairs), "upper": (collisions + unknown) / len(pairs)}

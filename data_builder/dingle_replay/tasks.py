"""Build immutable Dingle charging tasks from the four source households."""

from __future__ import annotations

import json
import math
import platform
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .manifest import (
    atomic_write_json,
    canonical_json_bytes,
    file_record,
    sha256_bytes,
    sha256_file,
)


REQUIRED_ENERGY_COLUMNS = (
    "timestamp",
    "id",
    "solar_kWh",
    "as230_import_kWh",
    "elec_price",
    "heatpump_kWh",
    "battery_soc",
)
REQUIRED_JOURNEY_COLUMNS = (
    "id",
    "start_time",
    "end_time",
    "event_type",
    "CHARGING_TIME",
    "ENERGY",
)
EXACT_DURATION_RULE = "exact_energy_over_power"


class BuildError(RuntimeError):
    """Raised when the frozen task data contract cannot be satisfied."""


def _utc_iso(value: pd.Timestamp) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _local_iso(value: pd.Timestamp, timezone: str) -> str:
    return value.tz_convert(timezone).isoformat()


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "schema_version",
        "algorithm_version",
        "source_root",
        "households",
        "charger_power_kw",
        "slot_minutes",
        "feasibility_duration_rule",
        "ev_capacity_kwh",
        "solar_capacity_kw",
        "periods",
        "development_date_modulus",
        "development_date_remainder",
        "local_timezone",
    }
    if set(config) != required:
        raise BuildError(
            f"config keys differ from frozen contract: expected={sorted(required)}, "
            f"actual={sorted(config)}"
        )
    if config["households"] != ["id01", "id02", "id03", "id04"]:
        raise BuildError("households must be exactly id01 through id04")
    if config["slot_minutes"] != 30 or config["charger_power_kw"] != 7.4:
        raise BuildError("the frozen contract requires 30-minute slots and 7.4 kW")
    if (
        config["schema_version"] != "dingle_task_builder_v2"
        or config["algorithm_version"] != "1.1.0"
        or config["feasibility_duration_rule"] != EXACT_DURATION_RULE
    ):
        raise BuildError("the v1.1 contract requires exact energy-over-power duration")
    return config


def _source_paths(config: dict[str, Any], repo_root: Path) -> list[Path]:
    root = repo_root / config["source_root"]
    paths: list[Path] = []
    for household in config["households"]:
        paths.append(root / "EnergyStreams" / f"{household}_merged_30min.csv")
        paths.append(root / "JourneyCharge" / f"{household}_Sorted_Mobility.csv")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise BuildError(f"missing source files: {missing}")
    return paths


def source_inventory(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in _source_paths(config, repo_root):
        with path.open("r", encoding="utf-8-sig") as handle:
            columns = handle.readline().strip().split(",")
        relative = path.relative_to(repo_root).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "columns": columns,
            }
        )
    collection_sha256 = sha256_bytes(canonical_json_bytes(files))
    return {
        "schema_version": "dingle_source_inventory_v1",
        "collection_sha256": collection_sha256,
        "files": files,
    }


def write_inventory(
    config_path: Path, repo_root: Path, output_root: Path
) -> Path:
    config = load_config(config_path)
    inventory = source_inventory(config, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"source_inventory_{inventory['collection_sha256'][:20]}.json"
    if output.exists():
        with output.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != inventory:
            raise BuildError(f"existing inventory differs: {output}")
        return output
    atomic_write_json(output, inventory)
    return output


def _read_energy_sources(
    config: dict[str, Any], repo_root: Path
) -> dict[str, pd.DataFrame]:
    root = repo_root / config["source_root"] / "EnergyStreams"
    streams: dict[str, pd.DataFrame] = {}
    for household in config["households"]:
        path = root / f"{household}_merged_30min.csv"
        frame = pd.read_csv(path, low_memory=False)
        missing = sorted(set(REQUIRED_ENERGY_COLUMNS) - set(frame.columns))
        if missing:
            raise BuildError(f"{path} missing columns: {missing}")
        frame = frame.loc[:, REQUIRED_ENERGY_COLUMNS].copy()
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"], errors="coerce", utc=True
        )
        if frame["timestamp"].isna().any():
            raise BuildError(f"{path} contains unparseable timestamps")
        if frame["timestamp"].duplicated().any():
            raise BuildError(f"{path} contains duplicate timestamps")
        for column in REQUIRED_ENERGY_COLUMNS[2:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        streams[household] = frame.set_index("timestamp").sort_index()
    return streams


def _canonical_tariff(
    streams: dict[str, pd.DataFrame], tolerance: float = 1e-12
) -> tuple[pd.DataFrame, dict[pd.Timestamp, float], dict[str, Any]]:
    price_series = {
        household: frame["elec_price"].rename(household)
        for household, frame in streams.items()
    }
    joined = pd.concat(price_series.values(), axis=1, join="outer").sort_index()
    available_count = joined.notna().sum(axis=1)
    maximum = joined.max(axis=1, skipna=True)
    minimum = joined.min(axis=1, skipna=True)
    conflict = (available_count >= 2) & ((maximum - minimum).abs() > tolerance)
    if conflict.any():
        examples = [_utc_iso(value) for value in joined.index[conflict][:5]]
        raise BuildError(f"household tariff signals conflict at {examples}")
    canonical = joined.bfill(axis=1).iloc[:, 0].rename("elec_price")
    output = canonical.reset_index().rename(columns={"index": "timestamp"})
    output["available_households"] = available_count.to_numpy()
    mapping = {
        timestamp: float(value)
        for timestamp, value in canonical.items()
        if pd.notna(value)
    }
    audit = {
        "timestamp_count": int(len(joined)),
        "timestamps_with_price": int(canonical.notna().sum()),
        "timestamps_all_four_available": int((available_count == 4).sum()),
        "conflict_count": int(conflict.sum()),
        "minimum_available_households": int(available_count.min()),
    }
    return output, mapping, audit


def _read_charging_sessions(
    config: dict[str, Any], repo_root: Path
) -> pd.DataFrame:
    root = repo_root / config["source_root"] / "JourneyCharge"
    frames: list[pd.DataFrame] = []
    for household in config["households"]:
        path = root / f"{household}_Sorted_Mobility.csv"
        frame = pd.read_csv(path, low_memory=False)
        missing = sorted(set(REQUIRED_JOURNEY_COLUMNS) - set(frame.columns))
        if missing:
            raise BuildError(f"{path} missing columns: {missing}")
        frame["source_row_number"] = np.arange(2, len(frame) + 2)
        frame["source_file"] = path.relative_to(repo_root).as_posix()
        frame["event_type_normalized"] = (
            frame["event_type"].astype(str).str.strip().str.lower()
        )
        frame = frame.loc[frame["event_type_normalized"] == "charging"].copy()
        frames.append(frame)
    sessions = pd.concat(frames, ignore_index=True)
    sessions["start_utc"] = pd.to_datetime(
        sessions["start_time"], errors="coerce", utc=True
    )
    sessions["end_utc"] = pd.to_datetime(
        sessions["end_time"], errors="coerce", utc=True
    )
    sessions["energy_kwh"] = pd.to_numeric(sessions["ENERGY"], errors="coerce")
    return sessions


def _mark_overlaps(sessions: pd.DataFrame) -> pd.Series:
    overlapping = pd.Series(False, index=sessions.index)
    for _, group in sessions.groupby("id", sort=True):
        valid = group.dropna(subset=["start_utc", "end_utc"]).sort_values(
            ["start_utc", "end_utc", "source_row_number"]
        )
        active: list[tuple[int, pd.Timestamp]] = []
        for index, row in valid.iterrows():
            start = row["start_utc"]
            active = [(other, end) for other, end in active if end > start]
            if active:
                overlapping.loc[index] = True
                for other, _ in active:
                    overlapping.loc[other] = True
            active.append((index, row["end_utc"]))
    return overlapping


def _assign_period_and_split(
    sessions: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    sessions = sessions.copy()
    sessions["period"] = "outside_registered_period"
    for name, bounds in config["periods"].items():
        start = pd.Timestamp(bounds["start_utc"])
        end = pd.Timestamp(bounds["end_exclusive_utc"])
        mask = (sessions["start_utc"] >= start) & (sessions["start_utc"] < end)
        sessions.loc[mask, "period"] = name
    sessions["split"] = "not_applicable"
    primary = sessions["period"] == "pre_flex_observed"
    local_start = sessions.loc[primary, "start_utc"].dt.tz_convert(
        config["local_timezone"]
    )
    sessions.loc[primary, "local_date"] = local_start.dt.date.astype(str)
    sessions.loc[primary, "local_month"] = local_start.dt.strftime("%Y-%m")
    for (_, _), group in sessions.loc[primary].groupby(
        ["id", "local_month"], sort=True
    ):
        dates = sorted(group["local_date"].unique())
        date_index = {date: index for index, date in enumerate(dates)}
        for index, row in group.iterrows():
            role = (
                "development"
                if date_index[row["local_date"]]
                % config["development_date_modulus"]
                == config["development_date_remainder"]
                else "evaluation"
            )
            sessions.loc[index, "split"] = role
    sessions.loc[sessions["period"] == "greencom_trial", "split"] = "positive_control"
    return sessions


def _integrated_tariff_cost(
    start: pd.Timestamp,
    energy_kwh: float,
    charger_power_kw: float,
    tariff: dict[pd.Timestamp, float],
    slot_minutes: int,
) -> tuple[float, pd.Timestamp] | None:
    duration = pd.Timedelta(hours=energy_kwh / charger_power_kw)
    finish = start + duration
    cursor = start
    cost = 0.0
    delivered = 0.0
    while cursor < finish:
        interval_start = cursor.floor(f"{slot_minutes}min")
        price = tariff.get(interval_start)
        if price is None or not math.isfinite(price):
            return None
        interval_end = interval_start + pd.Timedelta(minutes=slot_minutes)
        segment_end = min(interval_end, finish)
        hours = (segment_end - cursor).total_seconds() / 3600
        segment_energy = charger_power_kw * hours
        cost += price * segment_energy
        delivered += segment_energy
        cursor = segment_end
    if not math.isclose(delivered, energy_kwh, rel_tol=0, abs_tol=1e-8):
        raise BuildError("tariff integration failed energy conservation")
    return cost, finish


def _candidate_times(
    release: pd.Timestamp,
    latest_start: pd.Timestamp,
    slot_minutes: int,
) -> list[pd.Timestamp]:
    if release > latest_start:
        return []
    candidates = [release]
    boundary = release.ceil(f"{slot_minutes}min")
    while boundary <= latest_start:
        if boundary != release:
            candidates.append(boundary)
        boundary += pd.Timedelta(minutes=slot_minutes)
    return candidates


def _latest_start_exact(
    availability_end: pd.Timestamp,
    energy_kwh: float,
    charger_power_kw: float,
) -> pd.Timestamp:
    """Return the physical latest start without snapping completion to a slot."""

    return availability_end - pd.Timedelta(
        hours=energy_kwh / charger_power_kw
    )


def _release_context(
    release: pd.Timestamp,
    stream: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    interval = release.floor(f"{config['slot_minutes']}min") - pd.Timedelta(
        minutes=config["slot_minutes"]
    )
    if interval in stream.index:
        row = stream.loc[interval]
    else:
        row = pd.Series(dtype="float64")
    definitions = {
        "household_import": ("as230_import_kWh", lambda value: value >= 0),
        "solar_generation": (
            "solar_kWh",
            lambda value: 0 <= value <= config["solar_capacity_kw"] * 0.5,
        ),
        "home_battery_soc": ("battery_soc", lambda value: 0 <= value <= 100),
        "heat_pump": ("heatpump_kWh", lambda value: value >= 0),
    }
    output_names = {
        "household_import": "household_import_kwh_30min",
        "solar_generation": "solar_generation_kwh_30min",
        "home_battery_soc": "home_battery_soc_percent",
        "heat_pump": "heat_pump_kwh_30min",
    }
    context: dict[str, Any] = {
        "snapshot_is_frozen": True,
        "source_interval_start_utc": _utc_iso(interval),
        "missing": {},
    }
    audit: dict[str, Any] = {"context_interval_utc": _utc_iso(interval)}
    for short_name, (source_name, validator) in definitions.items():
        raw = _finite_or_none(row.get(source_name))
        invalid = raw is not None and not validator(raw)
        value = None if raw is None or invalid else raw
        context[output_names[short_name]] = value
        context["missing"][short_name] = value is None
        audit[f"{short_name}_raw"] = raw
        audit[f"{short_name}_invalid"] = invalid
        audit[f"{short_name}_missing"] = raw is None
    return context, audit


def _build_task(
    row: pd.Series,
    dataset_id: str,
    stream: pd.DataFrame,
    tariff: dict[pd.Timestamp, float],
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
    release = row["start_utc"]
    end = row["end_utc"]
    energy = float(row["energy_kwh"])
    charger_power = float(config["charger_power_kw"])
    continuous_hours = energy / charger_power
    duration_slots = math.ceil(
        continuous_hours / (config["slot_minutes"] / 60)
    )
    if config["feasibility_duration_rule"] != EXACT_DURATION_RULE:
        raise BuildError("unsupported feasibility duration rule")
    latest = _latest_start_exact(end, energy, charger_power)
    candidate_times = _candidate_times(release, latest, config["slot_minutes"])
    if not candidate_times:
        return None, "no_feasible_candidate", {}
    available_candidates = []
    for candidate in candidate_times:
        integrated = _integrated_tariff_cost(
            candidate,
            energy,
            charger_power,
            tariff,
            config["slot_minutes"],
        )
        if integrated is None:
            continue
        cost, finish = integrated
        available_candidates.append(
            {
                "start_time_utc": _utc_iso(candidate),
                "finish_time_utc": _utc_iso(finish),
                "projected_full_session_tariff_cost": round(cost, 6),
            }
        )
    if not available_candidates:
        return None, "all_feasible_candidates_missing_tariff", {}
    costs = [option["projected_full_session_tariff_cost"] for option in available_candidates]
    unique_costs = sorted(set(costs))
    minimum = min(costs)
    maximum = max(costs)
    span = maximum - minimum
    for option in available_candidates:
        cost = option["projected_full_session_tariff_cost"]
        option["cost_rank"] = unique_costs.index(cost) + 1
        option["cost_delta_from_min"] = round(cost - minimum, 6)
        option["normalized_cost_0_1"] = (
            0.0 if span == 0 else round((cost - minimum) / span, 6)
        )
    context, context_audit = _release_context(release, stream, config)
    identity = {
        "dataset_id": dataset_id,
        "household_id": row["id"],
        "source_row_number": int(row["source_row_number"]),
    }
    task_id = f"dingle-task-{sha256_bytes(canonical_json_bytes(identity))[:24]}"
    task = {
        "schema_version": "dingle_frozen_task_v2",
        "dataset_id": dataset_id,
        "task_id": task_id,
        "household_id": row["id"],
        "source_file": row["source_file"],
        "source_row_number": int(row["source_row_number"]),
        "period": row["period"],
        "split": row["split"],
        "release_time_utc": _utc_iso(release),
        "availability_end_utc": _utc_iso(end),
        "observed_start_utc": _utc_iso(release),
        "energy_required_kwh": round(energy, 6),
        "charger_power_kw": charger_power,
        "continuous_duration_hours": round(continuous_hours, 6),
        "duration_slots": duration_slots,
        "feasibility_duration_rule": config["feasibility_duration_rule"],
        "latest_start_utc": _utc_iso(latest),
        "release_context": context,
        "candidate_options": available_candidates,
    }
    return task, None, context_audit


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row).decode("utf-8"))
            handle.write("\n")


def build_dataset(
    config_path: Path,
    repo_root: Path,
    processed_root: Path,
) -> Path:
    config = load_config(config_path)
    inventory = source_inventory(config, repo_root)
    config_hash = sha256_bytes(canonical_json_bytes(config))
    implementation_files = [Path(__file__), Path(__file__).with_name("manifest.py")]
    implementation = {
        path.name: file_record(path) for path in implementation_files
    }
    implementation_hash = sha256_bytes(canonical_json_bytes(implementation))
    dataset_identity = {
        "algorithm_version": config["algorithm_version"],
        "config_sha256": config_hash,
        "implementation_sha256": implementation_hash,
        "source_collection_sha256": inventory["collection_sha256"],
    }
    dataset_id = f"dingle-replay-{sha256_bytes(canonical_json_bytes(dataset_identity))[:20]}"
    final_dir = processed_root / dataset_id
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen dataset: {final_dir}")
    processed_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".dingle-build-", dir=processed_root))
    try:
        streams = _read_energy_sources(config, repo_root)
        tariff_frame, tariff_mapping, tariff_audit = _canonical_tariff(streams)
        sessions = _assign_period_and_split(
            _read_charging_sessions(config, repo_root), config
        )
        sessions["overlap"] = _mark_overlaps(sessions)
        tasks: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        context_rows: list[dict[str, Any]] = []
        reason_counts: Counter[str] = Counter()

        for _, row in sessions.sort_values(
            ["id", "start_utc", "source_row_number"], na_position="last"
        ).iterrows():
            reasons: list[str] = []
            if pd.isna(row["start_utc"]) or pd.isna(row["end_utc"]):
                reasons.append("invalid_or_missing_time")
            elif row["end_utc"] <= row["start_utc"]:
                reasons.append("end_not_after_start")
            energy = _finite_or_none(row["energy_kwh"])
            if energy is None or energy <= 0:
                reasons.append("energy_nonpositive_or_missing")
            elif energy > config["ev_capacity_kwh"]:
                reasons.append("energy_above_ev_capacity")
            if row["overlap"]:
                reasons.append("overlapping_household_session")
            if row["period"] == "outside_registered_period":
                reasons.append("outside_registered_period")
            if not reasons:
                window_hours = (
                    row["end_utc"] - row["start_utc"]
                ).total_seconds() / 3600
                if energy / config["charger_power_kw"] > window_hours + 1e-12:
                    reasons.append("continuous_duration_exceeds_window")

            task: dict[str, Any] | None = None
            context_audit: dict[str, Any] = {}
            if not reasons:
                task, candidate_reason, context_audit = _build_task(
                    row,
                    dataset_id,
                    streams[row["id"]],
                    tariff_mapping,
                    config,
                )
                if candidate_reason:
                    reasons.append(candidate_reason)

            primary_reason = reasons[0] if reasons else "included"
            reason_counts[primary_reason] += 1
            audit = {
                "household_id": row["id"],
                "source_file": row["source_file"],
                "source_row_number": int(row["source_row_number"]),
                "start_time_raw": row["start_time"],
                "end_time_raw": row["end_time"],
                "energy_raw": row["ENERGY"],
                "period": row["period"],
                "split": row["split"],
                "included": task is not None,
                "primary_reason": primary_reason,
                "all_reasons": "|".join(reasons),
                "overlap": bool(row["overlap"]),
            }
            audit_rows.append(audit)
            if task is not None:
                tasks.append(task)
                context_rows.append(
                    {
                        "task_id": task["task_id"],
                        "household_id": task["household_id"],
                        "period": task["period"],
                        "split": task["split"],
                        "release_time_utc": task["release_time_utc"],
                        "availability_end_utc": task["availability_end_utc"],
                        "energy_required_kwh": task["energy_required_kwh"],
                        "candidate_count": len(task["candidate_options"]),
                        "window_hours": round(
                            (
                                pd.Timestamp(task["availability_end_utc"])
                                - pd.Timestamp(task["release_time_utc"])
                            ).total_seconds()
                            / 3600,
                            6,
                        ),
                        **context_audit,
                    }
                )

        if not tasks:
            raise BuildError("no eligible tasks remained after exclusions")
        task_ids = [task["task_id"] for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise BuildError("task IDs are not unique")
        if any(task["candidate_options"][0]["start_time_utc"] != task["observed_start_utc"] for task in tasks):
            raise BuildError("earliest feasible candidate differs from observed start")
        if any(
            pd.Timestamp(option["finish_time_utc"])
            > pd.Timestamp(task["availability_end_utc"])
            for task in tasks
            for option in task["candidate_options"]
        ):
            raise BuildError("a candidate finishes after availability end")

        atomic_write_json(temporary_dir / "source_inventory.json", inventory)
        tariff_frame.to_csv(
            temporary_dir / "canonical_tariff.csv",
            index=False,
            lineterminator="\n",
            date_format="%Y-%m-%dT%H:%M:%SZ",
        )
        pd.DataFrame(audit_rows).to_csv(
            temporary_dir / "task_audit.csv", index=False, lineterminator="\n"
        )
        _write_jsonl(temporary_dir / "frozen_tasks.jsonl", tasks)
        pd.DataFrame(context_rows).to_csv(
            temporary_dir / "task_context.csv", index=False, lineterminator="\n"
        )

        included_by_household = Counter(task["household_id"] for task in tasks)
        included_by_period_split = Counter(
            f"{task['period']}/{task['split']}" for task in tasks
        )
        candidate_counts = np.array(
            [len(task["candidate_options"]) for task in tasks], dtype=float
        )
        summary = {
            "dataset_id": dataset_id,
            "raw_charging_sessions": int(len(sessions)),
            "included_tasks": len(tasks),
            "excluded_sessions": int(len(sessions) - len(tasks)),
            "primary_reason_counts": dict(sorted(reason_counts.items())),
            "included_by_household": dict(sorted(included_by_household.items())),
            "included_by_period_split": dict(sorted(included_by_period_split.items())),
            "candidate_count": {
                "minimum": int(candidate_counts.min()),
                "median": float(np.median(candidate_counts)),
                "maximum": int(candidate_counts.max()),
            },
            "tariff_audit": tariff_audit,
            "deterministic_gates": {
                "task_ids_unique": True,
                "earliest_feasible_equals_observed": True,
                "all_candidates_finish_by_availability_end": True,
                "exact_duration_controls_latest_start": True,
            },
        }
        atomic_write_json(temporary_dir / "build_summary.json", summary)

        artifact_names = (
            "source_inventory.json",
            "canonical_tariff.csv",
            "task_audit.csv",
            "frozen_tasks.jsonl",
            "task_context.csv",
            "build_summary.json",
        )
        manifest = {
            "manifest_version": "dingle_dataset_manifest_v2",
            "dataset_id": dataset_id,
            "dataset_identity": dataset_identity,
            "config": config,
            "implementation": implementation,
            "runtime": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "platform": sys.platform,
            },
            "files": {
                name: file_record(temporary_dir / name) for name in artifact_names
            },
        }
        atomic_write_json(temporary_dir / "dataset_manifest.json", manifest)
        temporary_dir.rename(final_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return final_dir


def latest_dataset(processed_root: Path) -> Path:
    datasets = [
        path for path in processed_root.glob("dingle-replay-*") if path.is_dir()
    ]
    if not datasets:
        raise FileNotFoundError(f"no frozen dataset under {processed_root}")
    return max(datasets, key=lambda path: path.stat().st_mtime_ns)


def load_tasks(dataset_dir: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with (dataset_dir / "frozen_tasks.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def validate_dataset(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "dataset_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("manifest_version") not in {
        "dingle_dataset_manifest_v1",
        "dingle_dataset_manifest_v2",
    }:
        raise BuildError("unexpected dataset manifest version")
    if manifest.get("dataset_id") != dataset_dir.name:
        raise BuildError("dataset directory name does not match manifest ID")
    for name, expected in manifest["files"].items():
        path = dataset_dir / name
        if not path.is_file() or file_record(path) != expected:
            raise BuildError(f"artifact checksum mismatch: {name}")
    tasks = load_tasks(dataset_dir)
    if not tasks:
        raise BuildError("frozen task file is empty")
    task_ids = [task["task_id"] for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise BuildError("frozen task IDs are not unique")
    for task in tasks:
        if manifest["manifest_version"] == "dingle_dataset_manifest_v2":
            if task.get("schema_version") != "dingle_frozen_task_v2":
                raise BuildError(f"unexpected v2 task schema: {task['task_id']}")
            if task.get("feasibility_duration_rule") != EXACT_DURATION_RULE:
                raise BuildError(f"unexpected feasibility rule: {task['task_id']}")
        if task["candidate_options"][0]["start_time_utc"] != task["observed_start_utc"]:
            raise BuildError(f"earliest/observed mismatch: {task['task_id']}")
        if any(
            pd.Timestamp(option["finish_time_utc"])
            > pd.Timestamp(task["availability_end_utc"])
            for option in task["candidate_options"]
        ):
            raise BuildError(f"late candidate finish: {task['task_id']}")
    with (dataset_dir / "build_summary.json").open("r", encoding="utf-8") as handle:
        build_summary = json.load(handle)
    return {
        "dataset_id": dataset_dir.name,
        "task_count": len(tasks),
        "artifact_count": len(manifest["files"]),
        "earliest_feasible_equals_observed": True,
        "all_candidates_finish_by_availability_end": True,
        "build_summary": build_summary,
    }

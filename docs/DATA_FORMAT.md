# Local Input Contract

This document defines file formats, not a supplied dataset. No task example below is taken from the authors' records. Synthetic fixtures are generated only inside the test code.

## Task JSONL

Each line is one JSON object. Preserve stable task IDs and chronological, unique, nonempty candidate lists. Every start and finish must include an explicit timezone; UTC `Z` is recommended. Exact costs use at most six decimals. Monetary units must be consistent within an experiment.

| Field | Meaning / type |
|---|---|
| `task_id` | Unique string, stable across treatments |
| `household_id` | String used to exclude same-household pairs |
| `release_time_utc` | Start of the constructed replay window |
| `availability_end_utc` | Constructed replay completion deadline; not automatically physical unplug time |
| `latest_start_utc` | Latest feasible start |
| `energy_required_kwh` | Positive energy requirement |
| `charger_power_kw` | Positive constant charging power |
| `continuous_duration_hours` | Required energy divided by charging power |
| `release_context` | Snapshot object; see below |
| `candidate_options` | Chronological list of the candidate objects below |

Candidate fields:

- `start_time_utc`, `finish_time_utc`;
- `projected_full_session_tariff_cost`;
- `cost_rank`: all minimum-cost options have rank 1;
- `cost_delta_from_min`;
- `normalized_cost_0_1`.

The older paired/sequential payload also expects these snapshot entries:

`household_import_kwh_30min`, `solar_generation_kwh_30min`, `home_battery_soc_percent`, `heat_pump_kwh_30min`, and `missing`.

Additional source provenance fields can be retained locally. Nothing in this release supplies those fields from the paper data. Use the task builder when the documented source schema is available. For another dataset, construct the same task contract explicitly and check energy, feasibility and tariff accounting.

## Fixed pair CSV

Header: `pair_id,task_id_a,task_id_b`.

Include every unordered cross-household pair whose feasible candidates share at least one absolute UTC half-hour bin. A task can occur in multiple pairs. Pair count is not task count, and pairs do not require both tasks to have tied optima. `import_inputs.py` validates this definition or generates the list when `--pairs` is omitted.

## Proposal JSON for replay analysis

The top-level object maps model/policy names to complete maps from task ID to chosen UTC start. Each task must be represented exactly once per policy and use a supplied feasible candidate. The analytical script rejects incomplete maps; it does not manufacture an optimistic denominator.

If a model output failed, preserve that event and any completion rule in local records before providing a complete executable proposal map. The separate two-stage pipeline already keeps raw and executable schedules distinct.

## Local output records

The two-stage engine produces request/response ledgers per worker, model/runtime identities, complete plans before and after repair, round metrics, and aggregate summaries. These outputs contain user-supplied input content and are ignored by `.gitignore`. Do not add them to this code-only distribution.

The paired engine stores individual outcome objects and raw response caches. `summarize_swaps.py` uses first responses by default and never draws fleet plans from that response pool.

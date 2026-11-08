# Agent-Induced EV Charging Co-Starting: Code Release

This repository provides code for studying additional EV charging co-starting associated with scheduling-agent choices and for evaluating independent candidate presentation followed by exact minimum-change coordination.

**Code only.** No author dataset, task records, candidate tables, pair lists, permutations, model responses, schedules, or experimental result files are distributed. Users must supply their own local inputs. This release cannot regenerate the paper's numerical results without those inputs and records.

## Included experiments

| Component | Paper connection | Entry point |
|---|---|---|
| Task construction | Replay windows, feasible starts, full-session tariff costs | `data_builder/prepare_data.py` |
| Replay baselines and tie strata | Tables I and II | `scripts/analyze_replay.py` |
| Sequential start/wait | Original replay comparison; fresh Transformers port | `scripts/run_sequential.py` |
| Paired position swaps | Two-optimum choice-switch experiment | `experiments/paired_swap/` and `scripts/summarize_swaps.py` |
| Ordered/shuffled fresh model calls and central repair | Two-stage evaluation, current Table III | `experiments/two_stage/run.py` |
| Exact minimum-change assignment | Central coordination proposition | `experiments/two_stage/assignment.py` |

Table numbering refers to the manuscript after the duplicate presentation-results table was removed. See [protocol notes](docs/PROTOCOLS.md) for historical versus exported execution behavior.

## Structure

```text
data_builder/                 Original raw-source builder, with a portable CLI
experiments/two_stage/       Fresh 20-round Transformers evaluation and 4-GPU scripts
experiments/paired_swap/     Original presentation-control code and prompts
scripts/                    Local-input import and focused analysis commands
tests/                      Synthetic, standard-library correctness tests
docs/                       Input schema, protocol scope, validation notes
SOURCE_MAP.json             Code-source provenance; no private data hashes
MANIFEST.sha256             Checksums of the distributed files
```

## 1. Check the code without model calls or datasets

Python 3.11 or 3.12 is recommended. From the repository root:

```bash
python -m unittest discover -s tests -v
python scripts/import_inputs.py --help
python scripts/analyze_replay.py --help
```

Tests construct entirely synthetic inputs in temporary directories. They do not load the authors' data or require a GPU. A passing synthetic test is not a reproduction of the paper's model experiments.

## 2. Prepare your local inputs

The input format is documented in [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md). Prepare your own task JSONL and, optionally, a fixed pair CSV. All candidate times must carry timezone offsets; costs have at most six decimal places.

```bash
python scripts/import_inputs.py \
  --experiment two_stage \
  --tasks /path/to/local/tasks.jsonl \
  --pairs /path/to/local/pairs.csv
```

If `--pairs` is omitted, the importer constructs every unordered cross-household pair with at least one common feasible absolute half-hour interval. It verifies any supplied pair set against that definition. The importer updates expected counts to match your inputs and refuses to overwrite an existing input/experiment directory.

The paper uses 312 tasks, 218 tasks with multiple minimum-cost starts, and 103 evaluation pairs. Those are paper sample sizes, not files included in this release. Do not change a private dataset just to force it to match those counts.

For task construction from separately obtained source files:

```bash
python -m pip install -r requirements-analysis.txt
# Edit source_root in data_builder/config.example.json to your own source location.
python data_builder/prepare_data.py inventory --output /path/to/local/inventory
python data_builder/prepare_data.py build --output /path/to/local/processed
python data_builder/prepare_data.py validate --output /path/to/local/processed/<generated-dataset-id>
```

This source-specific builder retains the original four-household schema and date/split rules. It writes all constructed tasks to `frozen_tasks.jsonl`; selecting a different period/split is a separate, explicit preprocessing decision. It does not by itself promise to produce the paper's 312-task evaluation roster. Obtain the source dataset from the [official Figshare repository](https://doi.org/10.6084/m9.figshare.30294478), linked in the [data descriptor](https://doi.org/10.1038/s41597-026-07186-3). The [two-stage data README](experiments/two_stage/data/README.md) and [paired-swap data README](experiments/paired_swap/data/README.md) explain the required files and local preparation steps. Obtain source files separately under their applicable terms. No source dataset is bundled.

## 3. Compute replay statistics on local records

```bash
python scripts/analyze_replay.py \
  --tasks /path/to/local/tasks.jsonl \
  --pairs /path/to/local/pairs.csv \
  --out /path/to/local/replay_summary.json
```

This calculates the independent feasible and independent minimum-cost analytical expectations, common earliest-minimum selection, and strata with zero, one, or two tied-optimum tasks. No LLM calls are made.

To include saved model choices, add `--proposals /path/to/local/proposals.json`. The file maps each model name to a complete `{task_id: selected_start_utc}` object. Incomplete or infeasible maps are rejected; tasks are never silently dropped from the fixed pair denominator. The input is a *complete* schedule; any prior failure/completion rule must be documented with the local response records.

## 4. Install the Transformers environment

Use Linux with a BF16-capable CUDA GPU. Install PyTorch compatible with the server's driver/CUDA stack, then:

```bash
python -m pip install -r requirements-inference.txt
```

The inference code pins Transformers 5.17.0. The main experiment uses Qwen/Qwen3.5-9B and google/gemma-3-12b-it, local Safetensors checkpoints, BF16 and greedy decoding. Download model weights separately with the required model access. This release does not distribute weights or use Ollama for fresh calls.

Copy `experiments/two_stage/models.local.example.json` to `models.local.json` in the same directory and set your checkpoint paths. Keep local paths and generated identities out of Git.

## 5. Run the two-stage evaluation on four GPUs

After importing inputs and setting model paths:

```bash
cd experiments/two_stage
python run.py prepare
python run.py check
python run.py freeze-models --models models.local.json
bash preflight_4gpu.sh
bash launch_4gpu.sh
bash status_4gpu.sh
# After all workers finish:
python run.py analyze
```

The default allocation is one model replica per GPU: Qwen on GPUs 0/1 and Gemma on GPUs 2/3. Two shards per model divide the 20 rounds. Change `GPU_IDS=4,5,6,7` when needed. The scripts also accept `PYTHON_BIN`, `MODELS`, and `OUT` environment variables. Ordered and shuffled requests are freshly generated for every round; outputs are not recombined across jobs.

The protocol uses a complete independent candidate permutation per task and round, zero extra tariff cost, and at most two starts per absolute half-hour interval. The coordinator minimizes changes to each treatment's own raw proposals. Co-starting is measured before and after repair; it is not the central flow objective. A deficient assignment is reported as infeasible.

The parser accepts plain JSON or one complete outer Markdown JSON fence. Service errors and unresolved choices are recorded separately; an unresolved proposal receives its earliest exact minimum-cost start for the initial executable schedule, and final assignments are optimized from raw proposals. See the code and [protocol notes](docs/PROTOCOLS.md) for the consistent change-count partition.

The analysis writes local response/plan ledgers, round metrics, means, sample standard deviations, and paired-round differences. Those are local outputs, not part of this distribution. To resume, stop existing workers before rerunning the launcher; changed code, data, prompts, or identities require a new experiment copy.

## 6. Run and summarize paired position swaps

Import the user's local double-optimum subset into a separate experiment:

```bash
python scripts/import_inputs.py \
  --experiment paired_swap --two-optima-only \
  --tasks /path/to/local/tasks.jsonl --pairs /path/to/local/pairs.csv
cd experiments/paired_swap
cp models.local.example.json models.local.json
# Edit both local model paths, even if running only Qwen below.
python run.py prepare --scope full --models models.local.json --output /path/to/local/swap_run
CUDA_VISIBLE_DEVICES=0 python run.py run --scope full --phase test \
  --model qwen3.5-9b --models models.local.json --output /path/to/local/swap_run
cd ../..
python scripts/summarize_swaps.py --run /path/to/local/swap_run \
  --choice first --out /path/to/local/swap_summary.json
```

Only two-optimum tasks are imported. Each has five test permutation backgrounds, each paired with a swap of the two optimal options. The original generator reserves five calibration backgrounds before indexing the test backgrounds; a test-only run need not make calibration calls. Backgrounds can repeat when the finite permutation space is exhausted. With 42 local tasks this produces 210 comparisons and 420 first responses per model, but no 42-task author subset is included.

The retained presentation-control engine allows one explicit format-repair call and saves first/final choices separately. The focused extractor defaults to first responses, labels unresolved pairs, and does not assemble 200 offline fleet plans. The legacy engine's broader `analyze` command is retained for source completeness but is not the main two-stage evaluation.

## 7. Fresh sequential start/wait calls

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_sequential.py \
  --tasks /path/to/local/tasks.jsonl \
  --models experiments/two_stage/models.local.json \
  --model qwen --out /path/to/local/sequential_qwen
```

This new Transformers port uses fresh conversations at each decision time, records every request and raw answer, and advances on `start_now=false`. Invalid/all-wait tasks are explicitly logged and completed at their earliest minimum-cost start; no quality-repair calls are made. An existing output directory is refused. `--shard` and `--shards` support separate task shards; merge complete maps and check coverage before calculating a fleet rate.

**Historical distinction:** The paper's original sequential outcomes were obtained in an earlier Ollama run. They are not bundled here. This fresh Transformers port is a new execution protocol and must not be labelled as the source of those historical numerical outcomes.

## Metrics and scope

- Co-start rate: fraction of the fixed eligible task pairs starting in the same date-specific UTC half-hour interval.
- Minimum changes: number of task proposals revised by the coordinator, including required completions according to the protocol.
- Intervals above the start limit: intervals with more new starts than the quota; this is not measured electrical overload.
- Spread: sample standard deviation across complete rounds on the fixed tasks, not an independent-pair binomial confidence interval.

The local ordering is designed to disperse choices. The central flow exactly minimizes proposal changes under authorized tariff and start-count limits. Neither the quota nor the flow proof establishes a total-power, voltage, or feeder-safety guarantee.

## Availability and citation

This is a code-only companion release. Private inputs, outputs and historical model records are not provided, and no claim of numerical paper replication is made for the synthetic tests. See [validation notes](docs/VALIDATION.md) and [the input schema](docs/DATA_FORMAT.md).

Authors: Yulin Song, Chenxi Sun, and Jianwei Huang. Cite the associated manuscript using its final publication metadata when available; this package does not invent a publication DOI. A redistribution license has not been selected in this packaging step. Model and upstream dataset terms apply separately.

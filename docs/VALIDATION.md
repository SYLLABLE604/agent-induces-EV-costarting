# Code-Only Release Validation

Validated locally on November 5, 2026.

## Passed checks

`python -m unittest discover -s tests -v`: **8 tests passed**.

The tests cover:

1. Analytical independent-choice probabilities and tied-optimum pair strata on invented tasks.
2. Rejection of incomplete proposal maps rather than dropping tasks from the denominator.
3. Minimum-change flow objectives checked against brute-force enumeration over small schedules.
4. Detection of joint infeasibility despite individually nonempty authorized sets.
5. Valid JSON/fence handling and invalid/duplicate candidate-key rejection.
6. Paired swaps changing only the two optimal options, with other positions fixed.
7. Local input import followed by a complete 20-round, two-model, two-treatment **mock** pipeline: 240 invented task calls and 80 complete plans, with completion, final constraints and resume behavior checked by the analysis engine.
8. Extraction of paired-switch statistics distinguishing first responses from post-format-repair choices, using invented records.

All Python modules pass syntax parsing. The six documented command-line entry points load successfully for `--help` without GPU packages. All distributed shell scripts pass `bash -n`.

## Distribution checks

- Code/configuration/prompt files are copied through an explicit allowlist, not by copying source data directories.
- No author task JSONL, pair CSV, permutations, model-response records, schedules, experimental result files, caches, private model locks or model weights are included.
- No concrete private task IDs or private absolute machine paths are found in the release text.
- Python bytecode and test-generated artifacts are excluded from the ZIP.
- Every distributed file except the checksum list itself is covered by `MANIFEST.sha256`.

## Not verified here

No real CUDA inference or model-weight loading was performed in this packaging session. The model backend and scientific engines are retained from their source implementations, but copying their code does not validate a new server environment. Run the GPU preflight before formal calls.

The synthetic tests validate software behavior only. They do not regenerate or independently audit any published model result. Historical sequential outputs and the formal 20-round author records are not distributed.

The archive contains no `.git` directory or Git history. No GitHub upload was performed. A public software license remains for the owners to select.

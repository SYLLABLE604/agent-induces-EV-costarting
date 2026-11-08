# Protocol and Source Notes

## Preserved scientific code

- `experiments/two_stage/`: the original fresh ordered/shuffled 20-round code, including the min-cost-flow assignment, scalar metrics, response parser, process watchdog, batching, resume checks, and four-GPU launcher.
- `experiments/paired_swap/`: the original Transformers presentation-control implementation. It includes a broader historical analysis engine, but the recommended public entry point extracts individual test swaps only; it does not use offline fleet recombinations as the main evaluation.
- `data_builder/dingle_replay/`: original immutable task construction and source-manifest functions. The portable wrapper replaces repository-specific entry-point paths.

Only explicitly allowlisted source, configuration, prompt and shell files were copied. No data, prepared ledgers, model locks, real response caches, run outputs, source datasets, or old archives were copied. `SOURCE_MAP.json` records source-code provenance, not private data fingerprints.

## Adaptations in this release

1. Experiment version strings have a code-release suffix. Local model-path examples use placeholders. No frozen input manifests are distributed; users generate their own after importing local inputs.
2. `scripts/import_inputs.py` generates/validates local pair sets and updates expected counts before preparation. It refuses to overwrite existing records. Changing inputs after preparation requires a new experiment copy.
3. `scripts/analyze_replay.py` provides focused analytical and tie-stratum calculations without requiring historical run-directory names or LLM dependencies.
4. `scripts/summarize_swaps.py` extracts test-phase two-optimum switch counts and separately reports unresolved pairs. It defaults to first responses; selecting `--choice final` explicitly changes the estimand to post-format-repair choices.
5. `scripts/run_sequential.py` is a **new** Transformers start/wait port, using the latest inference backend with the earlier sequential input schema. It does not reproduce historical Ollama/GGUF runtime identities or outputs. Its explicit failure/all-wait completion rule is part of the new protocol.

## Three distinct experiment units

The original sequential replay, the paired position-swap study, and the fresh 20-round fleet evaluation are separate protocols. A task, a query, a response pair, an eligible task pair, and a complete fleet round are different units. Five swap backgrounds do not mean five complete fleet rounds. Twenty full-list shuffled rounds are not 200 offline assemblies.

## Pair-swap generation

The source generator samples two display positions for the two target optima, fills other positions with randomly ordered remaining options, then constructs the second input by swapping only the two target entries. The plus orientation fixes earliest-before-latest; the minus orientation reverses it. The resulting pair supplies both orientations. Thus a plus background alone should not be described as an unconstrained uniform permutation of every option. Small candidate spaces can reuse paired backgrounds. The code retains five calibration and five test background indices; the paper's 210 comparisons use five test pairs for each of 42 tasks.

## Guarantees and randomness

The central solver assigns every task to an authorized start if a full assignment exists, and minimizes proposal changes under the stated interval start-count quota. Co-starting is an outcome, not a secondary flow objective. Nonempty individual sets do not imply joint feasibility.

The fresh evaluation independently seeds the complete candidate permutation for each task and round. Ordered and shuffled conditions use fresh model calls with identical candidate attributes. Greedy generation can still vary with hardware/batching; runtime and generation identities are retained. Quotas constrain new starts, not the number of ongoing charging sessions or network loading.

## License status

No redistribution license was chosen during this packaging operation. The source owners should select a license before a public release. Model weights and upstream datasets have their own terms and are not included. This package does not grant rights to either.

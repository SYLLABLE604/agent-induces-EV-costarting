# Obtaining the Source Dataset

No source data or author-generated experimental records are included in this directory.

## Official source

Download the upstream dataset from the repository identified in the source paper's **Data availability** section:

- **Dataset:** [Residential Energy Dataset with Electric Vehicles, Photovoltaic Generation and Tariff Variability in Ireland — Figshare](https://doi.org/10.6084/m9.figshare.30294478).
- **Data descriptor:** D. Avgoloupis et al., [Residential Energy Dataset with Electric Vehicles, Photovoltaic Generation and Tariff Variability in Ireland](https://doi.org/10.1038/s41597-026-07186-3), *Scientific Data*, vol. 13, article 834, 2026.

Use the files and download controls on the Figshare landing page. Record the dataset version and retain the upstream citation and license information when obtaining the files. This package links to the upstream publication; it does not redistribute the dataset.

## Files expected by this code

After extracting the upstream download, locate the folder containing `EnergyStreams/` and `JourneyCharge/`. The task builder expects the following relative paths for the four source household identifiers:

```text
Dataset/
├── EnergyStreams/
│   ├── id01_merged_30min.csv
│   ├── id02_merged_30min.csv
│   ├── id03_merged_30min.csv
│   └── id04_merged_30min.csv
└── JourneyCharge/
    ├── id01_Sorted_Mobility.csv
    ├── id02_Sorted_Mobility.csv
    ├── id03_Sorted_Mobility.csv
    └── id04_Sorted_Mobility.csv
```

`EnergyStreams` supplies household energy and tariff time series; `JourneyCharge` supplies mobility and charging-event records. The builder selects charging events and computes candidate-session costs from the energy-stream tariff field. Use the released processed `EnergyStreams` files expected by this builder; the upstream `RawData` files are a separate preprocessing choice.

## Build local experiment inputs

Run these commands from this code package's root directory:

1. Put the downloaded source files in a local directory, for example `local_data/Dataset/`, with the layout above. Set `source_root` in `data_builder/config.example.json` to that directory, relative to the package root.
2. Install the analysis dependencies and construct tasks:

   ```bash
   python -m pip install -r requirements-analysis.txt
   python data_builder/prepare_data.py inventory --output /path/to/local/inventory
   python data_builder/prepare_data.py build --output /path/to/local/processed
   python data_builder/prepare_data.py validate --output /path/to/local/processed/<generated-dataset-id>
   ```

3. The builder writes `frozen_tasks.jsonl` inside the generated dataset directory. Make any intended period/split selection explicitly before importing the local task roster. The complete upstream dataset is not already the paper's 312-task roster, and the 103 eligible pairs are constructed from task candidates rather than downloaded as a source file.

4. Install your selected local task file into this experiment:

   ```bash
   python scripts/import_inputs.py --experiment two_stage \
     --tasks /path/to/local/selected_tasks.jsonl
   ```

   The importer generates all eligible cross-household task pairs. If you already have a fixed pair CSV, add `--pairs /path/to/local/pairs.csv` to validate and use it.

See the [main README](../../../README.md) for the model configuration, GPU preflight and 20-round evaluation commands, and the [input schema](../../../docs/DATA_FORMAT.md) for field definitions.

Generated inputs, pair lists and outputs remain local and are ignored by Git. No data were downloaded or added when this documentation was prepared.

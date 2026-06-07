# Reproducibility Assets

This directory centralizes submission-grade reproducibility metadata:

- `qc_threshold_manifest.csv`: script-level preprocessing and QC thresholds with source anchors.
- `repro_manifest.csv` / `repro_manifest.json`: package pins, lock hashes, container recipe hashes, and git freeze metadata.
- `generate_statistical_assets.py`: consolidates manuscript-aligned statistics assets for annotation, integration, mechanism and DKD targeted summaries.
- `statistics/`: machine-readable statistics tables cited by the manuscript and Supplementary.
- `locks/`: exact environment snapshots (`conda --explicit`, `conda list`, `pip freeze`, `R session`, runtime string).
- `container/`: Docker and Apptainer recipes that reconstruct the `smn` environment from lock files.

The intended repository freeze tag for submission is:

- `submission-v1-freeze`

# Reproducibility Assets

| Goal | Start here |
| --- | --- |
| Try Cytoneighbor without study data | Install from source and run `python tests/test_quick_run.py` or open the [tutorials](../tutorials/). |
| Recompute key revision results | Run `python repro/revision/replay_tables.py` with NumPy installed. See the [revision guide](revision/) for inputs and endpoints. |
| Regenerate tables from the original experiment outputs | Download the [original results](../results/) and use the statistics command below. |
| Re-run an original experiment | Download the [study inputs](../data/), install its dependencies and use the commands below. |

## Original experiments

Install the core package and the benchmark dependencies from the repository root.

```bash
python -m pip install -e ".[benchmarks]"
```

| Workflow | Command from the repository root | Input |
| --- | --- | --- |
| Marker-retained NK/NKT resolution | `python main.py run-experiment benchmark-resolution` | `data/Human immune.h5ad` |
| Original scarcity study | `python main.py run-experiment mechanism-analysis` | `data/Pancreas.h5ad` |
| Original DKD workflow | `python main.py run-experiment disease-discovery` | `data/Diabetic Kidney Disease.h5ad` |
| Batch integration | `python experiments/Rebatch/a-main_remove.py` | `data/Human immune.h5ad` |
| Cytoneighbor leave-one-batch-out | `python experiments/Annotation/Cytoneighbor.py` | Defaults to `data/Mouse_Kidney_mod1.h5ad`. Set `CYTONEIGHBOR_DATA` to select a different input. |

The resolution driver uses a Seurat-style PCA proxy. The actual R Seurat
annotation driver is [seurat.py](../experiments/Annotation/seurat.py).
The R-based annotation drivers require `Rscript` on PATH and the
packages declared in their [R scripts](../experiments/Annotation/NetWork/).
The CellTypist, scArches and classical ML drivers also require their
respective method packages. Review the dataset and label settings at the
top of each baseline driver. The default zebrafish input for several
baselines has a pending download link. [Dataset guide](../data/)

[configs/config.yaml](../configs/config.yaml) records experiment defaults
and [qc_threshold_manifest.csv](qc_threshold_manifest.csv) records
preprocessing and thresholds. Record the input hashes and software versions
when comparing a new run to the archived outputs.

## Regenerate original statistical summaries

Download the archived `CV_Perfomace/`, `CV_Rebatch/` and `disease_discovery/`
folders into `results/`. Keep their filenames and subdirectories. The generator
reads the saved fold and integration CSVs and the DKD `experiment_results.json`.
The public mechanism folder contains figures and reports. The command below
uses the [bundled recall inputs](original_inputs/mechanism_recall_scores.json)
extracted from the available original mechanism export.

```bash
python repro/generate_statistical_assets.py --results-dir results --output-dir results/statistics_recomputed --mechanism-results repro/original_inputs/mechanism_recall_scores.json
```

The [bundled statistical tables](statistics/) are from a historical
snapshot. The generated values follow the particular archive copy or run you
supply. For example, duplicate result folders or a newer DKD or mechanism
export can yield different historical summaries. Use the
[revision replay](revision/) for the frozen revision endpoints.

The [locks](locks/) and [container recipes](container/) document the
historical `smn` environment. These include platform-specific packages
and machine paths. Use the source installation for a portable starting point.

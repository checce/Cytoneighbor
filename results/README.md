# Experiment Results

The [Cytoneighbor results archive on Google Drive](https://drive.google.com/drive/folders/1S41DI2W0Iy3YsorIWSHqWgZaWn2zZG7A?usp=sharing)
contains the original experiment outputs. Download the directories needed
for your workflow and place them under `results/`, keeping the directory
names below. For the revision, start with the
[frozen aggregate tables and checker](../repro/revision/).

| Original directory | Contents |
| --- | --- |
| `CV_Perfomace/` | Annotation fold outputs and checkpoints. The directory name follows the original scripts. |
| `CV_Rebatch/` | Batch integration embeddings and checkpoints |
| `benchmark_resolution/` | Integration comparison and resolution outputs |
| `figure2_annotation_assets/` | Annotation example and its local neighbourhood |
| `mechanism_analysis/` | Original pancreatic delta-cell scarcity analysis |
| `disease_discovery/` | Original DKD analysis and `figure5_rebuild_data/` |

The archive may show multiple folders with the same name. Use the copy
containing the inputs for the protocol you intend to replay. The
[reproducibility guide](../repro/) distinguishes the original workflows
from the revision analyses and identifies which checks run on GitHub
assets alone. Archived figures, filenames and JSON keys reflect the original
experiment versions. Use the revision tables for the donor-level DKD inference
and the additional training augmentation comparisons.

For regeneration of the original summary tables, the
[DKD results folder with experiment_results.json](https://drive.google.com/drive/folders/1griQ2A8UXW4NLI2Gz_Vv-YQgmAb-DFZL)
is one available copy. Use the
[bundled mechanism recall inputs](../repro/original_inputs/mechanism_recall_scores.json)
with the statistical generator. See the [reproduction guide](../repro/).

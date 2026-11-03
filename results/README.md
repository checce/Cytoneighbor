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
| `mechanism_analysis/` | Original scarcity and Gene-CutMix outputs |
| `disease_discovery/` | DKD analyses and `figure5_rebuild_data/` |

The archive may show multiple folders with the same name. Use the copy
containing the inputs for the protocol you intend to replay. The
[reproducibility guide](../repro/) distinguishes the original workflows
from the revision analyses and identifies which checks run on GitHub
assets alone.

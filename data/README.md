# Cytoneighbor Datasets

To follow the tutorials, start with the Human Immune dataset or use their
synthetic examples. The study inputs are AnnData files and can be downloaded
from the [Cytoneighbor dataset folder on Google Drive](https://drive.google.com/drive/folders/11PuJRaw_JC7xJhphQklac7eg9AuGHsCE?usp=sharing).
Place downloaded files directly in `data/` using the exact names below.

## Dataset inventory

These dimensions describe the input matrices before experiment-specific
preprocessing.

| File | Cells | Genes | Use |
| --- | ---: | ---: | --- |
| `Human immune.h5ad` | 33,506 | 12,303 | Annotation, integration and the tutorials |
| `Pancreas.h5ad` | 16,382 | 18,771 | Pancreatic delta-cell scarcity experiments |
| `Diabetic Kidney Disease.h5ad` | 39,176 | 27,980 | Kidney state matching and the donor analysis |
| `Lung_atlas_public.h5ad` | 32,472 | 15,148 | Leave-one-batch-out annotation |
| `Mouse_Kidney_mod1.h5ad` | 11,296 | 29,013 | Leave-one-batch-out annotation |
| `8d2e51ab-0edd-46c8-bdac-558de319d6c2_sample_id_7_donor_id_6.h5ad` | 12,808 | 61,455 | Human tendon snRNA-seq annotation |
| `Zebrafish embryonic cells.h5ad` | 26,022 | 25,258 | Annotation and reference-absent-type analysis. The study input is currently unlinked. |

The separate PBMC scalability experiment uses the RNA UMI matrix from the
[Seurat v4 PBMC CITE-seq resource](https://docs.scvi-tools.org/en/stable/api/reference/scvi.data.pbmc_seurat_v4_cite_seq.html).
The [source download](https://ndownloader.figshare.com/files/27458840) is a
161,764-cell atlas. Save it as `data/pbmc_seurat_v4.h5ad` for the revision
protocol. The recorded SHA-256 is
`c3b0100a6ce27beb64eff53692e09f98da2a58cfdfea08d15ff204f834b41396`.
The measured pools use up to 100,000 cells from this atlas.

## Download and verify

```bash
python scripts/download_data.py --list
python -m pip install gdown
python scripts/download_data.py --file "Human immune.h5ad"
python scripts/download_data.py --verify --file "Human immune.h5ad"
```

Run the helper without `--file` to download the currently linked study
inputs. If Drive rate-limits the helper, download the files through the
folder in your browser and run `--verify`. The
[input checksums](input-checksums.csv) record the exact study copies.
Follow the [reproduction guide](../repro/) for the next step.

## Data attribution

Please cite the source studies and follow the original terms of use.
See the [preprocessing manifest](../repro/qc_threshold_manifest.csv)
for accession identifiers and the experiment-specific filters.

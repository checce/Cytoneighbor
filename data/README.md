# Cytoneighbor Datasets

This directory should contain the single-cell RNA-seq datasets (`.h5ad`
files) required to run the tutorials and reproduce the paper experiments.

## Download

All datasets are available at:

**[Google Drive: Cytoneighbor Datasets](https://drive.google.com/drive/folders/11PuJRaw_JC7xJhphQklac7eg9AuGHsCE?usp=sharing)**

Download the `.h5ad` files and place them directly in this directory
(no subdirectories).

## Dataset Inventory

| File | Cells | Description | Used In |
|---|---|---|---|
| `Human immune.h5ad` | ~335,000 | Multi-batch, multi-donor PBMC atlas | Tutorials, Fig 2, Fig 3 |
| `Pancreas.h5ad` | ~14,000 | Human pancreatic islet cells | Fig 4 (rare-cell) |
| `Diabetic Kidney Disease.h5ad` | ~23,000 | PT cells, healthy and T2DM donors | Fig 5 (disease) |
| `Lung_atlas_public.h5ad` | ~32,000 | Multi-donor lung cell atlas | LOBO benchmarks |
| `Mouse_Kidney_mod1.h5ad` | ~12,000 | Mouse kidney scRNA-seq | LOBO benchmarks |
| `Zebrafish embryonic cells.h5ad` | ~32,000 | Zebrafish embryogenesis | LOBO benchmarks |

## Usage

After downloading, verify the files:

```bash
ls -lh data/*.h5ad
```

Run the sanity check:

```bash
python tests/test_quick_run.py
```

Then open the tutorials:

```bash
jupyter notebook tutorials/tutorial_annotation.ipynb
```

## Citation

Please cite the original data generators:

- **Human Immune**: Freytag et al., Oetjen et al., Villani et al., Sun et al.
- **Pancreas**: Baron et al. (2016), Muraro et al. (2016), Segerstolpe et al. (2016), Xin et al. (2016)
- **Diabetic Kidney Disease**: Wilson et al. (2022)
- **Lung Atlas**: Travaglini et al. (2020)
- **Mouse Kidney**: Park et al. (2018)
- **Zebrafish embryogenesis**: Wagner et al. (2018), Farrell et al. (2018)

Full accession identifiers are documented in `repro/qc_threshold_manifest.csv`.

## Automated Download

```bash
pip install gdown
python scripts/download_data.py
```

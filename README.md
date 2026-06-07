<p align="center">
  <h1 align="center">&#128300; Cytoneighbor</h1>
  <p align="center">
    <strong>Neighbour-evidence-guided single-cell state interpretation</strong>
  </p>
  <p align="center">
    <a href="https://pypi.org/project/cytoneighbor">
      <img alt="PyPI" src="https://img.shields.io/badge/pypi-v0.1.0-blue">
    </a>
    <a href="https://www.python.org/downloads/">
      <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
    </a>
    <a href="https://github.com/checce/Cytoneighbor/actions/workflows/tests.yml">
      <img alt="CI" src="https://github.com/checce/Cytoneighbor/actions/workflows/tests.yml/badge.svg">
    </a>
    <a href="https://opensource.org/licenses/MIT">
      <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
    </a>
  </p>
</p>

---

## &#128269; What is Cytoneighbor?

**Cytoneighbor** is a neighbour-evidence framework for interpretable single-cell
state analysis. Rather than returning a single label or UMAP coordinate and
hoping it is correct, Cytoneighbor constructs a **ranked neighbour list** for
every query cell and provides **diagnostic readouts** that reveal *whether* the
evidence supports the biological interpretation — and *why*.

The framework combines three core components:

- **Scalable candidate retrieval** with FAISS HNSW for fast nearest-neighbour search across millions of cells
- **Exact pairwise re-scoring** via a **Symmetric Relation Encoder (SymRelEnc)** that learns discriminative cell-pair relationships
- **Gene-CutMix**, a training-time data augmentation strategy that strengthens rare-population signals in sparse expression regions

Cytoneighbor fits naturally into a standard Scanpy workflow: train a model,
build a neighbour graph, and then use it as a drop-in replacement for any
downstream function that accepts a `neighbors_key`.

> &#128214; **Paper**: Hu H. et al. *Inspectable neighbor evidence for single-cell state interpretation with Cytoneighbor* (in preparation).

---

## &#127775; Why Cytoneighbor?

| Your goal | Typical approach | What Cytoneighbor does differently |
|---|---|---|
| Annotate cells | Closed-set forced label | **Unknown-gating**: cells with weak evidence remain *Unknown* instead of being misclassified |
| Integrate batches | Global alignment blurs boundaries | **Boundary-preserving**: related types (e.g. NK and NKT) stay separable after batch correction |
| Find rare cells | Oversampling or downweighting | **Gene-CutMix**: restores neighbour evidence for rare populations without inflating the reference |
| Compare disease vs healthy | Pooled-control differential expression | **State-matched**: separates compositional redistribution from within-state expression changes |

---

## &#128230; Installation

### Quick install (recommended)

```bash
pip install cytoneighbor
```

### From source (with GPU acceleration)

```bash
git clone https://github.com/checce/Cytoneighbor.git
cd Cytoneighbor

# CPU-only (no GPU required)
pip install -e .

# With GPU FAISS support
pip install -e ".[gpu]"

# With integration baseline tools (scVI)
pip install -e ".[gpu,integration]"

# Development install (includes testing tools)
pip install -e ".[dev]"
```

### Conda environment (exact reproducibility)

```bash
conda env create -f repro/locks/environment-smn.yml
conda activate smn
pip install -e .
```

### Verify your installation

```bash
python tests/test_quick_run.py
```

Expected output:

```text
[1/5] Generating mock single-cell data (100 cells x 200 genes)...
[2/5] Loading data into Cytoneighbor (load_data)...
[3/5] Training model (fit, 2 epochs)...
[4/5] Building neighbour graph (transform)...
[5/5] Running annotation (annotate with Unknown gating)...
============================================================
  All 5/5 checks passed in X.X seconds.
  Cytoneighbor is successfully installed and functioning.
============================================================
```

No data download is required — the sanity check generates mock data in memory.

---

## &#9889; Quick Start

```python
import scanpy as sc
import cytoneighbor as cn

# 1. Load your data (any AnnData-compatible format)
ref  = sc.read_h5ad("reference.h5ad")    # well-annotated reference
qry  = sc.read_h5ad("query.h5ad")        # your query to annotate

# 2. One call: train on reference, annotate query with unknown-gating
result = cn.quick_annotate(ref, "cell_type", qry)

# 3. Inspect
print(result.obs["cytoneighbor_prediction"].value_counts())
```

> **Try it now** — no installation required:
> ---

## &#128218; API at a Glance

```python
import cytoneighbor as cn

# --- One-liner for the most common use case ---
result = cn.quick_annotate(ref, "cell_type", query)

# --- Step-by-step for full control ---
cn.load_data(adata, data=X, labels=labels)    # prepare expression matrix
cn.fit(adata, label_key="cell_type")           # train SymRelEnc model
cn.transform(adata, n_neighbors=15)            # build neighbour graph
cn.annotate(ref, "cell_type", query, enable_unknown=True)  # transfer labels

# --- Save and load trained models ---
cn.save_model(adata, "my_model.zip")
cn.load_model(adata, "my_model.zip")
```

| Function | What it does |
|---|---|
| `quick_annotate(ref, label_key, query)` | Train + annotate in one call |
| `fit(adata, label_key, ...)` | Train the SymRelEnc model on reference data |
| `transform(adata, n_neighbors, ...)` | Build a pairwise-score neighbour graph |
| `annotate(ref, label_key, query, ...)` | Transfer labels using ranked neighbour evidence |
| `load_data(adata, data, labels)` | Prepare expression matrix and labels for training |
| `save_model(adata, path)` | Serialize trained model to disk |
| `load_model(adata, path)` | Deserialize trained model from disk |

---

## &#128736;&#65039; Core Workflows

### &#129514; Cell-type Annotation with Unknown-gating

The primary use case: you have a well-annotated reference and an unlabelled
query — and the query may contain cell types *not present* in your reference.

```python
import cytoneighbor as cn
import scanpy as sc

# 1. Load your data
ref  = sc.read_h5ad("your_reference.h5ad")
qry  = sc.read_h5ad("your_query.h5ad")

# 2. Train and annotate with unknown-gating enabled
cn.fit(ref, label_key="your_cell_type_column")
cn.annotate(ref, "your_cell_type_column", qry, enable_unknown=True)

# 3. Use predictions downstream
qry.write("annotated.h5ad")
qry.obs[["cytoneighbor_prediction", "cytoneighbor_confidence"]].to_csv("predictions.csv")

# 4. Run differential expression per predicted cell type
sc.tl.rank_genes_groups(qry, groupby="cytoneighbor_prediction")
```

When `enable_unknown=True`, Cytoneighbor writes diagnostic columns to
`query.obs`:

| Column | Meaning |
|---|---|
| `cytoneighbor_prediction` | Predicted label (or *Unknown*) |
| `cytoneighbor_confidence` | Top-1 label probability (0–1) |
| `cytoneighbor_margin` | Gap between top-1 and top-2 probabilities |
| `cytoneighbor_norm_entropy` | Normalised prediction entropy (0–1) |
| `cytoneighbor_is_unknown` | Boolean: was this cell gated as Unknown? |

**How to interpret**: low confidence + low margin + high entropy = weak
neighbour evidence. Cytoneighbor flags these cells as *Unknown* instead of
forcing an incorrect label — you can then inspect them separately or assign
them to novel cell types.

### &#128225; Boundary-preserving Batch Integration

Integrate multi-batch, multi-donor single-cell data while preserving
biologically meaningful boundaries between related cell types.

```python
import cytoneighbor as cn
import scanpy as sc

# 1. Load multi-batch data
adata = sc.read_h5ad("your_multi_batch_data.h5ad")

# 2. Standard preprocessing
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000, batch_key="your_batch_column")

# 3. Build Cytoneighbor neighbour graph
cn.fit(adata, label_key="your_cell_type_column")
cn.transform(adata, n_neighbors=15)

# 4. Use the graph for ANY downstream Scanpy function
sc.tl.umap(adata, neighbors_key="neighbors")          # UMAP from Cytoneighbor graph
sc.tl.leiden(adata, neighbors_key="neighbors")         # clustering
sc.tl.rank_genes_groups(adata, groupby="leiden")       # marker detection

# 5. Evaluate integration quality
# Compare batch mixing (iLISI) and cell-type purity (cLISI)
# against uncorrected baseline — see tutorials for full example
```

The key is `neighbors_key="neighbors"` — it plugs Cytoneighbor into your
existing Scanpy workflow without changing anything else.

### &#128300; Rare-cell Recovery with Gene-CutMix

Rare cell types (e.g. delta cells in pancreas, NKT cells in PBMC) often lose
neighbour evidence due to severe abundance imbalance. Gene-CutMix generates
synthetic training pairs that preserve the discriminative signal without
inflating the reference.

```python
# Gene-CutMix is enabled by default during fit().
# Tune its behaviour via optional parameters:
cn.fit(
    adata,
    label_key="cell_type",
    synth_enable=True,           # enable Gene-CutMix (default: True)
    synth_mult_per_dt=2.0,       # synthesis multiplier per rare type
    synth_cap_per_dt=500,        # upper cap on synthetic cells per type
)
```

### &#128137; Disease-state Matching

For disease-vs-healthy comparisons, Cytoneighbor constructs state-matched
neighbourhoods that separate compositional redistribution from genuine
within-state expression changes.

```python
# Train on healthy reference, annotate disease query
cn.fit(healthy_ref, label_key="cell_type")
cn.annotate(healthy_ref, "cell_type", disease_query, enable_unknown=False)

# Use the neighbour graph for state-matched differential expression
cn.transform(combined_data, n_neighbors=30)
sc.tl.rank_genes_groups(combined_data, groupby="disease_state")
```

---

## &#128214; Tutorials

Step-by-step guides with every cell output preserved so you can preview before
running.

### 1. Cell-type Annotation

Train → annotate → visualize → export. Covers unknown-gating, confidence
diagnostics, and adapting the workflow to your own data.

### 2. Batch Integration

Build a neighbour graph → visualize integration quality → run downstream
clustering and marker detection → evaluate against raw baseline.

> &#128161; **Mock data fallback**: both tutorials automatically generate
> synthetic data if the real dataset is not found, so you can follow along
> immediately. See `tutorials/README.md` for prerequisites and dataset info.

---

## &#128451; Project Structure

```text
Cytoneighbor/
|
+-- cytoneighbor/                   #  Core package  (pip install cytoneighbor)
|   +-- __init__.py                 #     Public API + quick_annotate() entry point
|   +-- model.py                    #     SymRelEnc, FAISS HNSW, Gene-CutMix, annotate
|   +-- trainer.py                  #     Training wrapper (CytoneighborWrapper, TrainingManager)
|   +-- utils.py                    #     Config, data loading, metrics, visualization
|
+-- tutorials/                      #  Interactive guides
|   +-- tutorial_annotation.ipynb   #     Cell-type annotation with unknown-gating
|   +-- tutorial_integration.ipynb  #     Boundary-preserving batch integration
|   +-- README.md                   #     Tutorial prerequisites & dataset info
|
+-- experiments/                    #  Paper reproducibility scripts
|   +-- Annotation/                 #     LOBO annotation benchmarks (9 baselines)
|   +-- Rebatch/                    #     Batch integration benchmarks
|   +-- benchmark_resolution.py     #     Resolution stress-test (NK vs NKT)
|   +-- mechanism_analysis.py       #     Gene-CutMix mechanism study
|   +-- disease_discovery.py        #     DKD disease-state matching
|
+-- tests/                          #  Automated tests
|   +-- test_quick_run.py           #     30-second sanity check (no data download)
|
+-- repro/                          #  Exact reproducibility assets
|   +-- locks/                      #     Conda & pip environment lock files
|   +-- container/                  #     Docker & Apptainer recipes
|   +-- statistics/                 #     Manuscript statistical tables
|   +-- qc_threshold_manifest.csv   #     Preprocessing and QC thresholds
|
+-- scripts/
|   +-- download_data.py            #  Google Drive dataset download helper
|
+-- configs/
|   +-- config.yaml                 #  Experiment configuration
|
+-- data/
|   +-- README.md                   #  Dataset inventory & download instructions
|
+-- results/
|   +-- README.md                   #  Pre-computed results inventory
|
+-- paper/                          #  Manuscript LaTeX source & figures
|   +-- main.tex
|   +-- supplementary.tex
|   +-- figures/
|
+-- .github/workflows/
|   +-- tests.yml                   #  Auto-test on push & PR
|   +-- publish.yml                 #  Auto-publish to PyPI on release
|
+-- pyproject.toml                  #  Package metadata, dependencies, build config
+-- main.py                         #  CLI: run-all, run-experiment, train
+-- MANIFEST.in                     #  Source distribution content control
+-- README.md                       #  You are here
+-- LICENSE                         #  MIT
+-- .gitignore
```

---

## &#128200; Reproducing the Paper

### Prerequisites

1. Clone the repository and install with the Conda lock file:

```bash
git clone https://github.com/checce/Cytoneighbor.git
cd Cytoneighbor
conda env create -f repro/locks/environment-smn.yml
conda activate smn
pip install -e .
```

2. Download the required datasets from
   [Google Drive](https://drive.google.com/drive/folders/11PuJRaw_JC7xJhphQklac7eg9AuGHsCE?usp=sharing)
   and place `.h5ad` files in `data/`. See `data/README.md` for the dataset
   inventory.

### Running experiments

```bash
# Run all experiments
python main.py run-all

# Run a specific experiment
python main.py run-experiment benchmark-resolution
python main.py run-experiment mechanism-analysis
python main.py run-experiment disease-discovery

# Train a model on your own dataset
python main.py train --dataset data/your_data.h5ad --cell-type cell_type
```

### Pre-computed results

Full experiment outputs (~8 GB, including trained model checkpoints and
intermediate analysis files) are archived on Google Drive:
[Results Archive](https://drive.google.com/drive/folders/1S41DI2W0Iy3YsorIWSHqWgZaWn2zZG7A?usp=sharing)

Statistical tables cited in the manuscript and Supplementary Materials are
included in the repository under `repro/statistics/`.

### Containerized reproduction

Docker and Apptainer recipes are provided in `repro/container/`:

```bash
# Docker
docker build -t cytoneighbor -f repro/container/Dockerfile.smn .
docker run --gpus all -it cytoneighbor python main.py run-all

# Apptainer
apptainer build cytoneighbor.sif repro/container/Apptainer.smn.def
apptainer run --nv cytoneighbor.sif python main.py run-all
```

---

## &#128188; Using Cytoneighbor on Your Own Data

### Minimum requirements

- **Reference AnnData**: must have a column in `.obs` with cell-type labels
  (e.g. `"cell_type"`, `"final_annotation"`)
- **Query AnnData**: raw or log-normalised counts (same gene set as reference)
- **Hardware**: runs on CPU, but GPU (CUDA) is recommended for datasets with
  more than 50,000 cells

### Recommended preprocessing

```python
import scanpy as sc

# Normalize and log-transform
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Select highly variable genes (optional but recommended for large datasets)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
adata = adata[:, adata.var.highly_variable]

# Cytoneighbor works with the raw expression matrix — no PCA required.
```

### Common questions

**Q: My query has cell types not in the reference. What happens?**

A: That is exactly what unknown-gating is designed for. Cells with weak
neighbour evidence across *all* reference types are retained as `"Unknown"`
instead of being misclassified. You can inspect these cells separately to
identify novel populations.

**Q: Can I use Cytoneighbor for cross-species mapping?**

A: Yes. Train on a mouse reference dataset and annotate a human query (or vice
versa) using orthologous gene sets as input features. The SymRelEnc model
learns discriminative cell-pair relationships that generalize across species
when gene symbols are matched.

**Q: How long does training take?**

A: On a dataset with 50,000 cells and 3,000 genes, one epoch takes ~30 seconds
on a single GPU. The default 50 epochs with early stopping typically converges
in under 10 minutes. CPU training is 3-5× slower.

**Q: Can I save and share trained models?**

A: Yes. Use `cn.save_model(adata, "model.zip")` and
`cn.load_model(adata, "model.zip")`. Saved models contain the full SymRelEnc
weights and can be distributed independently.

---

## &#128172; Citation

If you use Cytoneighbor in your research, please cite:

> Hu H. et al. *Inspectable neighbor evidence for single-cell state interpretation with Cytoneighbor*. (in preparation).

---

## &#129309; Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Install development dependencies (`pip install -e ".[dev]"`)
4. Run the test suite (`python tests/test_quick_run.py`)
5. Submit a pull request

For bug reports, feature requests, or questions, please
[open an issue](https://github.com/checce/Cytoneighbor/issues).

---

## &#128220; License

MIT License. See [LICENSE](LICENSE) for full text.

---

## &#128231; Contact

For questions or support:
- [GitHub Issues](https://github.com/checce/Cytoneighbor/issues)
- Email: checce@github

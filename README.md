<p align="center">
  <h1 align="center">&#128300; Cytoneighbor</h1>
  <p align="center">
    <strong>Neighbour-evidence-guided single-cell state interpretation</strong>
  </p>
  <p align="center">
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

[Get started](#install-cytoneighbor) · [Tutorials](tutorials/) · [Download data](data/) · [Reproduce the study](repro/)

---

## &#128269; What is Cytoneighbor?

**Cytoneighbor** is a neighbour-evidence framework for interpretable single-cell
state analysis. Cytoneighbor constructs a **ranked neighbour list** for every query cell.
Inspect the supporting neighbours and scores to see where a label or a graph
has strong evidence and where it needs closer review.

The framework combines three core components:

- **Scalable candidate retrieval** with FAISS HNSW to shortlist candidates before neural scoring. The revision reports measured resource use up to 100,000 cells in a single PBMC workload
- **Exact pairwise re-scoring** via a **Symmetric Relation Encoder (SymRelEnc)** that learns discriminative cell-pair relationships
- **Gene-CutMix**, a training-time augmentation strategy evaluated under a controlled pancreatic delta-cell scarcity protocol

Cytoneighbor fits naturally into a standard Scanpy workflow: train a model,
build a neighbour graph, and then use it as a drop-in replacement for any
downstream function that accepts a `neighbors_key`.

> &#128214; **Paper**: Hu H. et al. *Cytoneighbor: a scalable neighbor evidence framework for single-cell state analysis* (in preparation).

---

## &#127775; Why Cytoneighbor?

| Your goal | Typical approach | What Cytoneighbor does differently |
|---|---|---|
| Annotate cells | Closed-set forced label | **Unknown-gating**: defer low-evidence calls and inspect the confidence, margin and entropy |
| Integrate batches | Global alignment blurs boundaries | **Boundary-preserving**: inspect how well neighbourhoods retain related-type boundaries and measure the trade-off with batch mixing |
| Find rare cells | Oversampling or downweighting | **Gene-CutMix**: test augmentation with real reference cells at inference |
| Compare disease vs healthy | Pooled-control differential expression | **State-matched**: separates compositional redistribution from within-state expression changes |

---

<a name="install-cytoneighbor"></a>

## &#128230; Installation

### From source (recommended)

Use Python 3.10 or 3.11 in a fresh environment. Run the following from a terminal.

```bash
git clone https://github.com/checce/Cytoneighbor.git
cd Cytoneighbor

# Core package and quick sanity check
python -m pip install -e .

# Or install the notebook and test dependencies as well
python -m pip install -e ".[dev]"
```

The default FAISS backend is CPU-based. PyTorch can use CUDA on a compatible
system. See the [recorded environment](repro/locks/) and
[historical container notes](repro/container/) for the original software stack.

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

Try the [annotation tutorial](tutorials/tutorial_annotation.ipynb) with synthetic data before downloading a real dataset.

---

## &#128218; API at a Glance

```python
import cytoneighbor as cn

# --- One-liner for the most common use case ---
result = cn.quick_annotate(ref, "cell_type", query)

# --- Step-by-step for full control ---
cn.load_data(ref, data=ref.X, labels=ref.obs["cell_type"].to_numpy())
cn.fit(ref)
cn.transform(ref, n_neighbors=15)
cn.annotate(ref, "cell_type", query, enable_unknown=True)

# --- Save and load trained models ---
cn.save_model(ref, "my_model.zip")
cn.load_model(ref, "my_model.zip")
```

| Function | What it does |
|---|---|
| `quick_annotate(ref, label_key, query)` | Train + annotate in one call |
| `fit(adata, ...)` | Train the SymRelEnc model after loading expression and labels |
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
cn.load_data(ref, data=ref.X, labels=ref.obs["your_cell_type_column"].to_numpy())
cn.fit(ref)
cn.annotate(ref, "your_cell_type_column", qry, enable_unknown=True)

# 3. Use predictions downstream
qry.write("annotated.h5ad")
qry.obs[["cytoneighbor_prediction", "cytoneighbor_confidence"]].to_csv("predictions.csv")

```

When `enable_unknown=True`, Cytoneighbor writes diagnostic columns to
`query.obs`:

| Column | Meaning |
|---|---|
| `cytoneighbor_prediction` | Predicted label (or *Unknown*) |
| `cytoneighbor_confidence` | Top-class preference score (0–1) |
| `cytoneighbor_margin` | Gap between the two highest class preference scores |
| `cytoneighbor_norm_entropy` | Normalised prediction entropy (0–1) |
| `cytoneighbor_is_unknown` | Boolean: was this cell gated as Unknown? |

**How to interpret**. Treat *Unknown* as a deferral for review. Default thresholds
are development heuristics. The revision reports known-type error screening and
reference-absent types separately. Inspect both error interception and correct
labels deferred before transferring the rule to a new dataset.

### &#128225; Boundary-preserving Batch Integration

Build a graph across batches and examine related cell-type boundaries alongside
batch mixing. The study compares graph clustering and label transfer separately.

```python
import cytoneighbor as cn
import scanpy as sc

# 1. Load multi-batch data
adata = sc.read_h5ad("your_multi_batch_data.h5ad")

# 2. Standard preprocessing
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000, batch_key="your_batch_column", subset=True)

# 3. Build Cytoneighbor neighbour graph
cn.load_data(adata, data=adata.X, labels=adata.obs["your_cell_type_column"].to_numpy())
cn.fit(adata)
cn.transform(adata, n_neighbors=15)

# 4. Use the graph in Scanpy
sc.tl.umap(adata, neighbors_key="neighbors")          # UMAP from Cytoneighbor graph
sc.tl.leiden(adata, neighbors_key="neighbors")         # clustering, install .[dev]
sc.tl.rank_genes_groups(adata, groupby="leiden")       # marker detection

# 5. Evaluate integration quality
# Compare batch mixing (iLISI) and cell-type purity (cLISI)
# against uncorrected baseline — see tutorials for full example
```

Pass `neighbors_key="neighbors"` to Scanpy to use the Cytoneighbor graph.

### &#128300; Rare-cell Recovery with Gene-CutMix

Gene-CutMix generates synthetic profiles for training. The revision compares
several controls under pancreatic delta-cell scarcity and uses real cells in
both inference reference pools. The direction of the effect depends on the
reference pool, comparator and seed.

```python
# Gene-CutMix is enabled by default during fit().
# Tune its behaviour via optional parameters:
cn.load_data(adata, data=adata.X, labels=adata.obs["cell_type"].to_numpy())
cn.fit(
    adata,
    synth_enable=True,           # enable Gene-CutMix (default: True)
    synth_mult_per_dt=2.0,       # synthesis multiplier per rare type
    synth_cap_per_dt=500,        # upper cap on synthetic cells per type
)
```

### &#128137; Disease-state Matching

For disease-vs-healthy comparisons, Cytoneighbor constructs state-matched
neighbourhoods for comparing state composition and within-state expression.
The DKD example uses donors as the statistical units and reports exploratory
patterns alongside adjusted results.

```python
# Train on healthy reference, annotate disease query
cn.load_data(healthy_ref, data=healthy_ref.X, labels=healthy_ref.obs["cell_type"].to_numpy())
cn.fit(healthy_ref)
cn.annotate(healthy_ref, "cell_type", disease_query, enable_unknown=False)

# Inspect the predicted reference labels before assigning disease states.
print(disease_query.obs["cytoneighbor_prediction"].value_counts())
```

---

## &#128214; Tutorials

Open the [annotation](tutorials/tutorial_annotation.ipynb) and
[integration](tutorials/tutorial_integration.ipynb) notebooks to follow the workflows.

### 1. Cell-type Annotation

Train → annotate → visualize → export. Covers unknown-gating, confidence
diagnostics, and adapting the workflow to your own data.

### 2. Batch Integration

Build a neighbour graph → visualize integration quality → run downstream
clustering and marker detection → evaluate against raw baseline.

> &#128161; **Mock data fallback**: both tutorials automatically generate
> synthetic data if the real dataset is not found, so you can follow along
> immediately. See the [tutorial guide](tutorials/) for prerequisites and dataset info.

---

## &#128451; Project Structure

```text
Cytoneighbor/
|
+-- cytoneighbor/                   #  Core package
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
|   +-- Annotation/                 #     LOBO annotation and baseline drivers
|   +-- Rebatch/                    #     Batch integration benchmarks
|   +-- benchmark_resolution.py     #     Resolution stress-test (NK vs NKT)
|   +-- mechanism_analysis.py       #     Gene-CutMix mechanism study
|   +-- disease_discovery.py        #     DKD disease-state matching
|
+-- tests/                          #  Automated tests
|   +-- test_quick_run.py           #     30-second sanity check (no data download)
|
+-- repro/                          #  Reproduction guides and frozen evidence
|   +-- locks/                      #     Conda & pip environment lock files
|   +-- container/                  #     Docker & Apptainer recipes
|   +-- revision/                  #  Revision tables and summary checker
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
|   +-- input-checksums.csv         #  Exact input fingerprints
|
+-- results/
|   +-- README.md                   #  Pre-computed results inventory
|
+-- paper/                          #  Historical LaTeX source & figures
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

<a name="reproduce-the-paper"></a>

## &#128200; Reproducing the Paper

### Prerequisites

1. Follow the [source installation steps](#install-cytoneighbor) from the repository root.

2. Download the required datasets from
   [Google Drive](https://drive.google.com/drive/folders/11PuJRaw_JC7xJhphQklac7eg9AuGHsCE?usp=sharing)
   and place `.h5ad` files in `data/`. See the [dataset guide](data/) for filenames and verification.

### Running the original workflows

Install the additional Python dependencies with `python -m pip install -e ".[benchmarks]"`.
Run these commands from the repository root. The
[experiment guide](repro/#original-experiments) lists input files and baseline dependencies.

```bash
# Run individual experiments via CLI
python main.py run-experiment benchmark-resolution
python main.py run-experiment mechanism-analysis
python main.py run-experiment disease-discovery

# Leave-one-batch-out annotation with Cytoneighbor
python experiments/Annotation/Cytoneighbor.py  # runs Cytoneighbor on the default dataset

# Batch integration benchmarks
python experiments/Rebatch/a-main_remove.py

# Train a model on your own dataset
python main.py train --dataset "data/your_data.h5ad" --cell-type cell_type
```

The revision controls use the distinct protocols in the
[revision evidence guide](repro/revision/).

### Pre-computed results and revision checks

- [Original results archive](results/) lists the model checkpoints and figure
  source outputs available on Google Drive.
- [Original statistical summaries](repro/statistics/) provide compact tables.
- [Revision evidence](repro/revision/) provides frozen tables and compact replay inputs.
  You can check their integrity and replay key revision metrics without downloading models or cell matrices.

```bash
python repro/revision/check_summaries.py  # standard library only
python repro/revision/replay_tables.py    # requires NumPy
```

The replay starts from archived rank counts, donor estimates, graph contingency
counts and recorded resource measurements. Repeating training and inference for
every revision control requires additional frozen inputs and checkpoints.

---

## &#128188; Using Cytoneighbor on Your Own Data

### Minimum requirements

- **Reference AnnData**: must have a column in `.obs` with cell-type labels
  (e.g. `"cell_type"`, `"final_annotation"`)
- **Query AnnData**: align the feature names to the reference and use the same
  expression scale and preprocessing in both datasets.
- **Hardware**: a CPU is sufficient for the tutorials. Memory and runtime for
  larger jobs depend on the feature count, model, graph and available hardware.

### Recommended preprocessing

```python
import scanpy as sc

# Start from counts. Skip these steps for already log-normalised input.
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Select highly variable genes (optional but recommended for large datasets)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
adata = adata[:, adata.var.highly_variable]

# Use the same selected genes for reference and query.
```

### Common questions

**Q: My query has cell types not in the reference. What happens?**

A: Review deferred cells alongside their neighbour scores and marker evidence.
A reference-absent type can still receive a known label, and a correctly
classified known cell can be deferred. The default gate is heuristic.

**Q: Can I use Cytoneighbor for cross-species mapping?**

A: Cross-species mapping would require a shared ortholog feature set and
independent validation on the target species. The study evaluated within-species tasks.

**Q: How long does training take?**

A: Use the [recorded PBMC stage measurements](repro/revision/tables/pbmc_stage_measurements.csv)
as a reference for the measured configuration. Training budgets, memory peaks
and hardware differ across experiments.

**Q: Can I save and share trained models?**

A: Yes. Use `cn.save_model(adata, "model.zip")` and
`cn.load_model(adata, "model.zip")`. Saved models contain the full SymRelEnc
weights and can be distributed independently.

---

## &#128172; Citation

If you use Cytoneighbor in your research, please cite:

> Hu H. et al. *Cytoneighbor: a scalable neighbor evidence framework for single-cell state analysis*. (in preparation).

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

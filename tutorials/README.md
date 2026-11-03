# Tutorials

Step-by-step guides to help you get the most out of Cytoneighbor.

---

## Available Tutorials

### 1. Cell-type Annotation with Unknown-gating

**What you will learn:**
- Train a Cytoneighbor model on a reference dataset
- Transfer labels to a query with **unknown-gating** enabled
- Visualize predictions on UMAP and inspect confidence scores
- Export annotated data for downstream analysis
- Adapt the workflow to your own datasets

**Best for:** Users who have a well-annotated reference and want to annotate
an unlabelled query -- especially when the query may contain cell types not
present in the reference.

---

### 2. Boundary-preserving Batch Integration

**What you will learn:**
- Build a Cytoneighbor neighbour graph across multiple batches
- Visualize integration quality (batch mixing vs. cell-type separation)
- Run downstream analysis (clustering, marker detection) using the graph
- Evaluate batch correction against a raw baseline
- Adapt the workflow to your own multi-batch data

**Best for:** Users integrating data from multiple donors, experiments, or
protocols who want to preserve biologically meaningful boundaries between
closely related cell types.

---

## Prerequisites

- Python 3.10 or later
- A source checkout with Cytoneighbor and the tutorial dependencies installed
- The tutorials use a publicly available immune-cell dataset, or fall back
  to synthetic mock data if the dataset is not found

---

## Running Locally

```bash
# Clone and install
git clone https://github.com/checce/Cytoneighbor.git
cd Cytoneighbor
python -m pip install -e ".[dev]"

# Launch tutorials
jupyter notebook tutorials/
```

---

## Data

Tutorials expect the **Human Immune** dataset in `data/Human immune.h5ad`.
If the file is not found, the notebooks automatically generate synthetic
mock data so you can follow along immediately.

Download the real dataset from
[Google Drive](https://drive.google.com/drive/folders/11PuJRaw_JC7xJhphQklac7eg9AuGHsCE?usp=sharing).

See the [dataset guide](../data/) for the full inventory.

---

## Next Steps

- Read the [API reference](../README.md#-api-at-a-glance) in the main README
- Run `python tests/test_quick_run.py` to verify your installation
- Follow the [reproducibility guide](../repro/) to run the original workflows or check the revision tables

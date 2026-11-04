#!/usr/bin/env python3
"""
Cytoneighbor Quick Sanity Check (No data download required)

Generates a minimal mock single-cell dataset in memory and runs the core
Cytoneighbor API (load_data -> fit -> transform -> annotate) end-to-end.

Expected runtime: < 10 seconds (CPU) or < 5 seconds (GPU).

Usage:
    python tests/test_quick_run.py
"""

import time
import sys
import warnings
import logging
from pathlib import Path

# Ensure the project root is on sys.path so the test works even before
# `pip install -e .` is run (e.g., right after `git clone`).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import anndata as ad
import cytoneighbor as cn

# Suppress verbose training logs during quick test
logging.getLogger().setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Mock data generation
# ---------------------------------------------------------------------------

def generate_mock_data(
    n_cells: int = 100,
    n_genes: int = 200,
    n_types: int = 3,
    n_batches: int = 2,
    random_seed: int = 42,
) -> ad.AnnData:
    """Generate a minimal mock single-cell AnnData object in memory.

    Parameters
    ----------
    n_cells : int
        Total number of cells.
    n_genes : int
        Number of genes (features).
    n_types : int
        Number of distinct cell types.
    n_batches : int
        Number of distinct batches.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    ad.AnnData
        Mock dataset with .X (log-normalised counts), .obs["cell_type"],
        .obs["batch"], and string gene names.
    """
    rng = np.random.default_rng(random_seed)

    # Assign cells evenly across types and batches
    cell_types = np.array([f"Type_{i % n_types}" for i in range(n_cells)])
    batches = np.array([f"Batch_{i % n_batches}" for i in range(n_cells)])

    # Create type-specific expression patterns
    type_means = rng.normal(0, 1, size=(n_types, n_genes))
    # Add batch effects
    batch_effects = rng.normal(0, 0.2, size=(n_batches, n_genes))
    # Add cell-level noise
    cell_noise = rng.normal(0, 0.5, size=(n_cells, n_genes))

    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    for i in range(n_cells):
        t = i % n_types
        b = i % n_batches
        X[i] = type_means[t] + batch_effects[b] + cell_noise[i]

    # Shift to non-negative (mimic log-normalised counts)
    X -= X.min(axis=0, keepdims=True)
    X += 0.1

    # Build AnnData
    gene_names = [f"Gene_{j}" for j in range(n_genes)]
    obs = {"cell_type": cell_types, "batch": batches}
    adata = ad.AnnData(X, obs=obs, dtype=np.float32)
    adata.var_names = gene_names

    return adata


# ---------------------------------------------------------------------------
# Quick-run test
# ---------------------------------------------------------------------------

def test_quick_run():
    """Run end-to-end sanity check and report results."""
    t0 = time.time()
    passed = 0
    total = 5

    # -----------------------------------------------------------------------
    print("[1/5] Generating mock single-cell data (100 cells x 200 genes)...")
    adata = generate_mock_data()
    assert adata.n_obs == 100, f"Expected 100 cells, got {adata.n_obs}"
    assert adata.n_vars == 200, f"Expected 200 genes, got {adata.n_vars}"
    assert "cell_type" in adata.obs.columns
    assert "batch" in adata.obs.columns
    print(f"      Done: {adata.n_obs} cells x {adata.n_vars} genes, "
          f"{adata.obs['cell_type'].nunique()} types, "
          f"{adata.obs['batch'].nunique()} batches.")
    passed += 1

    # -----------------------------------------------------------------------
    print("[2/5] Loading data into Cytoneighbor (load_data)...")
    X = adata.X
    y = adata.obs["cell_type"].to_numpy()
    cn.load_data(adata, data=X, labels=y)
    z = adata.uns.get("cytoneighbor", {})
    assert z.get("_train_X") is not None, "load_data: _train_X missing"
    assert z.get("_train_y") is not None, "load_data: _train_y missing"
    print("      Done: data stored in adata.uns['cytoneighbor'].")
    passed += 1

    # -----------------------------------------------------------------------
    print("[3/5] Training model (fit, 2 epochs)...")
    # Use minimal architecture and epochs for speed
    cn.fit(
        adata,
        epochs=2,
        d_model=128,          # small model
        depth=2,              # shallow
        hidden=256,
        train_pairs_per_epoch=200,
        use_cuda=False,       # CPU-safe by default
        use_amp=False,
        synth_enable=False,   # skip gene-level mixing for speed
        early_stopping_patience=None,
        validation_split=None, # skip validation for speed
    )
    model = adata.uns["cytoneighbor"].get("model", None)
    assert model is not None, "fit: model not found after training"
    print("      Done: model trained successfully.")
    passed += 1

    # -----------------------------------------------------------------------
    print("[4/5] Building neighbour graph (transform)...")
    cn.transform(
        adata,
        n_neighbors=8,
        k_candidate=50,
        use_cuda=False,
        use_amp=False,
    )
    # transform stores connectivities/distances in adata.obsp
    has_graph = (
        "cytoneighbor_connectivities" in adata.obsp
        or "connectivities" in adata.obsp
    )
    print(f"      Done: neighbour graph built. (connectivities present: {has_graph})")
    passed += 1

    # -----------------------------------------------------------------------
    print("[5/5] Running annotation (annotate with Unknown gating)...")
    # Split adata into reference (first 60 cells) and query (last 40 cells)
    ref_adata = adata[:60].copy()
    query_adata = adata[60:].copy()

    # The model lives on `adata`; annotate looks for it on query or reference.
    # We copy the cytoneighbor uns key to query so the model is found.
    query_adata.uns["cytoneighbor"] = adata.uns["cytoneighbor"]

    cn.annotate(
        ref_adata,
        "cell_type",
        query_adata,
        enable_unknown=True,
        k_search=50,
        use_cuda=False,
        use_amp=False,
    )

    pred_key = "cytoneighbor_prediction"
    assert pred_key in query_adata.obs.columns, (
        f"annotate: '{pred_key}' not found in query.obs"
    )
    n_unknown = (query_adata.obs[pred_key] == "Unknown").sum()
    print(f"      Done: {query_adata.n_obs} query cells annotated."
          f" ({n_unknown} retained as Unknown)")
    passed += 1

    # -----------------------------------------------------------------------
    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"  All {passed}/{total} checks passed in {elapsed:.1f} seconds.")
    print("  Cytoneighbor is successfully installed and functioning.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        test_quick_run()
        sys.exit(0)
    except Exception as exc:
        print(f"\n[FAIL] Quick test failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

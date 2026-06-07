"""
Cytoneighbor --- Neighbour-evidence-guided single-cell state interpretation.

Cytoneighbor is a neighbour-evidence framework for inspectable single-cell
state interpretation in single-cell transcriptomics. It combines scalable
candidate retrieval (FAISS HNSW) with symmetric pairwise re-scoring
(SymRelEnc) and Gene-CutMix data augmentation to address four key tasks:

1. **Unknown-gating annotation** --- intercepts forced closed-set labels
   under incomplete reference coverage.
2. **Boundary-preserving integration** --- maintains biologically meaningful
   boundaries between related cell types during batch alignment.
3. **Rare-cell recovery** --- restores displaced rare-cell neighbour evidence
   under severe abundance imbalance via Gene-CutMix.
4. **Disease-state matching** --- constructs state-matched disease-healthy
   comparisons that separate compositional from within-state changes.

Quick Start
-----------
>>> import cytoneighbor as cn
>>> import scanpy as sc

>>> ref = sc.read_h5ad("reference.h5ad")
>>> qry = sc.read_h5ad("query.h5ad")

>>> cn.fit(ref, label_key="cell_type")
>>> cn.transform(qry, ref)
>>> cn.annotate(ref, "cell_type", qry, enable_unknown=True)

Main API
--------
fit          Train a Cytoneighbor model on reference data.
transform    Build a neighbour graph for downstream Scanpy analysis.
annotate     Transfer labels using ranked neighbour evidence.
load_data    Load and preprocess single-cell expression data.
save_model   Serialize a trained model to disk.
load_model   Deserialize a trained model from disk.
quick_annotate  End-to-end annotation in one call (fit + annotate).
"""

from cytoneighbor.model import (
    fit,
    transform,
    annotate,
    load_data,
    save_model,
    load_model,
)

from cytoneighbor.trainer import CytoneighborWrapper

import anndata as ad
from typing import Optional

__version__ = "0.1.0"
__all__ = [
    "fit",
    "transform",
    "annotate",
    "load_data",
    "save_model",
    "load_model",
    "quick_annotate",
    "CytoneighborWrapper",
]


def quick_annotate(
    reference: ad.AnnData,
    label_key: str,
    query: Optional[ad.AnnData] = None,
    *,
    enable_unknown: bool = True,
    epochs: int = 50,
    early_stopping_patience: int = 10,
    **kwargs,
) -> ad.AnnData:
    """End-to-end annotation in a single call.

    This convenience function runs :func:`fit` and :func:`annotate` in one
    step, covering the most common use case: training on a reference and
    transferring labels to a query dataset with unknown gating enabled.

    Parameters
    ----------
    reference : anndata.AnnData
        Reference dataset with known cell-type labels in ``reference.obs``.
    label_key : str
        Column name in ``reference.obs`` containing cell-type labels.
    query : anndata.AnnData, optional
        Query dataset to annotate. If ``None``, ``reference`` is used as
        the query (self-annotation for diagnostics).
    enable_unknown : bool
        If ``True`` (default), cells with weak neighbour evidence are
        retained as ``"Unknown"`` rather than forced into the closest label.
    epochs : int
        Number of training epochs.
    early_stopping_patience : int
        Stop training after this many epochs without improvement.
    **kwargs
        Additional arguments passed to :func:`fit`.

    Returns
    -------
    anndata.AnnData
        The query dataset with predictions written to
        ``query.obs["cytoneighbor_prediction"]`` and diagnostic metrics
        in ``query.obs`` and ``query.obsm``.

    Examples
    --------
    >>> import cytoneighbor as cn
    >>> import scanpy as sc

    >>> ref = sc.read_h5ad("reference.h5ad")
    >>> qry = sc.read_h5ad("query.h5ad")

    >>> result = cn.quick_annotate(ref, "cell_type", qry)
    >>> print(result.obs["cytoneighbor_prediction"].value_counts())
    """
    fit(reference, label_key=label_key, epochs=epochs,
        early_stopping_patience=early_stopping_patience, **kwargs)

    if query is None:
        query = reference

    annotate(
        reference,
        label_key,
        query,
        enable_unknown=enable_unknown,
    )

    return query

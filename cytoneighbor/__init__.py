"""Cytoneighbor provides inspectable neighbor evidence for single-cell analysis.

The framework combines candidate retrieval with supervised symmetric pairwise
rescoring. It exposes ranked neighbor evidence for annotation and graph-based
analysis. Optional Unknown retention defers queries for review. The study also
evaluates training augmentation by mixing gene expression and compares healthy
and disease cells within inferred proximal tubule states.

Quick start
-----------
>>> import cytoneighbor as cn
>>> import scanpy as sc
>>> ref = sc.read_h5ad("reference.h5ad")
>>> qry = sc.read_h5ad("query.h5ad")
>>> result = cn.quick_annotate(ref, "cell_type", qry)
>>> print(result.obs["cytoneighbor_prediction"].value_counts())

Use `load_data` before calling `fit` directly. Use `transform` to build a
neighbor graph on an AnnData object with a trained or loaded model.
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
    load_data(reference, data=reference.X, labels=reference.obs[label_key].to_numpy())
    fit(reference, epochs=epochs,
        early_stopping_patience=early_stopping_patience, **kwargs)

    if query is None:
        query = reference

    annotate(
        reference,
        label_key,
        query,
        enable_unknown=enable_unknown,
        use_cuda=kwargs.get("use_cuda", True),
        use_amp=kwargs.get("use_amp", True),
    )

    return query

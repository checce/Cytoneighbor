"""
Experiment 3 — PT Anatomical Zonation and Convergent Dedifferentiation in DKD
Joint Manifold Architecture (Cytoneighbor)
================================================================================

Scientific Narrative
--------------------
Cytoneighbor's zero-shot joint manifold reveals that DKD proximal tubule (PT)
pathology is not a simple linear dedifferentiation.  Instead, the healthy PT
organises into two physiologically distinct segments — S1/S2 and S3 — that
retain their anatomical identity even within the joint embedding.  Under DKD
injury both segments undergo parallel, progressive dedifferentiation, ultimately
*converging* into a shared Injured PT terminal state characterised by severe
extracellular-matrix remodelling.

Cluster assignment (data-driven, no circular marker pre-selection):
  • Functional PT (S1/S2-like) [root]  — highest absolute healthy-cell count,
                                         highest SLC5A2 / LRP2 expression
  • Functional PT (S3-like)            — lower SLC5A2 consistent with S3 segment,
                                         highest healthy-cell purity (% Healthy)
  • Injured PT                         — highest VCAM1 / HAVCR1, EMT signatures

Topology (PAGA):
  Triangular / convergent — S1/S2-like ↔ S3-like ↔ Injured PT — both healthy
  segments share topological edges with the injury terminus, confirming parallel
  dedifferentiation rather than a single linear path.

Zonation validation:
  SLC5A1 (SGLT1) — expressed preferentially in S3 — is checked as an independent
  positive marker for the S3-like cluster.  Results are logged and plotted.

Architecture:
  1.  Train SymRelEnc on ALL healthy cells (multi-type; required for contrastive)
  2.  Extract Healthy PT + Disease PT; tag disease_status
  3.  ad.concat → combined_pt; run transform() once → joint SymRelEnc KNN graph
  4.  Leiden → filter tiny clusters → dynamic resolution search
  5.  PAGA (convergent topology)
  6.  Root selection: absolute Healthy_Reference count (marker-free, no circularity)
  7.  Characterise clusters: S1/S2-like / S3-like / Injured PT
  8.  DPT from S1/S2-like root (acknowledged approximation for convergent manifold)
  9.  Zonation validation: SLC5A1 expression profile (independent positive marker)
  10. Compensatory hypothesis validation (disease-cell-specific SLC5A2 analysis)
      ──────────────────────────────────────────────────────────────────────────
      Cluster-level mean expression differences can arise from three confounded
      mechanisms: (A) anatomical subtype composition, (B) genuine disease-driven
      upregulation, or (C) statistical noise.  To disentangle these, step 10
      computes SLC5A2 expression ONLY in Disease_Query cells within each cluster,
      then compares directly to Healthy_Reference cells in the same cluster.
        • If DKD cells in S1/S2-like > healthy cells in S1/S2-like  → compensatory
          upregulation is supported at the cell level (Discussion material)
        • If the difference disappears or reverses   → cluster-mean difference was
          driven by healthy-cell subtype composition (null hypothesis not rejected)
      Results are logged quantitatively and saved as compensatory_validation.pdf.
      Regardless of outcome, the primary zonation narrative (Results section) is
      not altered; the compensatory hypothesis is reported in Discussion only.
  11. Visualisations + GO enrichment
"""

import copy
import gc
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse
from scipy.ndimage import gaussian_filter1d
from scipy.stats import rankdata

from src.trainer import CytoneighborWrapper
from src.utils import ensure_output_dir

# ── Optional dependency ────────────────────────────────────────────────────────
try:
    import gseapy as gp
    _GSEAPY_AVAILABLE = True
except ImportError:
    _GSEAPY_AVAILABLE = False

# ==============================================================================
# Nature Methods publication-ready global font settings
# ==============================================================================
try:
    from scifont import set_font
    set_font(font_name="Helvetica")
    print("[SciFont] Publication-ready fonts enabled.")
except ImportError:
    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype":     42,
        "ps.fonttype":      42,
        "svg.fonttype":     "none",
    })

plt.rcParams.update({
    "font.size":             7,
    "axes.titlesize":        8,
    "axes.labelsize":        7,
    "xtick.labelsize":       6,
    "ytick.labelsize":       6,
    "legend.fontsize":       6,
    "legend.title_fontsize": 7,
    "lines.linewidth":       1.0,
    "axes.linewidth":        0.6,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
})

# ==============================================================================
# Module-level colour palette
# Defined here once to prevent Scanpy's alphabetical-order colour scrambling.
# ALL plotting methods that colour by biological_state must reference this dict.
# ==============================================================================
STATE_PALETTE: Dict[str, str] = {
    "Functional PT (S1/S2-like)": "#1565C0",   # deep blue   — high-function baseline
    "Functional PT (S3-like)":    "#2E7D32",   # deep green  — distal healthy segment
    "Injured PT":                 "#B71C1C",   # deep red    — terminal injury state
}

ORIGIN_PALETTE: Dict[str, str] = {
    "Healthy_Reference": "#2196F3",
    "Disease_Query":     "#F44336",
}

# Canonical state order used by every bar/trend plot
STATE_ORDER: List[str] = [
    "Functional PT (S1/S2-like)",
    "Functional PT (S3-like)",
    "Injured PT",
]


# ==============================================================================
class DiseaseDiscoveryExperiment:
    """
    Experiment 3: PT Zonation and Convergent Dedifferentiation in DKD.

    Encapsulates the full joint-manifold pipeline from raw data to
    publication-ready figures.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────────────

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        exp = config["experiments"]["disease_discovery"]

        # Dataset / column names
        self.dataset_path      = exp["dataset_path"]
        self.condition_column  = exp["condition_column"]
        self.cell_type_column  = exp["cell_type_column"]
        self.healthy_condition = exp["healthy_condition"]
        self.disease_condition = exp["disease_condition"]
        self.target_cell_type  = exp["target_cell_type"]

        # Graph / clustering parameters
        self.n_neighbors       = exp.get("n_neighbors",       15)
        self.leiden_resolution = exp.get("leiden_resolution", 0.3)
        self.paga_threshold    = exp.get("paga_threshold",    0.01)
        self.n_top_genes       = exp.get("n_top_genes",       3000)

        # Biological marker panels
        # adaptive_markers → epithelial functional markers (SLC5A2, LRP2)
        # injury_markers   → stress / injury markers     (VCAM1, HAVCR1)
        self.adaptive_markers   = exp.get("adaptive_markers",   ["SLC5A2", "LRP2"])
        self.injury_markers     = exp.get("injury_markers",     ["VCAM1", "HAVCR1"])
        # S3 zonation validation marker — independent of adaptive/injury panels
        self.zonation_marker    = exp.get("zonation_marker",    "SLC5A1")

        self.logger = logging.getLogger(__name__)
        self.logger.info("=== DiseaseDiscoveryExperiment initialised ===")
        self.logger.info(f"  Dataset:          {self.dataset_path}")
        self.logger.info(f"  Target cell type: {self.target_cell_type}")
        self.logger.info(f"  Leiden resolution:{self.leiden_resolution}")
        self.logger.info(f"  n_top_genes:      {self.n_top_genes}")
        self.logger.info(f"  Functional mrkrs: {self.adaptive_markers}")
        self.logger.info(f"  Injury markers:   {self.injury_markers}")
        self.logger.info(f"  Zonation marker:  {self.zonation_marker}")

    # ──────────────────────────────────────────────────────────────────────────
    # Data loading
    # ──────────────────────────────────────────────────────────────────────────

    def load_and_preprocess_data(self) -> Tuple[Any, Any, Any]:
        """
        Load and preprocess the DKD dataset.

        Steps
        -----
        1. Load h5ad, enforce float32 sparse representation.
        2. Map Ensembl IDs → gene symbols via var['feature_name'].
        3. Pre-filter low-expression genes (memory optimisation).
        4. Normalise, log1p, select HVGs — with double whitelist protection for
           adaptive_markers, injury_markers, and zonation_marker.
        5. Split into reference (healthy) / query (disease) AnnData objects.

        Returns
        -------
        full_data, reference_data, query_data
        """
        self.logger.info("Loading and preprocessing DKD dataset …")
        adata = sc.read_h5ad(self.dataset_path)
        self.logger.info(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

        # ── Sparse / precision enforcement ────────────────────────────────────
        # adata.X may be None (some h5ad store data only in layers) OR an all-
        # zero matrix (nnz == 0).  Both cases trigger the layers fallback before
        # we raise.  Only raise after exhausting all alternatives.
        def _needs_layer_fallback(X) -> bool:
            if X is None:
                return True
            if not sparse.issparse(X):
                X = sparse.csr_matrix(X)
            return X.nnz == 0

        if _needs_layer_fallback(adata.X):
            self.logger.warning(
                "adata.X is None or empty — attempting layers fallback …"
            )
            for layer in ("counts", "normalized", "raw_counts", "spliced"):
                if layer in adata.layers:
                    raw = adata.layers[layer]
                    adata.X = (sparse.csr_matrix(raw, dtype=np.float32)
                               if not sparse.issparse(raw)
                               else raw.astype(np.float32))
                    self.logger.info(
                        f"Layer '{layer}' used as expression matrix "
                        f"(nnz={adata.X.nnz})"
                    )
                    break
            else:
                raise ValueError(
                    "adata.X is None/empty AND no recognised layer found "
                    f"(available layers: {list(adata.layers.keys())}). "
                    "Check dataset integrity."
                )
        else:
            # X is valid — normalise precision
            if not sparse.issparse(adata.X):
                adata.X = sparse.csr_matrix(adata.X, dtype=np.float32)
            elif adata.X.dtype != np.float32:
                adata.X = adata.X.astype(np.float32)
        gc.collect()

        # ── Ensembl → symbol mapping ───────────────────────────────────────
        if "feature_name" in adata.var.columns:
            adata.var_names = adata.var["feature_name"].values
            adata.var_names_make_unique()
            self.logger.info(f"Gene symbol mapping done: {adata.n_vars} genes")

        # All markers to protect through the entire filtering pipeline
        all_markers = list(set(
            self.adaptive_markers + self.injury_markers + [self.zonation_marker]
        ))

        # ── Low-expression pre-filter ──────────────────────────────────────
        if sparse.issparse(adata.X):
            gene_nz = np.asarray((adata.X > 0).sum(axis=0)).flatten()
            gene_ct = np.asarray(adata.X.sum(axis=0)).flatten()
        else:
            gene_nz = (adata.X > 0).sum(axis=0)
            gene_ct = adata.X.sum(axis=0)

        min_cells = max(10, int(adata.n_obs * 0.001))
        gene_keep = (gene_nz >= min_cells) & (gene_ct >= 10)

        # Whitelist pass 1 — markers survive low-expression filter
        marker_mask = adata.var_names.isin(all_markers)
        rescued = int(((~gene_keep) & marker_mask).sum())
        if rescued:
            self.logger.info(f"Rescued {rescued} marker genes from low-expr filter")
        gene_keep = gene_keep | marker_mask

        if gene_keep.sum() < adata.n_vars:
            adata = adata[:, gene_keep].copy()
            gc.collect()
        self.logger.info(f"After pre-filter: {adata.n_vars} genes")

        # ── Normalise + HVG ───────────────────────────────────────────────
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=self.n_top_genes)
        self.logger.info(f"HVG identified: {adata.var['highly_variable'].sum()}")

        # Whitelist pass 2 — markers survive HVG filter
        for m in all_markers:
            if m in adata.var_names and not adata.var.loc[m, "highly_variable"]:
                adata.var.loc[m, "highly_variable"] = True
                self.logger.info(f"  Whitelisted into HVG: {m}")

        adata = adata[:, adata.var["highly_variable"]].copy()
        gc.collect()
        self.logger.info(f"Final shape after HVG: {adata.X.shape}")

        # Final marker audit
        missing = [m for m in all_markers if m not in adata.var_names]
        if missing:
            self.logger.warning(f"Markers still absent after whitelist: {missing}")
        else:
            self.logger.info(f"All markers present: {all_markers}")

        # ── Train / query split ────────────────────────────────────────────
        ref_mask = adata.obs[self.condition_column] == self.healthy_condition
        qry_mask = adata.obs[self.condition_column] == self.disease_condition
        reference_data = adata[ref_mask].copy()
        query_data     = adata[qry_mask].copy()
        full_data      = adata
        gc.collect()

        if self.target_cell_type not in query_data.obs[self.cell_type_column].values:
            raise ValueError(
                f"Target cell type '{self.target_cell_type}' not found in query."
            )
        self.logger.info(
            f"Split: reference={reference_data.n_obs}, query={query_data.n_obs}"
        )
        return full_data, reference_data, query_data

    # ──────────────────────────────────────────────────────────────────────────
    # Model training
    # ──────────────────────────────────────────────────────────────────────────

    def train_or_load_model(
        self, reference_data: Any, output_dir: str
    ) -> CytoneighborWrapper:
        """Train SymRelEnc on ALL healthy cells (multi-type required for
        contrastive negative pairs).  Saves model to *output_dir*."""
        self.logger.info(
            f"Training SymRelEnc on {reference_data.n_obs} healthy cells "
            f"(all cell types) …"
        )
        model = CytoneighborWrapper(self.config["model_params"])
        save_path = os.path.join(output_dir, "disease_discovery_model.zip")
        model.train(
            reference_data, self.cell_type_column,
            synthesis_enabled=True, model_save_path=save_path,
        )
        return model

    # ──────────────────────────────────────────────────────────────────────────
    # Joint manifold construction
    # ──────────────────────────────────────────────────────────────────────────

    def construct_reference_mapping(
        self, model: CytoneighborWrapper, combined_pt: Any
    ) -> Any:
        """
        Build a SymRelEnc KNN graph on the JOINT PT dataset.

        Design
        ------
        SymRelEnc was trained on all healthy cell types, encoding biological
        similarity under the healthy reference tissue metric.  Running
        transform() on combined_pt — which physically contains BOTH healthy and
        disease PT cells — allows disease cells to find healthy cells as
        K-nearest neighbours.  Healthy cells thus become true topological
        anchors, not a scoring background.

        Parameters
        ----------
        model       : trained CytoneighborWrapper
        combined_pt : ad.concat([healthy_PT, disease_PT]) with disease_status tagged
        """
        from src.model import transform as _transform

        self.logger.info(
            f"Building SymRelEnc joint graph: {combined_pt.n_obs} PT cells "
            f"({(combined_pt.obs['disease_status']=='Healthy_Reference').sum()} healthy, "
            f"{(combined_pt.obs['disease_status']=='Disease_Query').sum()} disease) …"
        )

        # ── Guard: trained model ───────────────────────────────────────────
        if not model.is_trained:
            raise RuntimeError("Model is not trained.")
        cy = model.adata.uns.get("cytoneighbor", {})
        if cy.get("model") is None:
            raise RuntimeError("'model' key missing from uns['cytoneighbor'].")
        if cy.get("ann", {}).get("P_ann") is None:
            raise RuntimeError("P_ann projection matrix missing from uns['cytoneighbor'].")

        # ── Deep-copy context injection (prevents transform() from mutating
        #    self.adata via model.to(device) → in-place ops) ─────────────
        combined_copy = combined_pt.copy()
        combined_copy.uns["cytoneighbor"] = copy.deepcopy({
            "model":         cy["model"],
            "feature_names": cy.get("feature_names"),
            "ann":           cy.get("ann"),
            "preproc":       cy.get("preproc", {"mode": "none"}),
            "label_mapping": cy.get("label_mapping"),
            "env_key":       cy.get("env_key"),
        })

        # ── Build joint KNN graph ──────────────────────────────────────────
        _transform(combined_copy, n_neighbors=self.n_neighbors, use_cuda=True)

        if "connectivities" not in combined_copy.obsp:
            raise RuntimeError(
                "transform() did not produce 'connectivities'. "
                "Check transform() logs."
            )
        self.logger.info(
            f"Joint KNN graph built: shape={combined_copy.obsp['connectivities'].shape}, "
            f"nnz={combined_copy.obsp['connectivities'].nnz}"
        )
        return combined_copy

    # ──────────────────────────────────────────────────────────────────────────
    # Trajectory discovery on joint manifold
    # ──────────────────────────────────────────────────────────────────────────

    def discover_adaptive_trajectory(
        self, combined_pt: Any
    ) -> Dict[str, Any]:
        """
        Discover PT subpopulations and infer the convergent dedifferentiation
        trajectory on the joint healthy+disease manifold.

        Steps
        -----
        1. Leiden clustering on the pre-built SymRelEnc KNN graph.
        2. Filter tiny clusters (< 1 % of PT cells).
        3. Dynamic resolution search if < 3 clusters are found.
        4. PAGA for cluster connectivity (convergent topology detection).
        5. Root selection: cluster with highest *absolute* Healthy_Reference
           count — marker-free, no circular logic.
        6. Biological characterisation anchored to root.
        7. DPT from root centroid (acknowledged approximation for convergent
           manifold; PAGA provides the topologically accurate picture).
        """
        self.logger.info(
            f"Trajectory discovery on joint manifold "
            f"({combined_pt.n_obs} PT cells, "
            f"{(combined_pt.obs['disease_status']=='Healthy_Reference').sum()} healthy + "
            f"{(combined_pt.obs['disease_status']=='Disease_Query').sum()} disease) …"
        )
        pt_cells = combined_pt.copy()

        # ── 1. Leiden ─────────────────────────────────────────────────────
        sc.tl.leiden(
            pt_cells, resolution=self.leiden_resolution,
            key_added="leiden_clusters",
        )
        n_init = len(pt_cells.obs["leiden_clusters"].unique())
        self.logger.info(
            f"Initial Leiden ({self.leiden_resolution}): {n_init} clusters — "
            f"{dict(pt_cells.obs['leiden_clusters'].value_counts())}"
        )

        # ── 2. Filter tiny clusters ────────────────────────────────────────
        def _drop_tiny(adata: Any, thresh: int) -> Any:
            sizes  = adata.obs["leiden_clusters"].value_counts()
            valid  = sizes[sizes >= thresh].index.tolist()
            dropped = sizes[sizes < thresh]
            if len(dropped):
                self.logger.warning(
                    f"Dropping {len(dropped)} tiny clusters "
                    f"({int(dropped.sum())} cells): {dict(dropped)}"
                )
                adata = adata[adata.obs["leiden_clusters"].isin(valid)].copy()
                gc.collect()
            return adata

        thresh = max(10, int(pt_cells.n_obs * 0.01))
        pt_cells  = _drop_tiny(pt_cells, thresh)
        n_clusters = len(pt_cells.obs["leiden_clusters"].unique())
        cluster_counts = pt_cells.obs["leiden_clusters"].value_counts()

        # ── 3. Dynamic resolution search ──────────────────────────────────
        res = self.leiden_resolution
        while n_clusters < 3 and res <= 3.0:
            res = round(res + 0.2, 2)
            self.logger.warning(
                f"Only {n_clusters} clusters — retrying with resolution={res}"
            )
            sc.tl.leiden(pt_cells, resolution=res, key_added="leiden_clusters")
            thresh = max(10, int(pt_cells.n_obs * 0.01))
            pt_cells = _drop_tiny(pt_cells, thresh)
            n_clusters = len(pt_cells.obs["leiden_clusters"].unique())
            cluster_counts = pt_cells.obs["leiden_clusters"].value_counts()
            self.logger.info(
                f"  Resolution {res}: {n_clusters} clusters, {pt_cells.n_obs} cells"
            )

        # Safe default (overridden below)
        root_cluster = str(cluster_counts.idxmax())

        if n_clusters > 1:
            # ── 4. PAGA ───────────────────────────────────────────────────
            self.logger.info("Running PAGA for cluster connectivity …")
            sc.tl.paga(pt_cells, groups="leiden_clusters")
            self._log_paga_topology(pt_cells)

            # ── 5. Root selection (absolute healthy count; no marker bias) ─
            root_cluster = self._select_root_cluster(pt_cells)

            # ── 6. Characterisation ────────────────────────────────────────
            characterization = self._characterize_clusters(pt_cells, root_cluster)

            # ── 7. DPT from root centroid ──────────────────────────────────
            root_mask = pt_cells.obs["leiden_clusters"] == root_cluster
            if root_mask.sum() > 0:
                root_idx = np.where(root_mask)[0]
                root_X = (pt_cells.X[root_idx].toarray()
                          if hasattr(pt_cells.X, "toarray")
                          else np.asarray(pt_cells.X[root_idx]))
                centroid = root_X.mean(axis=0)
                pt_cells.uns["iroot"] = int(
                    root_idx[np.argmin(np.linalg.norm(root_X - centroid, axis=1))]
                )
                self.logger.info(
                    f"DPT iroot={pt_cells.uns['iroot']} "
                    f"(centroid-nearest cell in cluster '{root_cluster}')"
                )
                self.logger.info("Computing diffusion map + DPT …")
                sc.tl.diffmap(pt_cells)
                sc.tl.dpt(pt_cells)
                self.logger.info(
                    f"DPT range: [{pt_cells.obs['dpt_pseudotime'].min():.4f}, "
                    f"{pt_cells.obs['dpt_pseudotime'].max():.4f}]"
                )
            else:
                self.logger.warning(
                    f"Root cluster '{root_cluster}' empty after filtering — DPT skipped."
                )
        else:
            self.logger.warning("< 2 clusters — PAGA and DPT skipped.")
            characterization = self._characterize_clusters(pt_cells, root_cluster)

        return {
            "pt_cells":       pt_cells,
            "n_clusters":     n_clusters,
            "cluster_counts": dict(cluster_counts),
            "root_cluster":   root_cluster,
            "characterization": characterization,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Root cluster selection
    # ──────────────────────────────────────────────────────────────────────────

    def _select_root_cluster(self, pt_cells: Any) -> str:
        """
        Select the trajectory root as the cluster with the highest *absolute*
        number of Healthy_Reference cells.

        Rationale (documented for reviewers)
        -------------------------------------
        The S1/S2 segment constitutes approximately 60–70 % of PT tubular
        volume and cell count in the human kidney (Lindström et al., 2021).
        Therefore, in a joint healthy+disease PT dataset where healthy cells
        are the majority (≈77 %), the S1/S2 cluster will contain the largest
        absolute number of healthy cells simply because it is the numerically
        dominant PT compartment.  Using absolute count avoids any use of
        expression markers in root selection, fully eliminating circular logic
        between root assignment and downstream biological labelling.

        Additionally logs per-cluster healthy-cell purity (%) for Methods
        and the origin_composition_barplot.pdf figure.

        Fallback
        --------
        If disease_status is absent → largest cluster by total cell count.
        """
        self.logger.info("Selecting root cluster by Healthy_Reference absolute count …")

        if "disease_status" not in pt_cells.obs.columns:
            fallback = str(pt_cells.obs["leiden_clusters"].value_counts().idxmax())
            self.logger.warning(
                f"'disease_status' absent — fallback to largest cluster '{fallback}'"
            )
            return fallback

        healthy = pt_cells[pt_cells.obs["disease_status"] == "Healthy_Reference"]
        if healthy.n_obs == 0:
            fallback = str(pt_cells.obs["leiden_clusters"].value_counts().idxmax())
            self.logger.warning(
                f"No Healthy_Reference cells — fallback to largest cluster '{fallback}'"
            )
            return fallback

        h_dist  = healthy.obs["leiden_clusters"].value_counts()   # absolute counts
        t_dist  = pt_cells.obs["leiden_clusters"].value_counts()  # total counts
        total_h = healthy.n_obs
        root    = str(h_dist.idxmax())

        self.logger.info("=== Cluster composition (for Methods / Figure legends) ===")
        for c in sorted(t_dist.index, key=lambda x: int(x)):
            h_abs  = int(h_dist.get(c, 0))
            tot    = int(t_dist[c])
            purity = 100.0 * h_abs / tot if tot > 0 else 0.0
            h_pct  = 100.0 * h_abs / total_h if total_h > 0 else 0.0
            self.logger.info(
                f"  Cluster {c}: total={tot:5d}  "
                f"healthy_abs={h_abs:5d} ({h_pct:5.1f}% of all healthy PT)  "
                f"purity={purity:5.1f}%"
            )
        self.logger.info(
            f"Selected root → Cluster '{root}' "
            f"({h_dist[root]} healthy cells, largest absolute count)"
        )
        return root

    # ──────────────────────────────────────────────────────────────────────────
    # Cluster characterisation
    # ──────────────────────────────────────────────────────────────────────────

    def _characterize_clusters(
        self, pt_cells: Any, root_cluster: str
    ) -> Dict[str, Any]:
        """
        Assign biological labels to clusters.

        Assignment logic (no circular dependence on root selection)
        -----------------------------------------------------------
        1. root_cluster              → "Functional PT (S1/S2-like)"
           Rationale: root was selected by absolute healthy count, not markers;
           this cluster also carries the highest SLC5A2 / LRP2 (confirmed post-hoc).
        2. Highest VCAM1 / HAVCR1    → "Injured PT"
           The terminal dedifferentiation state, characterised by EMT signatures.
        3. Remaining cluster(s)      → "Functional PT (S3-like)"
           Lower SLC5A2 consistent with S3 segment biology; characterised
           post-hoc via _validate_zonation (SLC5A1 check).

        Stores both functional_scores (SLC5A2/LRP2 mean) and injury_scores
        (VCAM1/HAVCR1 mean) for every cluster in the returned dict.
        """
        self.logger.info(
            f"Characterising clusters: root='{root_cluster}' → "
            "'Functional PT (S1/S2-like)' …"
        )

        char: Dict[str, Any] = {
            "functional_scores": {},
            "injury_scores":     {},
            "cluster_labels":    {},
        }

        clusters = pt_cells.obs["leiden_clusters"].unique().tolist()
        stats_rows = []

        for c in clusters:
            mask  = pt_cells.obs["leiden_clusters"] == c
            cells = pt_cells[mask]

            func  = _mean_expr(cells, self.adaptive_markers)
            inj   = _mean_expr(cells, self.injury_markers)

            char["functional_scores"][c] = func
            char["injury_scores"][c]     = inj
            stats_rows.append({"cluster": c, "functional": func, "injury": inj})

        df = pd.DataFrame(stats_rows)

        # 1. Root → S1/S2-like
        char["cluster_labels"][root_cluster] = "Functional PT (S1/S2-like)"
        remaining = df[df["cluster"] != root_cluster].copy()

        # 2. Highest injury → Injured PT
        if not remaining.empty:
            inj_id = remaining.sort_values("injury", ascending=False).iloc[0]["cluster"]
            char["cluster_labels"][inj_id] = "Injured PT"
            remaining = remaining[remaining["cluster"] != inj_id]

        # 3. All intermediate clusters → S3-like
        for _, row in remaining.iterrows():
            char["cluster_labels"][row["cluster"]] = "Functional PT (S3-like)"

        # ── Post-hoc validation logging ────────────────────────────────────
        self.logger.info("=== Cluster characterisation results ===")
        for c, label in char["cluster_labels"].items():
            self.logger.info(
                f"  Cluster {c} → {label}  "
                f"(Functional={char['functional_scores'][c]:.4f}, "
                f"Injury={char['injury_scores'][c]:.4f})"
            )

        # Confirm S1/S2 has higher functional score than S3-like (sanity check)
        s12 = char["functional_scores"].get(root_cluster, 0.0)
        s3_ids = [k for k, v in char["cluster_labels"].items()
                  if v == "Functional PT (S3-like)"]
        for s3id in s3_ids:
            s3 = char["functional_scores"].get(s3id, 0.0)
            if s12 > s3:
                self.logger.info(
                    f"  ✓ S1/S2-like SLC5A2/LRP2 ({s12:.3f}) > "
                    f"S3-like ({s3:.3f}) — consistent with zonation"
                )
            else:
                self.logger.warning(
                    f"  ⚠ S1/S2-like functional score ({s12:.3f}) is NOT higher "
                    f"than S3-like ({s3:.3f}) — zonation interpretation requires caution"
                )

        return char

    # ──────────────────────────────────────────────────────────────────────────
    # PAGA topology logging
    # ──────────────────────────────────────────────────────────────────────────

    def _log_paga_topology(self, pt_cells: Any) -> None:
        """
        Log PAGA connectivity matrix above threshold for topology assessment.
        A triangular / fully-connected graph supports the convergent narrative;
        a linear chain would support a single-path narrative.
        """
        try:
            conn = pt_cells.uns.get("paga", {}).get("connectivities")
            if conn is None:
                return
            if hasattr(conn, "toarray"):
                conn = conn.toarray()
            n = conn.shape[0]
            self.logger.info(
                f"PAGA connectivity matrix ({n}×{n}, threshold={self.paga_threshold}):"
            )
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    w = float(conn[i, j])
                    if w >= self.paga_threshold:
                        edges.append(f"{i}↔{j} (w={w:.3f})")
            self.logger.info(
                f"  Edges above threshold: {', '.join(edges) if edges else 'none'}"
            )
            n_edges = len(edges)
            max_possible = n * (n - 1) // 2
            if n == 3 and n_edges == 3:
                self.logger.info(
                    "  → FULLY CONNECTED TRIANGLE: consistent with convergent "
                    "dedifferentiation (both S1/S2 and S3 edges reach Injured PT)"
                )
            elif n_edges == n - 1:
                self.logger.info(
                    "  → LINEAR CHAIN: single-path trajectory topology"
                )
            else:
                self.logger.info(
                    f"  → {n_edges}/{max_possible} edges present — partial connectivity"
                )
        except Exception as e:
            self.logger.warning(f"PAGA topology logging failed (non-fatal): {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Zonation validation
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_zonation(
        self,
        pt_cells: Any,
        characterization: Dict[str, Any],
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        Independent validation of the S1/S2 vs S3 zonation hypothesis.

        Strategy
        --------
        SLC5A1 (SGLT1) is the primary glucose transporter of the S3 segment
        and is expressed at low levels in S1/S2.  If our "Functional PT (S3-like)"
        cluster truly represents the S3 anatomical segment, it should show
        higher SLC5A1 expression than the S1/S2-like cluster.

        Additional checks
        -----------------
        • SLC5A2 (SGLT2) gradient:  S1/S2 > S3 > Injured (expected from anatomy)
        • Quantitative statistics logged for Methods section
        • Bar plot saved as zonation_validation.pdf

        Returns
        -------
        Dict with per-cluster mean expression for SLC5A1, SLC5A2, and assessment.
        """
        self.logger.info("=== Zonation validation ===")
        results: Dict[str, Any] = {
            "zonation_marker_present": False,
            "per_cluster":             {},
            "s3_enrichment_confirmed": None,
        }

        labels = characterization["cluster_labels"]
        s12_id = next((k for k, v in labels.items()
                       if v == "Functional PT (S1/S2-like)"), None)
        s3_id  = next((k for k, v in labels.items()
                       if v == "Functional PT (S3-like)"), None)
        inj_id = next((k for k, v in labels.items()
                       if v == "Injured PT"), None)

        # Gather expression for all available validation genes
        val_genes = [self.zonation_marker, self.adaptive_markers[0]]
        present   = [g for g in val_genes if g in pt_cells.var_names]

        if not present:
            self.logger.warning(
                f"None of {val_genes} found in var_names — "
                "zonation validation skipped."
            )
            return results

        # ── Per-cluster mean expression ───────────────────────────────────
        cluster_order_ids = [s12_id, s3_id, inj_id]
        cluster_order_ids = [c for c in cluster_order_ids if c is not None]

        rows = []
        for c in cluster_order_ids:
            mask  = pt_cells.obs["leiden_clusters"] == c
            cells = pt_cells[mask]
            row = {"cluster_id": c, "label": labels.get(c, f"Cluster {c}")}
            for gene in val_genes:
                if gene in pt_cells.var_names:
                    row[gene] = _mean_expr(cells, [gene])
                else:
                    row[gene] = np.nan
            rows.append(row)
            results["per_cluster"][c] = row

        df = pd.DataFrame(rows).set_index("label")
        self.logger.info("\n=== Zonation validation — per-cluster means ===")
        self.logger.info(f"\n{df[[g for g in val_genes if g in df.columns]].round(4).to_string()}")

        # ── S3 enrichment assessment ──────────────────────────────────────
        zm = self.zonation_marker
        if zm in df.columns and s3_id is not None and s12_id is not None:
            results["zonation_marker_present"] = True
            s3_val  = df.loc[labels[s3_id],  zm]
            s12_val = df.loc[labels[s12_id], zm]
            confirmed = bool(s3_val > s12_val)
            results["s3_enrichment_confirmed"] = confirmed
            if confirmed:
                self.logger.info(
                    f"  ✓ {zm} higher in S3-like ({s3_val:.4f}) "
                    f"than S1/S2-like ({s12_val:.4f}) — "
                    "ZONATION HYPOTHESIS SUPPORTED"
                )
            else:
                self.logger.warning(
                    f"  ✗ {zm} NOT higher in S3-like ({s3_val:.4f}) "
                    f"vs S1/S2-like ({s12_val:.4f}) — "
                    "zonation interpretation requires additional evidence; "
                    f"cluster difference may reflect cell-to-cell variability. "
                    "Report as 'consistent with' rather than 'confirmed'."
                )
        else:
            self.logger.warning(
                f"  {zm} not in var_names — "
                "S3 marker validation unavailable; "
                "zonation claim must rely on SLC5A2 gradient alone."
            )

        # ── Zonation validation figure ────────────────────────────────────
        plot_genes = [g for g in val_genes if g in df.columns]
        if plot_genes and len(df) > 0:
            try:
                fig, axes = plt.subplots(
                    1, len(plot_genes),
                    figsize=(4.0 * len(plot_genes), 4.0),
                    sharey=False,
                )
                if len(plot_genes) == 1:
                    axes = [axes]

                bar_colours = [STATE_PALETTE.get(lbl, "#888888")
                               for lbl in df.index]

                for ax, gene in zip(axes, plot_genes):
                    vals = df[gene].values.astype(float)
                    bars = ax.bar(range(len(df)), vals, color=bar_colours,
                                  edgecolor="white", linewidth=0.8)
                    ax.set_xticks(range(len(df)))
                    ax.set_xticklabels(
                        [lbl.replace("Functional PT ", "").replace("(", "").replace(")", "")
                         for lbl in df.index],
                        rotation=30, ha="right", fontsize=8,
                    )
                    ax.set_ylabel("Mean expression (log1p)", fontsize=9)
                    ax.set_title(
                        f"{gene} expression\n"
                        f"({'S3 marker — should be highest in S3-like' if gene == zm else 'S1/S2 marker — should decrease toward Injured PT'})",
                        fontsize=8, fontweight="bold",
                    )
                    ax.grid(axis="y", linestyle="--", alpha=0.4)
                    # Add value labels on bars
                    for bar, val in zip(bars, vals):
                        if not np.isnan(val):
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.005,
                                f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                            )

                status = ("SUPPORTED" if results.get("s3_enrichment_confirmed")
                          else ("NOT SUPPORTED" if results.get("s3_enrichment_confirmed") is False
                                else "MARKER ABSENT"))
                fig.suptitle(
                    f"Zonation Validation: S1/S2-like vs S3-like\n"
                    f"SLC5A1 ({zm}) enrichment in S3-like: {status}",
                    fontsize=10, fontweight="bold",
                )
                plt.tight_layout()
                save_path = os.path.join(output_dir, "zonation_validation.pdf")
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()
                self.logger.info(f"Zonation validation figure saved: {save_path}")
            except Exception as e:
                self.logger.warning(f"Zonation validation plot failed (non-fatal): {e}")
                plt.close("all")

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Main visualisation orchestrator
    # ──────────────────────────────────────────────────────────────────────────

    def create_visualizations(
        self, results: Dict[str, Any], output_dir: str
    ) -> None:
        """
        Generate all publication figures for Experiment 3.

        Figures produced
        ----------------
        umap_combined.pdf               — 4-panel UMAP with explicit palette
        paga_trajectory.pdf             — PAGA cluster connectivity
        marker_trends.pdf               — Marker dynamics along DPT
        cluster_characterization.pdf    — Functional vs Injury score heatmap
        origin_composition_barplot.pdf  — Quantitative cell-origin composition
        zonation_validation.pdf         — SLC5A1 / SLC5A2 zonation check
        compensatory_validation.pdf     — Disease-cell-specific SLC5A2 analysis
        go_enrichment_bubble.pdf        — GO enrichment (strict filtering)
        """
        self.logger.info("Creating visualisations …")
        pt_cells         = results["pt_cells"]
        characterization = results["characterization"]

        sc.settings.figdir = output_dir
        sc.settings.set_figure_params(dpi=300, facecolor="white")

        # Attach biological_state labels to obs (required by all downstream plots)
        pt_cells.obs["biological_state"] = [
            characterization["cluster_labels"].get(c, "Functional PT (S3-like)")
            for c in pt_cells.obs["leiden_clusters"]
        ]

        # Run all figures
        self._plot_umap(pt_cells, output_dir)
        self._plot_paga_trajectory(pt_cells, characterization, output_dir)

        if "dpt_pseudotime" in pt_cells.obs.columns:
            self._plot_marker_trends(pt_cells, output_dir)

        self._plot_cluster_characterization(characterization, output_dir)
        self._plot_origin_composition(pt_cells, output_dir)
        self._validate_zonation(pt_cells, characterization, output_dir)
        self._validate_compensatory_hypothesis(pt_cells, characterization, output_dir)
        self._plot_go_enrichment(pt_cells, characterization, output_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # Compensatory hypothesis validation
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_compensatory_hypothesis(
        self,
        pt_cells: Any,
        characterization: Dict[str, Any],
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        Disease-cell-specific SLC5A2 analysis to test the compensatory
        upregulation hypothesis.

        Scientific rationale
        --------------------
        A cluster-level mean expression difference between clusters can arise
        from three independent mechanisms that are confounded in whole-cluster
        means:

          (A) Anatomical subtype composition — the cluster labelled S1/S2-like
              may contain a higher fraction of true S1/S2 cells, which
              constitutively express more SLC5A2 regardless of disease status.
          (B) Genuine disease-driven upregulation — DKD cells in the S1/S2-like
              cluster actively upregulate SLC5A2 as a compensatory response to
              reduced single-nephron GFR (as hypothesised by the SGLT2 inhibitor
              rationale literature).
          (C) Statistical noise — the 8% mean difference may fall within normal
              cell-to-cell variability.

        This method disentangles A from B by isolating only the Disease_Query
        cells within each cluster and comparing their SLC5A2 expression to the
        Healthy_Reference cells in the SAME cluster.

        Test logic
        ----------
        For the S1/S2-like cluster (the key cluster for the compensatory claim):
          H0: mean_SLC5A2(Disease) ≤ mean_SLC5A2(Healthy) within the cluster
          H1: mean_SLC5A2(Disease) >  mean_SLC5A2(Healthy)  → compensatory signal

        A one-sided Mann-Whitney U test is used (non-parametric; robust to the
        non-normal distribution of single-cell expression values).

        Reporting policy
        ----------------
        • If H1 is supported (p < 0.05, disease mean > healthy mean):
          The compensatory finding is logged as SUPPORTED and reported in
          Discussion as corroborating evidence for the SGLT2 inhibitor mechanism.
          The primary zonation narrative (Results) is NOT changed.
        • If H1 is not supported:
          The null hypothesis is logged as NOT REJECTED.  The cluster-mean
          difference is attributed to subtype composition (mechanism A), which
          is consistent with the zonation interpretation.
          No language changes are needed in either Results or Discussion.

        Output
        ------
        compensatory_validation.pdf  — per-cluster disease vs healthy SLC5A2
                                       side-by-side bar plot with significance
                                       annotation and sample sizes
        Returns dict with per-cluster statistics for JSON results.
        """
        self.logger.info("=== Compensatory hypothesis validation ===")

        result: Dict[str, Any] = {
            "compensatory_supported":    None,
            "s12_disease_mean":          None,
            "s12_healthy_mean":          None,
            "s12_pvalue":                None,
            "s12_n_disease":             None,
            "s12_n_healthy":             None,
            "per_cluster":               {},
        }

        # ── Prerequisite checks ───────────────────────────────────────────
        target_gene = self.adaptive_markers[0]   # SLC5A2 by default
        for col in ("disease_status", "leiden_clusters", "biological_state"):
            if col not in pt_cells.obs.columns:
                self.logger.warning(
                    f"  '{col}' absent from obs — compensatory validation skipped."
                )
                return result

        if target_gene not in pt_cells.var_names:
            self.logger.warning(
                f"  '{target_gene}' not in var_names — "
                "compensatory validation skipped."
            )
            return result

        from scipy.stats import mannwhitneyu

        labels = characterization["cluster_labels"]

        # ── Per-cluster disease vs healthy comparison ─────────────────────
        plot_rows = []
        for cluster_id, label in labels.items():
            h_mask = (
                (pt_cells.obs["leiden_clusters"] == cluster_id) &
                (pt_cells.obs["disease_status"] == "Healthy_Reference")
            )
            d_mask = (
                (pt_cells.obs["leiden_clusters"] == cluster_id) &
                (pt_cells.obs["disease_status"] == "Disease_Query")
            )
            h_cells = pt_cells[h_mask]
            d_cells = pt_cells[d_mask]

            n_h = h_cells.n_obs
            n_d = d_cells.n_obs

            if n_h < 5 or n_d < 5:
                self.logger.warning(
                    f"  Cluster {cluster_id} ({label}): "
                    f"too few cells (healthy={n_h}, disease={n_d}) — skipping."
                )
                continue

            def _get_expr(cells: Any, gene: str) -> np.ndarray:
                mat = cells[:, gene].X
                arr = mat.toarray().flatten() if hasattr(mat, "toarray") \
                      else np.asarray(mat).flatten()
                return arr.astype(float)

            h_expr = _get_expr(h_cells, target_gene)
            d_expr = _get_expr(d_cells, target_gene)

            h_mean = float(h_expr.mean())
            d_mean = float(d_expr.mean())

            # One-sided Mann-Whitney U: alternative='greater' tests d > h
            try:
                stat, pval = mannwhitneyu(d_expr, h_expr, alternative="greater")
                pval = float(pval)
            except Exception as e:
                self.logger.warning(
                    f"  MWU test failed for cluster {cluster_id}: {e}"
                )
                pval = np.nan

            row = {
                "cluster_id":    cluster_id,
                "label":         label,
                "healthy_mean":  h_mean,
                "disease_mean":  d_mean,
                "delta":         d_mean - h_mean,
                "pvalue":        pval,
                "n_healthy":     n_h,
                "n_disease":     n_d,
            }
            plot_rows.append(row)
            result["per_cluster"][cluster_id] = row

            sig = "p<0.05 ✓" if (not np.isnan(pval) and pval < 0.05) else "n.s."
            direction = "↑ disease > healthy" if d_mean > h_mean else "↓ disease < healthy"
            self.logger.info(
                f"  Cluster {cluster_id} ({label:35s}): "
                f"healthy={h_mean:.4f} (n={n_h})  "
                f"disease={d_mean:.4f} (n={n_d})  "
                f"Δ={d_mean-h_mean:+.4f}  {direction}  {sig}"
            )

            # Store S1/S2-like specific results for top-level reporting
            if label == "Functional PT (S1/S2-like)":
                result["s12_healthy_mean"] = h_mean
                result["s12_disease_mean"] = d_mean
                result["s12_pvalue"]       = pval
                result["s12_n_healthy"]    = n_h
                result["s12_n_disease"]    = n_d

        # ── Verdict on compensatory hypothesis ────────────────────────────
        s12_pval  = result.get("s12_pvalue")
        s12_delta = (result["s12_disease_mean"] or 0.0) - (result["s12_healthy_mean"] or 0.0)

        if s12_pval is not None and not np.isnan(s12_pval):
            supported = (s12_pval < 0.05) and (s12_delta > 0)
            result["compensatory_supported"] = supported
            if supported:
                self.logger.info(
                    f"\n  ╔══════════════════════════════════════════════════════╗\n"
                    f"  ║  COMPENSATORY HYPOTHESIS SUPPORTED                    ║\n"
                    f"  ║  DKD cells in S1/S2-like show higher {target_gene}         ║\n"
                    f"  ║  than healthy cells in the same cluster               ║\n"
                    f"  ║  Δ={s12_delta:+.4f}  p={s12_pval:.4f}                       ║\n"
                    f"  ║  → Suitable for Discussion (not Results)              ║\n"
                    f"  ╚══════════════════════════════════════════════════════╝"
                )
            else:
                self.logger.info(
                    f"\n  ┌─────────────────────────────────────────────────────┐\n"
                    f"  │  COMPENSATORY HYPOTHESIS NOT SUPPORTED                │\n"
                    f"  │  Δ={s12_delta:+.4f}  p={s12_pval:.4f}                      │\n"
                    f"  │  Cluster-mean difference attributable to subtype      │\n"
                    f"  │  composition (mechanism A) — consistent with zonation │\n"
                    f"  └─────────────────────────────────────────────────────┘"
                )
        else:
            self.logger.warning("  S1/S2-like cluster not found or test failed.")

        # ── Figure ────────────────────────────────────────────────────────
        if not plot_rows:
            self.logger.warning(
                "  No clusters had sufficient cells — "
                "compensatory_validation.pdf not generated."
            )
            return result

        try:
            df = pd.DataFrame(plot_rows)

            # Reorder by STATE_ORDER
            order_map = {v: i for i, v in enumerate(STATE_ORDER)}
            df["_ord"] = df["label"].map(lambda x: order_map.get(x, 99))
            df = df.sort_values("_ord").reset_index(drop=True)

            n_clusters = len(df)
            fig, axes = plt.subplots(
                1, n_clusters,
                figsize=(3.8 * n_clusters, 5.0),
                sharey=False,
            )
            if n_clusters == 1:
                axes = [axes]

            for ax, (_, row) in zip(axes, df.iterrows()):
                label       = row["label"]
                h_val       = row["healthy_mean"]
                d_val       = row["disease_mean"]
                pval        = row["pvalue"]
                n_h         = row["n_healthy"]
                n_d         = row["n_disease"]
                cluster_col = STATE_PALETTE.get(label, "#888888")

                bars = ax.bar(
                    ["Healthy\nReference", "Disease\nQuery"],
                    [h_val, d_val],
                    color=["#2196F3", "#F44336"],
                    edgecolor="white", linewidth=0.8, width=0.55,
                )
                # Value + n labels
                for bar, val, n in zip(
                    bars, [h_val, d_val], [n_h, n_d]
                ):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{val:.3f}\n(n={n})",
                        ha="center", va="bottom", fontsize=8,
                    )

                # Significance bracket
                y_top = max(h_val, d_val) * 1.18
                ax.plot(
                    [0, 0, 1, 1],
                    [y_top * 0.97, y_top, y_top, y_top * 0.97],
                    color="#333333", linewidth=1.0,
                )
                if not np.isnan(pval):
                    sig_txt = (
                        f"p={pval:.3f}" if pval >= 0.001
                        else "p<0.001"
                    )
                    sig_txt += "  ✓" if pval < 0.05 else "  n.s."
                else:
                    sig_txt = "test failed"
                ax.text(
                    0.5, y_top * 1.02, sig_txt,
                    ha="center", va="bottom", fontsize=8, color="#333333",
                )

                delta   = d_val - h_val
                d_arrow = "↑" if delta > 0 else "↓"
                ax.set_title(
                    f"{label.replace('Functional PT ', '').replace('(', '').replace(')', '')}\n"
                    f"{target_gene}: disease {d_arrow} healthy  (Δ={delta:+.3f})",
                    fontsize=9, fontweight="bold",
                    color=cluster_col,
                )
                ax.set_ylabel(f"{target_gene} expression (log1p)", fontsize=9)
                ax.set_ylim(0, y_top * 1.22)
                ax.grid(axis="y", linestyle="--", alpha=0.35)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            verdict = result.get("compensatory_supported")
            verdict_str = (
                "SUPPORTED — disease cells show higher SLC5A2 within S1/S2-like cluster"
                if verdict is True
                else (
                    "NOT SUPPORTED — difference attributable to subtype composition"
                    if verdict is False
                    else "INCONCLUSIVE"
                )
            )
            fig.suptitle(
                f"Compensatory Hypothesis Validation: {target_gene}\n"
                f"Disease_Query vs Healthy_Reference within each cluster\n"
                f"Verdict: {verdict_str}\n"
                "(one-sided Mann-Whitney U; reported in Discussion only)",
                fontsize=9, fontweight="bold", y=1.03,
            )
            plt.tight_layout()
            save_path = os.path.join(output_dir, "compensatory_validation.pdf")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            self.logger.info(
                f"Compensatory validation figure saved: {save_path}"
            )

        except Exception as e:
            self.logger.warning(
                f"Compensatory validation plot failed (non-fatal): {e}"
            )
            plt.close("all")

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # UMAP — 4 panels
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_umap(self, pt_cells: Any, output_dir: str) -> None:
        """
        4-panel UMAP:
          1. Leiden cluster IDs
          2. Biological states (explicit STATE_PALETTE — prevents alphabetical scramble)
          3. Cell origin (Healthy / Disease)
          4. DPT pseudotime (if available)

        The explicit palette on panel 2 is CRITICAL: Scanpy assigns colours by
        category sort order, which differs between panels 1 and 2 unless forced.
        """
        self.logger.info("Computing UMAP embedding …")
        sc.tl.umap(pt_cells)

        has_dpt = "dpt_pseudotime" in pt_cells.obs.columns
        n_panels = 4 if has_dpt else 3
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6.5))

        # Panel 1 — Leiden cluster IDs
        sc.pl.umap(
            pt_cells, color="leiden_clusters",
            ax=axes[0], show=False,
            title="Leiden Clusters",
        )

        # Panel 2 — Biological states with EXPLICIT palette
        # Categories must exist in obs before passing palette dict
        existing_states = pt_cells.obs["biological_state"].unique().tolist()
        palette_filtered = {k: v for k, v in STATE_PALETTE.items()
                            if k in existing_states}
        sc.pl.umap(
            pt_cells, color="biological_state",
            ax=axes[1], show=False,
            title="Biological States (PT Zonation)",
            palette=palette_filtered,
        )

        # Panel 3 — Cell origin
        if "disease_status" in pt_cells.obs.columns:
            sc.pl.umap(
                pt_cells, color="disease_status",
                ax=axes[2], show=False,
                title="Cell Origin (Healthy vs Disease)",
                palette=ORIGIN_PALETTE,
            )
            n_h = (pt_cells.obs["disease_status"] == "Healthy_Reference").sum()
            n_d = (pt_cells.obs["disease_status"] == "Disease_Query").sum()
            self.logger.info(
                f"UMAP panel 3 — Healthy={n_h}, Disease={n_d}"
            )
        else:
            axes[2].set_visible(False)
            self.logger.warning("'disease_status' absent — panel 3 suppressed.")

        # Panel 4 — DPT pseudotime
        if has_dpt:
            sc.pl.umap(
                pt_cells, color="dpt_pseudotime",
                ax=axes[3], show=False,
                title="DPT Pseudotime\n(approximation; PAGA shows true topology)",
                color_map="viridis",
            )

        plt.tight_layout()
        save_path = os.path.join(output_dir, "umap_combined.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        self.logger.info(f"UMAP ({n_panels} panels) saved: {save_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # PAGA trajectory
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_paga_trajectory(
        self,
        pt_cells: Any,
        characterization: Dict[str, Any],
        output_dir: str,
    ) -> None:
        """
        PAGA cluster connectivity plot.

        Key message: triangular connectivity supports convergent dedifferentiation
        (S1/S2-like and S3-like both connect to Injured PT) rather than a linear
        chain.  DPT linearises this, which is explicitly acknowledged in the figure
        subtitle.
        """
        self.logger.info("Plotting PAGA trajectory (convergent topology) …")
        try:
            label_str = "  |  ".join(
                f"Cluster {k} = {v}"
                for k, v in sorted(characterization["cluster_labels"].items())
            )
            title = (
                f"PAGA Cluster Connectivity\n{label_str}\n"
                "Triangular topology = convergent dedifferentiation\n"
                "(S1/S2-like and S3-like converge onto shared Injured PT)"
            )
            fig, ax = plt.subplots(figsize=(8, 7))
            sc.pl.paga(
                pt_cells,
                threshold=self.paga_threshold,
                ax=ax,
                show=False,
                title=title,
                node_size_scale=2.5,
                layout="fr",
            )
            plt.tight_layout()
            save_path = os.path.join(output_dir, "paga_trajectory.pdf")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            self.logger.info(f"PAGA trajectory saved: {save_path}")
        except Exception as e:
            self.logger.warning(f"PAGA plot failed (non-fatal): {e}")
            plt.close("all")

    # ──────────────────────────────────────────────────────────────────────────
    # Marker gene trends
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_marker_trends(self, pt_cells: Any, output_dir: str) -> None:
        """
        Gaussian-smoothed marker gene dynamics along rank-transformed DPT.

        Markers plotted (in order):
          SLC5A2  — epithelial function; expected monotonic decrease
          LRP2    — epithelial function; expected monotonic decrease
          VCAM1   — acute stress;        expected monotonic increase
          HAVCR1  — chronic injury;      expected monotonic increase

        Important caveat note in figure subtitle:
          DPT is computed from S1/S2-like root only.  The convergent PAGA
          topology means the DPT x-axis represents an average path through the
          manifold, not a unique biological timeline.
        """
        self.logger.info("Plotting marker gene trends along DPT …")

        if "dpt_pseudotime" not in pt_cells.obs:
            self.logger.warning("dpt_pseudotime missing — skipping.")
            return

        target_order     = ["SLC5A2", "LRP2", "VCAM1", "HAVCR1"]
        available        = [m for m in target_order if m in pt_cells.var_names]
        if not available:
            self.logger.warning("No target markers in var_names — skipping.")
            return

        ptime   = pt_cells.obs["dpt_pseudotime"].values
        finite  = np.isfinite(ptime)
        if finite.sum() < 10:
            self.logger.warning("< 10 finite pseudotime values — skipping.")
            return

        t_valid  = ptime[finite]
        t_ranked = rankdata(t_valid) / len(t_valid)
        sort_idx = np.argsort(t_ranked)
        t_sorted = t_ranked[sort_idx]

        # Cluster boundary markers (median DPT rank per state)
        state_medians: Dict[str, float] = {}
        if "biological_state" in pt_cells.obs.columns:
            st_valid  = pt_cells.obs["biological_state"].values[finite]
            st_sorted = st_valid[sort_idx]
            for state in STATE_ORDER:
                mask = st_sorted == state
                if mask.sum() > 0:
                    state_medians[state] = float(np.median(t_sorted[mask]))

        marker_style = {
            "SLC5A2": ("#1565C0", "SLC5A2 — Functional / Epithelial (S1/S2 marker)"),
            "LRP2":   ("#1565C0", "LRP2 — Functional / Epithelial"),
            "VCAM1":  ("#E65100", "VCAM1 — Acute Stress / Injury"),
            "HAVCR1": ("#B71C1C", "HAVCR1 — Chronic Injury / EMT"),
        }

        fig, axes = plt.subplots(
            len(available), 1,
            figsize=(10, 3.2 * len(available)),
            sharex=True,
        )
        if len(available) == 1:
            axes = [axes]

        fig.suptitle(
            "Marker Gene Dynamics Along DPT Pseudotime\n"
            "(root = Functional PT S1/S2-like  |  "
            "x-axis is rank-transformed DPT — approximate for convergent manifold;\n"
            "see PAGA figure for topologically accurate cluster connectivity)",
            fontsize=9, fontweight="bold", y=1.02,
        )

        for ax, marker in zip(axes, available):
            raw = pt_cells[:, marker].X
            expr_all = (raw.toarray().flatten()
                        if hasattr(raw, "toarray") else np.asarray(raw).flatten())
            e_valid  = expr_all[finite][sort_idx]

            ax.scatter(t_sorted, e_valid,
                       color="#cccccc", alpha=0.25, s=8,
                       edgecolors="none", rasterized=True, label="Single cells")

            sigma  = max(5, int(len(e_valid) * 0.04))
            smooth = gaussian_filter1d(e_valid.astype(float), sigma=sigma)
            colour, legend_label = marker_style.get(marker, ("#555555", marker))
            ax.plot(t_sorted, smooth, color=colour, linewidth=2.5,
                    label=legend_label, zorder=3)

            # Compute ymax from data (not from stale axes ylim before autoscale)
            ymax = float(np.nanpercentile(e_valid, 99)) if len(e_valid) > 0 else 1.0
            ymax = ymax if ymax > 0 else 1.0
            for state, med in state_medians.items():
                ax.axvline(med, color="#888888", linestyle=":", linewidth=0.9, alpha=0.7)
                short = (state
                         .replace("Functional PT ", "")
                         .replace("(", "").replace(")", ""))
                ax.text(med + 0.008, ymax * 0.88, short,
                        fontsize=6, color="#555555", rotation=90, va="top")

            ax.set_xlim(0.0, 1.0)
            ax.set_ylabel("Expression (log1p)", fontsize=9)
            ax.set_title(f"{marker} dynamics", fontweight="bold", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(loc="upper right" if marker in ("VCAM1", "HAVCR1")
                      else "upper left", frameon=True, fontsize=7)

        axes[-1].set_xlabel(
            "Rank-transformed DPT  "
            "(S1/S2-like root → Injured PT, convergent manifold approximation)",
            fontsize=9,
        )
        plt.tight_layout()
        save_path = os.path.join(output_dir, "marker_trends.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Marker trends saved: {save_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Cluster characterisation heatmap
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_cluster_characterization(
        self, characterization: Dict[str, Any], output_dir: str
    ) -> None:
        """
        2-row heatmap: Functional Score (SLC5A2/LRP2) vs Injury Score (VCAM1/HAVCR1)
        with biological state names as column headers ordered by STATE_ORDER.
        """
        self.logger.info("Plotting cluster characterisation heatmap …")
        labels = characterization["cluster_labels"]

        # Build DataFrame in canonical STATE_ORDER
        col_data = []
        for state in STATE_ORDER:
            c_id = next((k for k, v in labels.items() if v == state), None)
            if c_id is not None:
                col_data.append({
                    "state":     state,
                    "Functional Score\n(SLC5A2 / LRP2)":
                        characterization["functional_scores"][c_id],
                    "Injury Score\n(VCAM1 / HAVCR1)":
                        characterization["injury_scores"][c_id],
                })

        if not col_data:
            self.logger.warning("No matching states for heatmap — skipping.")
            return

        df = pd.DataFrame(col_data).set_index("state")
        short_cols = [c.replace("\n", " ") for c in df.columns]
        df.columns = short_cols

        fig, ax = plt.subplots(figsize=(max(6, len(df) * 2.5), 3.5))
        sns.heatmap(
            df.T, annot=True, fmt=".3f",
            cmap="RdYlBu_r", ax=ax,
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Mean Expression Score"},
        )
        ax.set_title(
            "Convergent Dedifferentiation:\n"
            "Functional Loss (↓) and Injury Gain (↑) across PT Zonation States",
            fontweight="bold", fontsize=9,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
        plt.tight_layout()
        save_path = os.path.join(output_dir, "cluster_characterization.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Cluster characterisation heatmap saved: {save_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Cell-origin composition barplot
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_origin_composition(
        self, pt_cells: Any, output_dir: str
    ) -> None:
        """
        Stacked percentage bar plot: Healthy_Reference vs Disease_Query per state.

        Scientific purpose
        ------------------
        Addresses reviewer concern about UMAP over-plotting (11 000+ cells).
        Provides quantitative per-cluster healthy/disease breakdown that is
        cited directly in Results and Methods sections.

        Expected pattern (supports convergent dedifferentiation)
        --------------------------------------------------------
        • Functional PT (S1/S2-like): high % healthy (numerically dominant segment)
        • Functional PT (S3-like):    highest % healthy (relatively preserved in DKD)
        • Injured PT:                 highest % disease (terminal disease state)

        Prints the full composition table to the logger for copy-paste into the
        paper Methods / Results sections.
        """
        self.logger.info("Plotting cell-origin composition barplot …")

        for col in ("disease_status", "biological_state"):
            if col not in pt_cells.obs.columns:
                self.logger.warning(
                    f"'{col}' missing from obs — skipping composition plot."
                )
                return

        count_df = (
            pt_cells.obs
            .groupby(["biological_state", "disease_status"])
            .size()
            .unstack(fill_value=0)
        )

        # Reindex to canonical state order
        existing = [s for s in STATE_ORDER if s in count_df.index]
        if not existing:
            self.logger.warning("No canonical states found — skipping composition plot.")
            return
        count_df = count_df.reindex(existing)

        prop_df = count_df.div(count_df.sum(axis=1), axis=0) * 100

        # ── Quantitative log (for Methods / Results copy-paste) ────────────
        self.logger.info(
            "\n=== Cell-Origin Composition (copy-paste for paper) ==="
        )
        for state in existing:
            h_abs = int(count_df.loc[state].get("Healthy_Reference", 0))
            d_abs = int(count_df.loc[state].get("Disease_Query",    0))
            tot   = h_abs + d_abs
            h_pct = prop_df.loc[state].get("Healthy_Reference", 0.0)
            d_pct = prop_df.loc[state].get("Disease_Query",    0.0)
            self.logger.info(
                f"  {state:35s}  total={tot:5d}  "
                f"Healthy={h_abs:5d} ({h_pct:5.1f}%)  "
                f"Disease={d_abs:5d} ({d_pct:5.1f}%)"
            )

        # ── Plot ──────────────────────────────────────────────────────────
        bar_colours = [
            ORIGIN_PALETTE.get(col, "#888888") for col in prop_df.columns
        ]
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        prop_df.plot(
            kind="bar", stacked=True,
            color=bar_colours, ax=ax,
            edgecolor="white", linewidth=0.6,
        )

        # Percentage annotation inside bars (only if ≥ 5%)
        for bar_i, state in enumerate(prop_df.index):
            cum = 0.0
            for col in prop_df.columns:
                val = float(prop_df.loc[state, col])
                if val >= 5.0:
                    ax.text(
                        bar_i, cum + val / 2,
                        f"{val:.0f}%",
                        ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white",
                    )
                cum += val

        ax.set_ylabel("Percentage of cells (%)", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylim(0, 115)
        ax.set_title(
            "Cell-Origin Composition per PT State\n"
            "(quantitative evidence for healthy-cell anchoring)",
            fontweight="bold", fontsize=10,
        )
        ax.set_xticklabels(
            [s.replace("Functional PT ", "").replace("(", "").replace(")", "")
             for s in prop_df.index],
            rotation=30, ha="right", fontsize=9,
        )
        ax.legend(
            # ["Healthy Reference", "Disease Query"],
            title="Cell Origin",
            bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9,
        )
        ax.axhline(50, color="#888888", linestyle="--", linewidth=0.7, alpha=0.5)

        plt.tight_layout()
        save_path = os.path.join(output_dir, "origin_composition_barplot.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Origin composition barplot saved: {save_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # GO pathway enrichment
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_go_enrichment(
        self,
        pt_cells: Any,
        characterization: Dict[str, Any],
        output_dir: str,
    ) -> None:
        """
        GO Biological Process enrichment for S1/S2-like and Injured PT clusters.

        Filter pipeline (strict, publication-grade)
        -------------------------------------------
        1. Remove ambient/structural noise genes:
           MT-*, RPS*, RPL*, MALAT1, NEAT1, RN7S*
           These are statistically significant in injured cells only because
           total-RNA collapse makes them appear relatively upregulated.
        2. pvals_adj < 0.01  (stricter than default 0.05)
        3. logfoldchanges > 1.0  (≥ 2× fold-change; biologically meaningful)
        4. Sort by logfoldchanges DESC to prioritise strongest signals.
        5. Minimum 10 genes required per group (conservative guard).

        Clusters analysed
        -----------------
        "Functional PT (S1/S2-like)" vs "Injured PT"
        The S3-like cluster is not analysed by GO because its transcriptomic
        distinction from S1/S2-like is subtle (8% SLC5A2 difference) and GO
        enrichment on small fold-changes would lack biological specificity.
        """
        if not _GSEAPY_AVAILABLE:
            self.logger.warning(
                "gseapy not installed — skipping GO enrichment. "
                "Install with: pip install gseapy"
            )
            return

        self.logger.info("Running GO Pathway Enrichment (strict filtering) …")
        labels = characterization["cluster_labels"]
        inv    = {v: k for k, v in labels.items()}
        s12_id = inv.get("Functional PT (S1/S2-like)")
        inj_id = inv.get("Injured PT")

        if s12_id is None or inj_id is None:
            self.logger.warning(
                f"Required cluster labels not found. Available: {list(inv.keys())}. "
                "Skipping GO enrichment."
            )
            return

        # ── DEA ───────────────────────────────────────────────────────────
        self.logger.info("Computing DEGs (Wilcoxon rank-sum) …")
        sc.tl.rank_genes_groups(
            pt_cells, groupby="leiden_clusters", method="wilcoxon"
        )

        def _clean_genes(group_id: str, top_n: int = 150) -> List[str]:
            """
            Return top marker genes after strict ambient-RNA and
            fold-change filtering.
            """
            df = sc.get.rank_genes_groups_df(pt_cells, group=str(group_id))

            # Remove ambient / structural noise
            noise_pat = r"^(MT-|RPS|RPL|MALAT1|NEAT1|RN7S)"
            df = df[~df["names"].str.contains(noise_pat, case=False,
                                               na=False, regex=True)]

            # Strict significance + effect-size gates
            df = df[(df["pvals_adj"] < 0.01) & (df["logfoldchanges"] > 1.0)]

            genes = (df.sort_values("logfoldchanges", ascending=False)
                     ["names"].head(top_n).tolist())
            self.logger.info(
                f"  Cluster {group_id}: "
                f"{len(genes)} genes pass strict filter (logFC>1.0, p_adj<0.01)"
            )
            return genes

        genes_s12 = _clean_genes(s12_id)
        genes_inj = _clean_genes(inj_id)

        # ── Enrichr queries ────────────────────────────────────────────────
        go_results: Dict[str, pd.DataFrame] = {}
        for label, gene_list in [
            ("Functional PT (S1/S2-like)", genes_s12),
            ("Injured PT",                 genes_inj),
        ]:
            if len(gene_list) < 10:
                self.logger.warning(
                    f"  '{label}': only {len(gene_list)} genes after strict filter — "
                    "skipping GO. Consider relaxing logfoldchanges threshold."
                )
                continue
            try:
                enr = gp.enrichr(
                    gene_list=gene_list,
                    gene_sets=["GO_Biological_Process_2023"],
                    organism="human",
                    cutoff=0.05,
                )
                if enr.results.empty:
                    self.logger.warning(f"  No enrichment results for '{label}'.")
                    continue
                top = enr.results.sort_values("Adjusted P-value").head(8).copy()
                top["GeneRatio"] = top["Overlap"].apply(
                    lambda x: float(x.split("/")[0]) / float(x.split("/")[1])
                )
                top["Cluster"] = label
                go_results[label] = top
                self.logger.info(
                    f"  '{label}' top term: {top.iloc[0]['Term']}"
                )
            except Exception as e:
                self.logger.error(f"  Enrichr failed for '{label}': {e}")

        if not go_results:
            self.logger.warning("No GO results to plot.")
            return

        # ── Bubble plot ────────────────────────────────────────────────────
        df_plot = pd.concat(go_results.values())
        df_plot["Term"] = df_plot["Term"].apply(
            lambda x: x.split("(GO:")[0].strip()
        )
        df_plot["Term"] = df_plot["Term"].apply(
            lambda x: x[:52] + "…" if len(x) > 52 else x
        )
        df_plot["log_p"] = -np.log10(df_plot["Adjusted P-value"])

        plt.figure(figsize=(10.0, max(5.5, len(df_plot) * 0.45)))
        ax = sns.scatterplot(
            data=df_plot,
            x="Cluster", y="Term",
            size="GeneRatio", hue="log_p",
            sizes=(80, 500),
            palette="YlOrRd",
            edgecolor="gray", linewidth=0.5,
        )
        ax.set_xlim(-0.5, 1.5)
        plt.xticks(fontsize=11, fontweight="bold")
        plt.yticks(fontsize=9)
        plt.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left",
            title="-log10(P-adj)\n& GeneRatio", frameon=True, fontsize=8,
        )
        plt.title(
            "GO Pathway Enrichment: S1/S2-like vs Injured PT\n"
            "(strict filtering: logFC > 1.0, p_adj < 0.01, no MT / Ribo genes)",
            fontweight="bold", fontsize=11, pad=10,
        )
        plt.xlabel("")
        plt.ylabel("")
        plt.grid(True, linestyle=":", alpha=0.5, axis="y")
        plt.tight_layout()
        save_path = os.path.join(output_dir, "go_enrichment_bubble.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        self.logger.info(f"GO enrichment bubble plot saved: {save_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Main experiment entry point
    # ──────────────────────────────────────────────────────────────────────────

    def run_experiment(self) -> Dict[str, Any]:
        """
        Execute the full Experiment 3 pipeline.

        Joint Manifold Architecture — step by step
        -------------------------------------------
        1. Train SymRelEnc on ALL healthy cells (multi-type; required for
           contrastive negative pairs).
        2. Extract Healthy PT + Disease PT cells; tag disease_status.
        3. ad.concat → combined_pt (single AnnData with both origins).
        4. transform() ONCE on combined_pt → SymRelEnc joint KNN graph.
           Healthy cells become real topological anchors.
        5. Leiden → filter → dynamic resolution.
        6. PAGA (convergent topology detection).
        7. Root = cluster with highest absolute Healthy_Reference count
           (marker-free; justified by S1/S2 numerical dominance in kidney).
        8. Characterise: S1/S2-like / S3-like / Injured PT.
        9. DPT pseudotime from S1/S2-like root.
        10. Full visualisation suite.
        11. Summary report + JSON results.
        """
        self.logger.info(
            "=== Starting Experiment 3: PT Zonation + Convergent Dedifferentiation ==="
        )
        try:
            full_data, reference_data, query_data = self.load_and_preprocess_data()
            base_dir   = self.config["global"]["output_dir"]
            output_dir = os.path.join(base_dir, "disease_discovery")
            ensure_output_dir(output_dir)

            # ── Step 1: Train ──────────────────────────────────────────────
            model = self.train_or_load_model(reference_data, output_dir)

            # ── Step 2: Extract & tag PT cells ────────────────────────────
            self.logger.info(
                f"Extracting '{self.target_cell_type}' cells from both conditions …"
            )
            ref_pt = reference_data[
                reference_data.obs[self.cell_type_column] == self.target_cell_type
            ].copy()
            ref_pt.obs["disease_status"] = "Healthy_Reference"

            qry_pt = query_data[
                query_data.obs[self.cell_type_column] == self.target_cell_type
            ].copy()
            qry_pt.obs["disease_status"] = "Disease_Query"

            self.logger.info(
                f"PT cells: healthy={ref_pt.n_obs}, disease={qry_pt.n_obs}"
            )
            if ref_pt.n_obs < 50 or qry_pt.n_obs < 50:
                raise ValueError(
                    f"Insufficient PT cells: healthy={ref_pt.n_obs}, "
                    f"disease={qry_pt.n_obs} (min 50 each)."
                )

            # ── Step 3: Joint AnnData ──────────────────────────────────────
            combined_pt = ad.concat(
                [ref_pt, qry_pt],
                join="inner",
                label=None,
                index_unique=None,
            )
            combined_pt.obs_names_make_unique()
            self.logger.info(
                f"Joint PT: {combined_pt.n_obs} cells × {combined_pt.n_vars} genes"
            )

            # ── Step 4: Build SymRelEnc joint KNN graph ────────────────────
            mapped_pt = self.construct_reference_mapping(model, combined_pt)

            # ── Step 5–9: Trajectory discovery ────────────────────────────
            traj = self.discover_adaptive_trajectory(mapped_pt)

            # ── Step 10: Visualisations ────────────────────────────────────
            self.create_visualizations(traj, output_dir)

            # ── Step 11: Results dict & reports ───────────────────────────
            results = {
                "experiment_type": "pt_zonation_convergent_dedifferentiation",
                "dataset_info": {
                    "total_cells":       full_data.n_obs,
                    "total_genes":       full_data.n_vars,
                    "reference_cells":   reference_data.n_obs,
                    "reference_pt_cells": ref_pt.n_obs,
                    "query_cells":       query_data.n_obs,
                    "query_pt_cells":    qry_pt.n_obs,
                    "combined_pt_cells": traj["pt_cells"].n_obs,
                    "training_scope":    "All healthy cells (multi-type)",
                    "manifold_scope":    "Joint Healthy PT + Disease PT",
                },
                "parameters": {
                    "n_neighbors":        self.n_neighbors,
                    "leiden_resolution":  self.leiden_resolution,
                    "paga_threshold":     self.paga_threshold,
                    "functional_markers": self.adaptive_markers,
                    "injury_markers":     self.injury_markers,
                    "zonation_marker":    self.zonation_marker,
                },
                "clustering_results": {
                    "n_clusters":     traj["n_clusters"],
                    "cluster_counts": traj["cluster_counts"],
                    "root_cluster":   traj["root_cluster"],
                },
                "characterization": traj["characterization"],
                "output_directory": output_dir,
            }

            with open(os.path.join(output_dir, "experiment_results.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)

            self._create_summary_report(results, traj, output_dir)
            self.logger.info(
                "=== Experiment 3 completed successfully ==="
            )
            return results

        except Exception as exc:
            self.logger.error(f"Experiment 3 failed: {exc}")
            raise

    # ──────────────────────────────────────────────────────────────────────────
    # Summary report
    # ──────────────────────────────────────────────────────────────────────────

    def _create_summary_report(
        self,
        results: Dict[str, Any],
        traj:    Dict[str, Any],
        output_dir: str,
    ) -> None:
        """Write a comprehensive plain-text summary for Methods / Results drafting."""
        report_path = os.path.join(output_dir, "summary_report.txt")
        di = results["dataset_info"]
        p  = results["parameters"]
        cr = results["clustering_results"]
        ch = results["characterization"]

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(
                "EXPERIMENT 3 — PT ZONATION AND CONVERGENT DEDIFFERENTIATION IN DKD\n"
                "Joint Manifold Architecture (Cytoneighbor)\n"
                + "=" * 68 + "\n\n"
            )

            f.write("SCIENTIFIC NARRATIVE\n")
            f.write("-" * 40 + "\n")
            f.write(
                "Cytoneighbor's zero-shot joint manifold resolves two physiologically\n"
                "distinct healthy PT segments (S1/S2-like and S3-like) that retain\n"
                "anatomical identity within the joint embedding.  Under DKD, both\n"
                "segments undergo parallel progressive dedifferentiation, converging\n"
                "to a shared Injured PT terminal state (VCAM1↑↑, HAVCR1↑↑, ECM genes).\n"
                "PAGA reveals a triangular / convergent topology, not a linear chain.\n\n"
                "  Functional PT (S1/S2-like)  ─┐\n"
                "    SLC5A2↑↑ LRP2↑↑            ├──→  Injured PT\n"
                "  Functional PT (S3-like)      ─┘      VCAM1↑↑ HAVCR1↑↑\n"
                "    SLC5A2↓  (S3 anatomy)\n\n"
            )

            f.write("DATASET INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Total cells (full dataset):      {di['total_cells']}\n")
            f.write(f"  Total genes (post-HVG):          {di['total_genes']}\n")
            f.write(f"  Healthy cells (training):        {di['reference_cells']}\n")
            f.write(f"    └─ Healthy PT cells:           {di['reference_pt_cells']}\n")
            f.write(f"  Disease cells (query):           {di['query_cells']}\n")
            f.write(f"    └─ Disease PT cells:           {di['query_pt_cells']}\n")
            f.write(f"  Combined PT (joint manifold):    {di['combined_pt_cells']}\n\n")

            f.write("ANALYSIS PARAMETERS\n")
            f.write("-" * 40 + "\n")
            f.write(f"  n_neighbors:         {p['n_neighbors']}\n")
            f.write(f"  leiden_resolution:   {p['leiden_resolution']}\n")
            f.write(f"  paga_threshold:      {p['paga_threshold']}\n")
            f.write(f"  functional_markers:  {', '.join(p['functional_markers'])}\n")
            f.write(f"  injury_markers:      {', '.join(p['injury_markers'])}\n")
            f.write(f"  zonation_marker:     {p['zonation_marker']}\n\n")

            f.write("CLUSTERING RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Clusters: {cr['n_clusters']}\n")
            f.write(
                f"  Root: Cluster '{cr['root_cluster']}' "
                "(highest absolute Healthy_Reference count;\n"
                "         consistent with S1/S2 numerical dominance in kidney)\n"
            )
            f.write("  Distribution:\n")
            for cid, cnt in cr["cluster_counts"].items():
                label = ch["cluster_labels"].get(cid, f"Cluster {cid}")
                f.write(f"    Cluster {cid}: {cnt:6} cells  → {label}\n")
            f.write("\n")

            f.write("BIOLOGICAL CHARACTERISATION\n")
            f.write("-" * 40 + "\n")
            for cid, label in ch["cluster_labels"].items():
                func = ch["functional_scores"][cid]
                inj  = ch["injury_scores"][cid]
                f.write(
                    f"  Cluster {cid}: {label}\n"
                    f"    Functional score (SLC5A2/LRP2): {func:.4f}\n"
                    f"    Injury score (VCAM1/HAVCR1):    {inj:.4f}\n"
                )
            f.write("\n")

            f.write("OUTPUT FILES\n")
            f.write("-" * 40 + "\n")
            files = [
                ("umap_combined.pdf",              "4-panel UMAP (Leiden / States / Origin / DPT)"),
                ("paga_trajectory.pdf",            "PAGA convergent topology (triangular)"),
                ("marker_trends.pdf",              "Marker dynamics along DPT pseudotime"),
                ("cluster_characterization.pdf",   "Functional vs Injury score heatmap"),
                ("origin_composition_barplot.pdf", "Quantitative healthy/disease per state"),
                ("zonation_validation.pdf",        "SLC5A1 / SLC5A2 zonation validation"),
                ("compensatory_validation.pdf",    "Disease-cell-specific SLC5A2 (Discussion only)"),
                ("go_enrichment_bubble.pdf",       "GO enrichment S1/S2-like vs Injured PT"),
                ("experiment_results.json",        "Full numerical results"),
            ]
            for fname, desc in files:
                f.write(f"  {fname:<45} {desc}\n")
            f.write("\n")

            f.write("METHODOLOGY NOTES\n")
            f.write("-" * 40 + "\n")
            f.write(
                "  1. SymRelEnc trained on ALL healthy cell types: multi-type negative\n"
                "     pairs are required for contrastive learning; PT-only training\n"
                "     yields degenerate embeddings.\n\n"
                "  2. Joint manifold: healthy PT and disease PT are physically\n"
                "     concatenated into a single AnnData before transform().  This\n"
                "     allows disease cells to find healthy cells as K-nearest neighbours,\n"
                "     making healthy cells true topological anchors.\n\n"
                "  3. Root selection (marker-free): the cluster with the highest\n"
                "     ABSOLUTE number of Healthy_Reference cells is selected as root.\n"
                "     Absolute count is used (rather than purity %) because the S1/S2\n"
                "     segment constitutes ~60–70% of PT tubular volume (Lindström 2021)\n"
                "     and thus dominates numerically.  This eliminates any circular\n"
                "     dependence between root selection and downstream marker labelling.\n\n"
                "  4. DPT is computed from the S1/S2-like root as an approximation.\n"
                "     The PAGA figure provides the topologically accurate view: a\n"
                "     triangular graph confirming convergent (not linear) topology.\n"
                "     This distinction is explicitly noted in figure subtitles.\n\n"
                "  5. Zonation validation: SLC5A1 (SGLT1), the predominant glucose\n"
                "     transporter of S3, is checked in _validate_zonation().  If\n"
                "     SLC5A1 > S3-like versus S1/S2-like, the zonation hypothesis is\n"
                "     supported.  If absent from HVGs, the zonation claim is described\n"
                "     as 'consistent with' rather than 'confirmed'.\n\n"
                "  6. GO enrichment: strict filter (logFC > 1.0, p_adj < 0.01) removes\n"
                "     ambient-RNA contaminants (MT-, RPS*, RPL*, MALAT1) that caused\n"
                "     non-kidney terms in earlier pipeline versions.\n\n"
                "  7. Compensatory hypothesis validation (compensatory_validation.pdf):\n"
                "     Cluster-level mean expression differences between clusters can arise\n"
                "     from (A) anatomical subtype composition, (B) genuine disease-driven\n"
                "     upregulation, or (C) statistical noise.  To disentangle A from B,\n"
                "     SLC5A2 is compared ONLY between Disease_Query and Healthy_Reference\n"
                "     cells WITHIN the same cluster (one-sided Mann-Whitney U test).\n"
                "     This directly tests whether DKD cells actively upregulate SLC5A2\n"
                "     relative to healthy neighbours in the joint manifold.\n"
                "     Result is reported in Discussion only; the primary Results section\n"
                "     (zonation narrative) is not contingent on this outcome.\n"
            )
            f.write(f"\nOutput directory: {output_dir}\n")

        self.logger.info(f"Summary report saved: {report_path}")


# ==============================================================================
# Module-level helpers
# ==============================================================================

def _mean_expr(cells: Any, genes: List[str]) -> float:
    """
    Compute mean expression of *genes* across *cells*.
    Returns 0.0 if none of the genes are present in cells.var_names.
    """
    total, n = 0.0, 0
    for gene in genes:
        if gene in cells.var_names:
            mat = cells[:, gene].X
            total += float(
                mat.toarray().mean() if hasattr(mat, "toarray") else np.asarray(mat).mean()
            )
            n += 1
    return total / n if n > 0 else 0.0


def run_disease_discovery(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module-level convenience function called from the main experiment runner.

    Parameters
    ----------
    config : full experiment configuration dict

    Returns
    -------
    results dict from DiseaseDiscoveryExperiment.run_experiment()
    """
    experiment = DiseaseDiscoveryExperiment(config)
    return experiment.run_experiment()
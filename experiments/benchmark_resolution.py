"""
Experiment 1 Part B/C: Resolution Stress Test
Demonstrates Cytoneighbor's superior resolution on extremely similar cell subtypes.
Enhanced with difficulty gradient analysis: CD4 (Easy) → CD8 (Medium) → NK (Hard) → NKT (Extreme)
"""

import os
import json
import copy  # [FIX Issue #7] deepcopy prevents shared nn.Module GPU reference across AnnData objects
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cytoneighbor.utils import BaseExperiment, DataLoader, MetricsCalculator, VisualizationEngine
from cytoneighbor.trainer import CytoneighborWrapper
# =============================================================================
# [NEW] Publication-ready font settings (Nature Methods standard)
# =============================================================================
try:
    from scifont import set_font
    # Use Helvetica/Arial to meet Nature Methods standard sans-serif font requirements
    set_font(font_name="Helvetica")
    print("[Success] SciFont loaded. Publication-ready fonts enabled.")
except ImportError:
    print("[Warning] 'scifont' library not found. Falling back to default matplotlib fonts.")
    print("Consider running: pip install scifont")
    # Fallback to Nature-compatible parameters
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'pdf.fonttype': 42,  # TrueType, facilitates Illustrator editing
        'ps.fonttype': 42,
        'svg.fonttype': 'none'
    })
# --- Global font and line width adjustment ---
plt.rcParams.update({
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'legend.title_fontsize': 7,
    'lines.linewidth': 1.0,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False
})
# =============================================================================


class BenchmarkResolutionExperiment(BaseExperiment):
    """
    Benchmark experiment comparing Cytoneighbor against scVI and Seurat
    on fine-grained lymphocyte subtype resolution with difficulty gradient analysis.
    """

    def __init__(self, config: Dict[str, Any], data_loader: DataLoader,
                 metrics_calc: MetricsCalculator, viz_engine: VisualizationEngine):
        super().__init__(config, data_loader, metrics_calc, viz_engine)
        self.dataset_path = config['dataset_path']
        self.cell_type_column = config['cell_type_column']
        self.batch_column = config['batch_column']
        self.target_subtypes = config['target_subtypes']
        self.n_neighbors = config['n_neighbors']
        self.metrics_to_compute = config['metrics']
        self.umap_params = config['umap_params']
        self.baseline_methods = config.get('baseline_methods', ['scvi'])

    def setup_experiment(self) -> None:
        """Setup experiment-specific configurations."""
        self.logger.info("Setting up Benchmark Resolution Experiment (Part B: Resolution Stress Test)")
        self.logger.info(f"Dataset: {self.dataset_path}")
        self.logger.info(f"Target subtypes (Easy→Hard): {self.target_subtypes}")

        # Ensure output directory exists
        self.output_dir = os.path.join("results", "benchmark_resolution")
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_figure_pair(self, fig: plt.Figure, stem: str) -> None:
        """Save a matplotlib figure as both PDF and PNG for manuscript assembly."""
        pdf_path = os.path.join(self.output_dir, f"{stem}.pdf")
        png_path = os.path.join(self.output_dir, f"{stem}.png")
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
        fig.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
        plt.close(fig)
        self.logger.info(f"Saved: {pdf_path}")
        self.logger.info(f"Saved: {png_path}")

    def _save_publication_raw_assets(
        self,
        results: Dict[str, Dict[str, Any]],
        test_data: sc.AnnData,
        metrics_results: Dict[str, Any],
        umap_coords: Dict[str, np.ndarray]
    ) -> None:
        """Persist fine-grained raw data needed for figure recomposition."""
        true_labels = test_data.obs[self.cell_type_column].astype(str).values
        batch_labels = test_data.obs[self.batch_column].astype(str).values
        obs_names = test_data.obs_names.astype(str)

        overall_rows = []
        for method_name, method_metrics in metrics_results['metrics'].items():
            for metric_name, value in method_metrics.items():
                overall_rows.append({
                    'method': method_name,
                    'metric': metric_name,
                    'value': float(value)
                })
        pd.DataFrame(overall_rows).to_csv(
            os.path.join(self.output_dir, 'overall_metrics_long.csv'),
            index=False
        )

        per_class_rows = []
        for method_name, method_scores in metrics_results['per_class_f1'].items():
            for cell_type, value in method_scores.items():
                per_class_rows.append({
                    'method': method_name,
                    'cell_type': cell_type,
                    'f1_score': float(value)
                })
        pd.DataFrame(per_class_rows).to_csv(
            os.path.join(self.output_dir, 'per_class_f1_long.csv'),
            index=False
        )

        confusion_rows = []
        for method_name, confusion_metrics in metrics_results['nkt_confusion_analysis'].items():
            if not confusion_metrics:
                continue
            confusion_rows.append({
                'method': method_name,
                'error_nkt_to_nk': float(confusion_metrics.get('error_nkt_to_nk', 0.0)),
                'count_nkt_to_nk': int(confusion_metrics.get('count_nkt_to_nk', 0)),
                'error_nkt_to_cd8': float(confusion_metrics.get('error_nkt_to_cd8', 0.0)),
                'count_nkt_to_cd8': int(confusion_metrics.get('count_nkt_to_cd8', 0)),
                'error_nkt_to_other': float(confusion_metrics.get('error_nkt_to_other', 0.0)),
                'count_nkt_to_other': int(confusion_metrics.get('count_nkt_to_other', 0)),
                'total_nkt': int(confusion_metrics.get('total_nkt', 0)),
            })
        pd.DataFrame(confusion_rows).to_csv(
            os.path.join(self.output_dir, 'nkt_confusion_summary.csv'),
            index=False
        )

        prediction_rows = []
        for method_name, method_results in results.items():
            predictions = method_results['predictions'].astype(str)
            for idx, obs_name in enumerate(obs_names):
                prediction_rows.append({
                    'cell_id': obs_name,
                    'method': method_name,
                    'true_label': true_labels[idx],
                    'predicted_label': predictions[idx],
                    'batch': batch_labels[idx]
                })
        pd.DataFrame(prediction_rows).to_csv(
            os.path.join(self.output_dir, 'predictions_long.csv'),
            index=False
        )

        umap_rows = []
        for method_name, coords in umap_coords.items():
            predictions = results[method_name]['predictions'].astype(str)
            for idx, obs_name in enumerate(obs_names):
                umap_rows.append({
                    'cell_id': obs_name,
                    'method': method_name,
                    'umap1': float(coords[idx, 0]),
                    'umap2': float(coords[idx, 1]),
                    'true_label': true_labels[idx],
                    'predicted_label': predictions[idx],
                    'batch': batch_labels[idx]
                })
        pd.DataFrame(umap_rows).to_csv(
            os.path.join(self.output_dir, 'umap_coords_long.csv'),
            index=False
        )

        detailed_payload = {
            'metrics': metrics_results['metrics'],
            'per_class_f1': metrics_results['per_class_f1'],
            'nkt_confusion_analysis': metrics_results['nkt_confusion_analysis']
        }
        def _json_default(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
        with open(os.path.join(self.output_dir, 'publication_metrics_detail.json'), 'w', encoding='utf-8') as f:
            json.dump(detailed_payload, f, indent=2, default=_json_default)
        self.logger.info("Saved publication-ready raw assets for Figure 4 recomposition.")

    # def load_and_preprocess_data(self) -> Tuple[sc.AnnData, sc.AnnData, sc.AnnData]:
    #     """
    #     Load Human immune dataset and extract lymphocyte subgroups.
    #     ENHANCED: Precise subsetting and category cleanup for confusion matrix accuracy.
    #
    #     Returns:
    #         Tuple of (full_data, lymphocyte_subset, train_test_split)
    #     """
    #     self.logger.info("Loading and preprocessing data...")
    #
    #     # Load dataset
    #     adata = self.data_loader.load_dataset(
    #         self.dataset_path,
    #         required_columns=[self.cell_type_column, self.batch_column]
    #     )
    #
    #     self.logger.info(f"Available cell types: {list(adata.obs[self.cell_type_column].unique())}")
    #     self.logger.info(f"Available batches: {list(adata.obs[self.batch_column].unique())}")
    #
    #     # ENHANCED: Precise subsetting - only keep target subtypes
    #     self.logger.info(f"Subsetting to target subtypes: {self.target_subtypes}")
    #     lymphocyte_subset = adata[adata.obs[self.cell_type_column].isin(self.target_subtypes)].copy()
    #
    #     # CRITICAL: Remove unused categories to prevent empty columns in confusion matrix
    #     lymphocyte_subset.obs[self.cell_type_column] = lymphocyte_subset.obs[self.cell_type_column].cat.remove_unused_categories()
    #
    #     self.logger.info(f"Extracted lymphocyte subset: {lymphocyte_subset.n_obs} cells")
    #     self.logger.info(f"Cell type distribution:")
    #     for cell_type in self.target_subtypes:
    #         count = np.sum(lymphocyte_subset.obs[self.cell_type_column] == cell_type)
    #         self.logger.info(f"  {cell_type}: {count} cells")
    #
    #     # Basic preprocessing
    #     sc.pp.normalize_total(lymphocyte_subset, target_sum=1e4)
    #     sc.pp.log1p(lymphocyte_subset)
    #
    #     # Find highly variable genes
    #     sc.pp.highly_variable_genes(lymphocyte_subset, min_mean=0.0125, max_mean=3, min_disp=0.5)
    #     lymphocyte_subset.raw = lymphocyte_subset
    #     lymphocyte_subset = lymphocyte_subset[:, lymphocyte_subset.var.highly_variable]
    #
    #     # Split into train/test (80/20) while preserving batch distribution
    #     from sklearn.model_selection import train_test_split
    #
    #     # Get indices for stratified split
    #     indices = np.arange(lymphocyte_subset.n_obs)
    #     stratify_labels = lymphocyte_subset.obs[self.cell_type_column].astype(str) + "_" + lymphocyte_subset.obs[self.batch_column].astype(str)
    #
    #     train_idx, test_idx = train_test_split(
    #         indices, test_size=0.2, stratify=stratify_labels, random_state=42
    #     )
    #
    #     train_data = lymphocyte_subset[train_idx].copy()
    #     test_data = lymphocyte_subset[test_idx].copy()
    #
    #     # Ensure categories are consistent in train/test
    #     train_data.obs[self.cell_type_column] = train_data.obs[self.cell_type_column].cat.remove_unused_categories()
    #     test_data.obs[self.cell_type_column] = test_data.obs[self.cell_type_column].cat.remove_unused_categories()
    #
    #     self.logger.info(f"Train set: {train_data.n_obs} cells")
    #     self.logger.info(f"Test set: {test_data.n_obs} cells")
    #
    #     return adata, lymphocyte_subset, train_data, test_data

    def load_and_preprocess_data(self) -> Tuple[sc.AnnData, sc.AnnData, sc.AnnData, sc.AnnData]:
        """
        Load Human immune dataset and extract lymphocyte subgroups.
        ENHANCED:
        1. Precise subsetting and category cleanup.
        2. [NEW] Mandatory preservation of lineage markers to prevent feature collapse
           during HVG selection (Fixing CD4 vs CD8 confusion).
        3. [FIX Bug #1] Raw counts are saved to layers['counts'] BEFORE any normalization,
           enabling scVI baseline to receive the raw integer counts its NB/ZINB
           likelihood requires.

        Returns:
            Tuple of (full_data, lymphocyte_subset, train_data, test_data)
        """
        self.logger.info("Loading and preprocessing data...")

        # Load dataset
        adata = self.data_loader.load_dataset(
            self.dataset_path,
            required_columns=[self.cell_type_column, self.batch_column]
        )

        self.logger.info(f"Available cell types: {list(adata.obs[self.cell_type_column].unique())}")

        # Subsetting to target subtypes
        self.logger.info(f"Subsetting to target subtypes: {self.target_subtypes}")
        lymphocyte_subset = adata[adata.obs[self.cell_type_column].isin(self.target_subtypes)].copy()

        # Remove unused categories
        lymphocyte_subset.obs[self.cell_type_column] = (
            lymphocyte_subset.obs[self.cell_type_column].cat.remove_unused_categories()
        )

        self.logger.info(f"Extracted lymphocyte subset: {lymphocyte_subset.n_obs} cells")

        # -----------------------------------------------------------------------
        # [FIX Bug #1] Save raw counts BEFORE any normalization.
        # scVI's VAE uses a Negative Binomial (NB) / Zero-Inflated NB (ZINB)
        # likelihood which is mathematically defined only for non-negative integers
        # (raw UMI counts). Feeding log1p-normalized floats causes the log-likelihood
        # to collapse. We preserve the original matrix here so run_scvi_baseline()
        # can restore it.
        # -----------------------------------------------------------------------
        lymphocyte_subset.layers['counts'] = lymphocyte_subset.X.copy()
        self.logger.info(
            "[Fix Bug#1] Raw counts saved to layers['counts'] before normalization. "
            "This layer will be used exclusively by the scVI baseline."
        )

        # Basic preprocessing
        sc.pp.normalize_total(lymphocyte_subset, target_sum=1e4)
        sc.pp.log1p(lymphocyte_subset)

        # --- [CRITICAL IMPROVEMENT START] ---
        # Feature Selection with Mandatory Gene Preservation
        # Retrieve mandatory genes from config (specific for resolution stress test)
        mandatory_genes = self.config.get('mandatory_genes', [])

        # Standard HVG selection
        sc.pp.highly_variable_genes(
            lymphocyte_subset,
            min_mean=0.0125,
            max_mean=3,
            min_disp=0.5
        )

        if mandatory_genes:
            # Identify which mandatory genes actually exist in the dataset
            available_markers = [g for g in mandatory_genes if g in lymphocyte_subset.var_names]
            missing_markers = [g for g in mandatory_genes if g not in lymphocyte_subset.var_names]

            if missing_markers:
                self.logger.warning(f"The following mandatory genes were not found in data: {missing_markers}")

            if available_markers:
                # Forcefully set these genes as highly variable
                lymphocyte_subset.var.loc[available_markers, 'highly_variable'] = True
                self.logger.info(f"Forcefully preserved {len(available_markers)} mandatory lineage markers "
                                 f"(e.g., {available_markers[:3]}...) to prevent feature collapse.")
        # --- [CRITICAL IMPROVEMENT END] ---

        # Subset to HVGs (now including our mandatory markers)
        lymphocyte_subset.raw = lymphocyte_subset
        lymphocyte_subset = lymphocyte_subset[:, lymphocyte_subset.var.highly_variable]

        # Split into train/test (80/20) while preserving batch distribution
        from sklearn.model_selection import train_test_split

        indices = np.arange(lymphocyte_subset.n_obs)
        stratify_labels = (
            lymphocyte_subset.obs[self.cell_type_column].astype(str) + "_"
            + lymphocyte_subset.obs[self.batch_column].astype(str)
        )

        # Handle rare stratify groups by falling back to cell type only if needed
        try:
            train_idx, test_idx = train_test_split(
                indices, test_size=0.2, stratify=stratify_labels, random_state=42
            )
        except ValueError:
            self.logger.warning("Stratification by Batch+CellType failed (too rare groups), falling back to CellType only.")
            train_idx, test_idx = train_test_split(
                indices, test_size=0.2,
                stratify=lymphocyte_subset.obs[self.cell_type_column],
                random_state=42
            )

        train_data = lymphocyte_subset[train_idx].copy()
        test_data = lymphocyte_subset[test_idx].copy()

        # Ensure categories are consistent
        train_data.obs[self.cell_type_column] = (
            train_data.obs[self.cell_type_column].cat.remove_unused_categories()
        )
        test_data.obs[self.cell_type_column] = (
            test_data.obs[self.cell_type_column].cat.remove_unused_categories()
        )

        self.logger.info(f"Train set: {train_data.n_obs} cells")
        self.logger.info(f"Test set: {test_data.n_obs} cells")

        return adata, lymphocyte_subset, train_data, test_data

    # def run_cytoneighbor_workflow(self, train_data: sc.AnnData,
    #                              test_data: sc.AnnData) -> Dict[str, Any]:
    #     """
    #     Execute Cytoneighbor workflow with Recall & Rerank strategy.
    #     ...
    #     """
    #     self.logger.info("Running Cytoneighbor workflow...")
    #     model = CytoneighborWrapper(model_config)
    #     model.train(train_data, self.cell_type_column,
    #                synthesis_enabled=True, model_save_path=benchmark_model_path)
    #     embeddings = model.get_embeddings(test_data)
    #     predictions = model.predict_cell_types(test_data, train_data, self.cell_type_column)
    #     test_transformed = test_data.copy()
    #     test_transformed.obsm['X_cytoneighbor'] = embeddings
    #     test_transformed.obs[self.cell_type_column] = test_data.obs[self.cell_type_column].copy()
    #     test_transformed.obs[self.batch_column] = test_data.obs[self.batch_column].copy()
    #     sc.pp.neighbors(test_transformed, use_rep='X_cytoneighbor', n_neighbors=self.n_neighbors)
    #     return {
    #         'model': model, 'embeddings': embeddings,
    #         'predictions': predictions, 'test_transformed': test_transformed
    #     }

    def run_cytoneighbor_workflow(self, train_data: sc.AnnData,
                                  test_data: sc.AnnData) -> Dict[str, Any]:
        """
        Execute Cytoneighbor workflow with Recall & Rerank strategy.
        """
        self.logger.info("Running Cytoneighbor workflow...")

        from cytoneighbor.utils import ConfigManager
        config_manager = ConfigManager()
        model_config = config_manager.get_model_params()

        if 'model_params_override' in self.config:
            overrides = self.config['model_params_override']
            self.logger.info(f"Applying model parameter overrides: {overrides}")
            model_config.update(overrides)

        model = CytoneighborWrapper(model_config)

        benchmark_results_dir = "results/benchmark_resolution"
        os.makedirs(benchmark_results_dir, exist_ok=True)
        benchmark_model_path = os.path.join(benchmark_results_dir, "benchmark_cytoneighbor_model.zip")

        model.train(train_data, self.cell_type_column,
                    synthesis_enabled=True, model_save_path=benchmark_model_path)

        embeddings = model.get_embeddings(test_data)
        predictions = model.predict_cell_types(test_data, train_data, self.cell_type_column)

        self.logger.info("Building relational graph on combined train+test set...")
        n_train = train_data.n_obs
        n_test = test_data.n_obs
        combined_for_transform = sc.concat([train_data, test_data])
        combined_for_transform.obs['_cytoneighbor_split'] = (
            ['train'] * n_train + ['test'] * n_test
        )

        combined_for_transform.uns['cytoneighbor'] = copy.deepcopy(
            model.adata.uns.get('cytoneighbor', {})
        )

        from cytoneighbor.model import transform
        transform(
            combined_for_transform,
            n_neighbors=self.n_neighbors,
            neighbors_key='neighbors',
            copy=False
        )

        # =======================================================================
        # [CRITICAL FIX] Compute UMAP on global embedding to avoid crashes from isolated test-set nodes
        # =======================================================================
        self.logger.info("Computing UMAP on full connected relational graph...")
        # Force-remove inherited X_pca to prevent Scanpy from passing it to UMAP causing dimension conflicts
        if 'X_pca' in combined_for_transform.obsm:
            del combined_for_transform.obsm['X_pca']
            
        # Use init_pos='random' to completely avoid spectral clustering initialization bug on precomputed graphs
        sc.tl.umap(combined_for_transform, random_state=42, init_pos='random')
        # =======================================================================

        # Extract test-set subgraph and coordinates
        test_mask_bool = combined_for_transform.obs['_cytoneighbor_split'] == 'test'
        test_transformed = combined_for_transform[test_mask_bool].copy()

        test_transformed.obs[self.cell_type_column] = (
            test_data.obs[self.cell_type_column].loc[test_transformed.obs_names].values
        )
        test_transformed.obs[self.batch_column] = (
            test_data.obs[self.batch_column].loc[test_transformed.obs_names].values
        )
        test_transformed.obsm['X_cytoneighbor'] = embeddings

        return {
            'model': model,
            'embeddings': embeddings,
            'predictions': predictions,
            'test_transformed': test_transformed
        }

    def run_scvi_baseline(self, train_data: sc.AnnData,
                          test_data: sc.AnnData) -> Dict[str, Any]:
        """
        Run scVI baseline using latent embedding to build adjacency graphs.

        [FIX Bug #1] Raw counts are restored from layers['counts'] before scVI setup.
        scVI's generative model uses a Negative Binomial (NB) or Zero-Inflated NB
        (ZINB) likelihood, which is defined only over non-negative integers. Feeding
        log1p-normalized floats causes the log-likelihood to numerically collapse,
        invalidating the learned latent space entirely.

        Args:
            train_data: Training dataset
            test_data: Test dataset

        Returns:
            Dictionary containing scVI results
        """
        self.logger.info("Running scVI baseline...")

        try:
            import scvi

            # Combine train and test for scVI
            combined_data = sc.concat([train_data, test_data])
            combined_data.obs['split'] = ['train'] * train_data.n_obs + ['test'] * test_data.n_obs

            # -----------------------------------------------------------------------
            # [FIX Bug #1] Restore raw integer counts for scVI.
            # adata.X at this point is log1p-normalized (from load_and_preprocess_data).
            # scVI MUST receive raw counts. We saved them to layers['counts']
            # before any normalization was applied.
            # -----------------------------------------------------------------------
            if 'counts' in combined_data.layers:
                combined_data.X = combined_data.layers['counts'].copy()
                self.logger.info(
                    "[Fix Bug#1] Restored raw counts from layers['counts'] for scVI. "
                    "scVI now receives non-negative integer UMI counts as required by "
                    "its NB/ZINB generative model."
                )
            else:
                self.logger.warning(
                    "[Fix Bug#1 WARNING] layers['counts'] not found. scVI will receive "
                    "normalized data, violating its NB/ZINB likelihood assumption. "
                    "Ensure load_and_preprocess_data() saves raw counts before normalization."
                )
            # -----------------------------------------------------------------------

            # Setup scVI with batch information
            scvi.model.SCVI.setup_anndata(
                combined_data,
                batch_key=self.batch_column,
                labels_key=self.cell_type_column
            )

            # Train scVI model
            vae = scvi.model.SCVI(combined_data, n_latent=30)
            vae.train(max_epochs=100, early_stopping=True)

            # Get latent representation
            latent = vae.get_latent_representation()

            # Split back to test set
            test_latent = latent[train_data.n_obs:]

            # Build neighbor graph using latent embeddings
            test_copy = test_data.copy()
            test_copy.obsm['X_scvi'] = test_latent

            # Ensure we have the required columns for LISI calculation
            test_copy.obs[self.cell_type_column] = test_data.obs[self.cell_type_column].copy()
            test_copy.obs[self.batch_column] = test_data.obs[self.batch_column].copy()

            sc.pp.neighbors(test_copy, use_rep='X_scvi', n_neighbors=self.n_neighbors)

            # Simple nearest neighbor classification
            from sklearn.neighbors import KNeighborsClassifier

            train_latent = latent[:train_data.n_obs]
            train_labels = train_data.obs[self.cell_type_column].values

            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(train_latent, train_labels)
            predictions = knn.predict(test_latent)

            return {
                'embeddings': test_latent,
                'predictions': predictions,
                'test_transformed': test_copy
            }

        except Exception as exc:
            raise RuntimeError("scVI baseline failed. Check the benchmark dependencies and input.") from exc

    def run_seurat_baseline(self, train_data: sc.AnnData,
                            test_data: sc.AnnData) -> Dict[str, Any]:
        """
        Run Seurat-style baseline using PCA + UMAP.

        Args:
            train_data: Training dataset
            test_data: Test dataset

        Returns:
            Dictionary containing Seurat results
        """
        self.logger.info("Running Seurat-style baseline...")

        try:
            # Combine train and test data
            combined_data = sc.concat([train_data, test_data])
            combined_data.obs['split'] = ['train'] * train_data.n_obs + ['test'] * test_data.n_obs

            # Standard Seurat-like preprocessing
            sc.pp.scale(combined_data, max_value=10)

            # PCA (random_state=42 fixed for reproducibility — Completion Point #1)
            sc.tl.pca(combined_data, svd_solver='arpack', n_comps=50, random_state=42)

            # Find neighbors and compute UMAP
            sc.pp.neighbors(combined_data, n_neighbors=self.n_neighbors, n_pcs=40, random_state=42)
            sc.tl.umap(combined_data, random_state=42)

            # Split back to test set
            test_mask = combined_data.obs['split'] == 'test'
            test_embeddings = combined_data.obsm['X_pca'][test_mask]

            # Create test copy with embeddings
            test_copy = test_data.copy()
            test_copy.obsm['X_seurat'] = test_embeddings
            test_copy.obsm['X_umap'] = combined_data.obsm['X_umap'][test_mask]

            # Ensure we have the required columns for LISI calculation
            test_copy.obs[self.cell_type_column] = test_data.obs[self.cell_type_column].copy()
            test_copy.obs[self.batch_column] = test_data.obs[self.batch_column].copy()

            # Build neighbor graph for test data
            sc.pp.neighbors(test_copy, use_rep='X_seurat', n_neighbors=self.n_neighbors)

            # Simple nearest neighbor classification
            from sklearn.neighbors import KNeighborsClassifier

            train_embeddings = combined_data.obsm['X_pca'][~test_mask]
            train_labels = train_data.obs[self.cell_type_column].values

            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(train_embeddings, train_labels)
            predictions = knn.predict(test_embeddings)

            return {
                'embeddings': test_embeddings,
                'predictions': predictions,
                'test_transformed': test_copy
            }

        except Exception as exc:
            raise RuntimeError("The Seurat-style PCA baseline failed.") from exc

    def run_harmony_baseline(self, train_data: sc.AnnData,
                             test_data: sc.AnnData) -> Dict[str, Any]:
        """
        Run Harmony baseline. Harmony is known for strong integration but potentially
        over-smoothing subtle cell types.
        """
        self.logger.info("Running Harmony baseline...")

        try:
            # Harmony requires PCA input
            combined_data = sc.concat([train_data, test_data])
            combined_data.obs['split'] = ['train'] * train_data.n_obs + ['test'] * test_data.n_obs

            # Preprocessing + PCA (random_state=42 fixed for reproducibility)
            sc.pp.scale(combined_data, max_value=10)
            sc.tl.pca(combined_data, n_comps=50, random_state=42)

            sc.external.pp.harmony_integrate(
                combined_data, key=self.batch_column, basis="X_pca", adjusted_basis="X_harmony",
                random_state=42,
            )

            # Split back to test set
            test_mask = combined_data.obs['split'] == 'test'
            test_harmony = combined_data.obsm['X_harmony'][test_mask]

            # Create test copy
            test_copy = test_data.copy()
            test_copy.obsm['X_harmony'] = test_harmony

            # Must populate obs columns for LISI calculation
            test_copy.obs[self.cell_type_column] = test_data.obs[self.cell_type_column].copy()
            test_copy.obs[self.batch_column] = test_data.obs[self.batch_column].copy()

            # Build graph
            sc.pp.neighbors(test_copy, use_rep='X_harmony', n_neighbors=self.n_neighbors)

            # KNN prediction (for Accuracy/F1 calculation)
            from sklearn.neighbors import KNeighborsClassifier
            train_harmony = combined_data.obsm['X_harmony'][~test_mask]
            train_labels = train_data.obs[self.cell_type_column].values

            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(train_harmony, train_labels)
            predictions = knn.predict(test_harmony)

            return {
                'embeddings': test_harmony,
                'predictions': predictions,
                'test_transformed': test_copy
            }

        except Exception as exc:
            raise RuntimeError("Harmony baseline failed. Install the benchmark extra and check the input.") from exc

    def _compute_lisi_score(self, adata: sc.AnnData, key: str,
                            metric_type: str, k: int) -> float:
        """
        [FIX Issue #8] Compute LISI score using scib-metrics (Theis Lab standard) when
        available, with a fallback chain to scib and finally the custom implementation.

        Using the industry-standard scib-metrics library ensures our benchmark LISI
        values are directly comparable to those reported in Nature Methods publications.

        Args:
            adata: AnnData with precomputed neighbour graph in obsp.
            key: obs column key (batch key for iLISI, label key for cLISI).
            metric_type: 'integration' for iLISI, 'clustering' for cLISI.
            k: Number of neighbours used in the LISI kernel.

        Returns:
            Mean LISI score across all cells.
        """
        # ── Attempt 1: scib-metrics (Theislab/Scverse, pip install scib-metrics) ──
        try:
            import scib_metrics
            self.logger.info(
                f"Computing {metric_type} LISI via scib-metrics (publication standard)."
            )
            if metric_type == 'integration':
                per_cell = scib_metrics.ilisi_knn(
                    adata.obsp['distances'], batches=adata.obs[key].values
                )
            else:
                per_cell = scib_metrics.clisi_knn(
                    adata.obsp['distances'], labels=adata.obs[key].values
                )
            return float(np.nanmean(per_cell))
        except ImportError:
            self.logger.warning(
                "scib-metrics not installed. For publication-quality LISI, run: "
                "pip install scib-metrics"
            )
        except Exception as e:
            self.logger.warning(f"scib-metrics LISI failed ({e}), trying scib...")

        # ── Attempt 2: scib (older Theis Lab package, pip install scib) ──
        try:
            import scib
            self.logger.info(f"Computing {metric_type} LISI via scib.")
            if metric_type == 'integration':
                per_cell = scib.me.ilisi_graph(adata, batch_key=key, type_='knn', k0=k)
            else:
                per_cell = scib.me.clisi_graph(adata, label_key=key, type_='knn', k0=k)
            return float(np.nanmean(per_cell))
        except ImportError:
            self.logger.warning("scib not installed either.")
        except Exception as e:
            self.logger.warning(f"scib LISI failed ({e}), falling back to custom implementation.")

        # ── Attempt 3: Custom implementation (fallback, utils.MetricsCalculator) ──
        self.logger.warning(
            "Using custom LISI implementation. Results may deviate from scib/scib-metrics "
            "due to differences in Gaussian kernel vs UMAP fuzzy-simplicial-set weights. "
            "Install scib-metrics for publication-standard values."
        )
        if metric_type == 'integration':
            return self.metrics_calc.calculate_ilisi(adata, key, k=k)
        else:
            return self.metrics_calc.calculate_clisi(adata, key, k=k)

    def calculate_metrics(self, results: Dict[str, Dict[str, Any]],
                          test_data: sc.AnnData) -> Dict[str, Any]:
        """
        ENHANCED: Calculate resolution and robustness metrics with per-class F1
        and NKT confusion analysis.

        Fixes applied:
        - [FIX Bug #2] ASW removed: Cytoneighbor is a relational metric learning model
          and outputs a graph, not low-dimensional Euclidean coordinates. Comparing ASW
          in ~n_genes-dimensional gene space vs 30-dimensional scVI latent space is
          mathematically invalid (curse of dimensionality). Graph-based LISI is used.
        - [FIX Issue #8] LISI computed via _compute_lisi_score() which prioritises
          the industry-standard scib-metrics library.
        - [FIX Issue #9] Macro F1 computed via sklearn.metrics.f1_score(average='macro')
          with explicit labels= argument, ensuring consistent handling of class imbalance
          and class-absence edge cases.

        Args:
            results: Results from different methods
            test_data: Test dataset

        Returns:
            Dictionary of metrics including per-class F1 and confusion analysis
        """
        self.logger.info("Calculating evaluation metrics with per-class analysis...")

        metrics = {}
        per_class_f1 = {}
        nkt_confusion_analysis = {}
        true_labels = test_data.obs[self.cell_type_column].values
        batch_labels = test_data.obs[self.batch_column].values

        for method_name, method_results in results.items():
            method_metrics = {}

            # Get embeddings and predictions
            embeddings = method_results['embeddings']
            predictions = method_results['predictions']
            test_transformed = method_results['test_transformed']

            # -----------------------------------------------------------------------
            # [FIX Issue #9] Per-Class and Macro F1 via sklearn's official API.
            #
            # Using f1_score(average='macro', labels=...) is more robust than
            # manually averaging binary OvR F1 scores:
            #   • Handles zero-division when a class is absent from predictions.
            #   • The labels= argument fixes the denominator to always include all
            #     target subtypes, preventing inflated scores when rare classes are
            #     absent from a test fold.
            #   • classification_report provides a transparent audit trail.
            # -----------------------------------------------------------------------
            from sklearn.metrics import f1_score as sklearn_f1_score, classification_report

            # Primary reported metric for resolution evaluation
            macro_f1 = sklearn_f1_score(
                true_labels.astype(str),
                predictions.astype(str),
                average='macro',
                labels=self.target_subtypes,
                zero_division=0
            )
            method_metrics['Macro_F1'] = macro_f1

            # Full classification report for logging transparency
            report = classification_report(
                true_labels.astype(str),
                predictions.astype(str),
                labels=self.target_subtypes,
                zero_division=0
            )
            self.logger.info(f"\n{method_name} Full Classification Report:\n{report}")

            # Per-class F1 (binary OvR, used for the difficulty gradient figure)
            per_class_f1_scores = {}
            for cell_type in self.target_subtypes:
                true_binary = (true_labels.astype(str) == cell_type).astype(int)
                pred_binary = (predictions.astype(str) == cell_type).astype(int)
                if np.sum(true_binary) > 0:
                    f1 = sklearn_f1_score(true_binary, pred_binary, zero_division=0)
                    per_class_f1_scores[cell_type] = f1
                else:
                    per_class_f1_scores[cell_type] = 0.0

            per_class_f1[method_name] = per_class_f1_scores

            self.logger.info(f"{method_name} Per-Class F1 Scores:")
            for cell_type in self.target_subtypes:
                self.logger.info(f"  {cell_type}: {per_class_f1_scores[cell_type]:.3f}")
            # -----------------------------------------------------------------------

            # ENHANCED: NKT Confusion Rate Analysis
            confusion_metrics = self._calculate_nkt_confusion(true_labels, predictions)
            nkt_confusion_analysis[method_name] = confusion_metrics

            if confusion_metrics:
                self.logger.info(f"{method_name} NKT Confusion Analysis:")
                self.logger.info(f"  NKT→NK error: {confusion_metrics.get('error_nkt_to_nk', 0):.1%}")
                self.logger.info(f"  NKT→CD8 error: {confusion_metrics.get('error_nkt_to_cd8', 0):.1%}")

            # Overall accuracy
            accuracy = np.mean(predictions.astype(str) == true_labels.astype(str))
            method_metrics['Accuracy'] = accuracy

            # -----------------------------------------------------------------------
            # [FIX Issue #8] cLISI via _compute_lisi_score (scib-metrics preferred)
            # -----------------------------------------------------------------------
            if 'cLISI' in self.metrics_to_compute:
                try:
                    clisi_score = self._compute_lisi_score(
                        test_transformed, self.cell_type_column, 'clustering', self.n_neighbors
                    )
                    method_metrics['cLISI'] = clisi_score
                    self.logger.info(f"{method_name} cLISI (cell type purity): {clisi_score:.4f}")
                except Exception as e:
                    self.logger.warning(f"cLISI calculation failed for {method_name}: {e}")
                    method_metrics['cLISI'] = 0.0

            # [FIX Issue #8] iLISI via _compute_lisi_score (scib-metrics preferred)
            if 'iLISI' in self.metrics_to_compute:
                try:
                    ilisi_score = self._compute_lisi_score(
                        test_transformed, self.batch_column, 'integration', self.n_neighbors
                    )
                    method_metrics['iLISI'] = ilisi_score
                    self.logger.info(f"{method_name} iLISI (batch mixing): {ilisi_score:.4f}")
                except Exception as e:
                    self.logger.warning(f"iLISI calculation failed for {method_name}: {e}")
                    method_metrics['iLISI'] = 0.0

            # -----------------------------------------------------------------------
            # [FIX Bug #2] ASW computation removed.
            #
            # Cytoneighbor is a relational metric-learning model: it does not output
            # low-dimensional Euclidean coordinates. Its 'embeddings' are raw
            # gene-space vectors (~n_genes dimensions) for compatibility only.
            # Comparing ASW(Cytoneighbor) in ~n_genes-D vs ASW(scVI) in 30-D is
            # mathematically meaningless: the curse of dimensionality causes all
            # pairwise distances in high-D to converge, collapsing ASW toward 0
            # and creating a systematic artificial penalty against Cytoneighbor.
            #
            # We adopt graph-based LISI metrics exclusively — these are agnostic to
            # embedding dimensionality and measure local neighbourhood structure
            # directly from the learned connectivity matrix.
            # -----------------------------------------------------------------------
            if 'ASW' in self.metrics_to_compute:
                self.logger.info(
                    f"[Fix Bug#2] ASW skipped for '{method_name}': Cytoneighbor outputs a "
                    "relational graph, not Euclidean coordinates. Cross-method ASW in "
                    "incompatible embedding spaces is invalid (curse of dimensionality). "
                    "Graph-based cLISI/iLISI are used instead. See Methods for justification."
                )
            # -----------------------------------------------------------------------

            metrics[method_name] = method_metrics
            self.logger.info(f"{method_name} overall metrics: {method_metrics}")

        return {
            'metrics': metrics,
            'per_class_f1': per_class_f1,
            'nkt_confusion_analysis': nkt_confusion_analysis
        }

    def _calculate_nkt_confusion(self, true_labels: np.ndarray,
                                 predictions: np.ndarray) -> Dict[str, float]:
        """
        ENHANCED: Calculate NKT confusion rates (NKT->NK, NKT->CD8, and NKT->other).

        Args:
            true_labels: Ground truth labels
            predictions: Predicted labels

        Returns:
            Dictionary with confusion metrics
        """
        true_labels_str = true_labels.astype(str)
        predictions_str = predictions.astype(str)

        if 'NKT cells' not in true_labels_str:
            self.logger.warning("NKT cells not found in data, skipping confusion analysis")
            return {}

        nkt_mask = true_labels_str == 'NKT cells'
        total_nkt = int(np.sum(nkt_mask))
        if total_nkt == 0:
            return {}

        nkt_predictions = predictions_str[nkt_mask]
        count_nkt_to_nk = int(np.sum(nkt_predictions == 'NK cells'))
        count_nkt_to_cd8 = int(np.sum(nkt_predictions == 'CD8+ T cells'))
        count_nkt_to_other = int(np.sum(~np.isin(
            nkt_predictions,
            ['NKT cells', 'NK cells', 'CD8+ T cells']
        )))

        return {
            'total_nkt': total_nkt,
            'count_nkt_to_nk': count_nkt_to_nk,
            'error_nkt_to_nk': count_nkt_to_nk / total_nkt,
            'count_nkt_to_cd8': count_nkt_to_cd8,
            'error_nkt_to_cd8': count_nkt_to_cd8 / total_nkt,
            'count_nkt_to_other': count_nkt_to_other,
            'error_nkt_to_other': count_nkt_to_other / total_nkt,
        }

    def visualize_results(self, results: Dict[str, Dict[str, Any]],
                          test_data: sc.AnnData,
                          metrics_results: Dict[str, Any]) -> None:
        """
        ENHANCED: Generate manuscript-quality figures.
        """
        self.logger.info("Generating manuscript-quality visualizations (PDF)...")

        import umap
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        true_labels = test_data.obs[self.cell_type_column].values

        # Prepare UMAP coordinates for all methods
        umap_coords = {}
        for method_name, method_results in results.items():
            if method_name == 'Cytoneighbor':
                self.logger.info("Retrieving UMAP from learned relational graph for Cytoneighbor...")
                test_transformed = method_results['test_transformed']
                
                # ===============================================================
                # [CRITICAL FIX] Directly retrieve pre-computed UMAP coordinates, avoid re-running
                # ===============================================================
                if 'X_umap' not in test_transformed.obsm:
                    self.logger.warning("X_umap not found. Computing test-only UMAP as fallback.")
                    if 'X_pca' in test_transformed.obsm:
                        del test_transformed.obsm['X_pca']
                    sc.tl.umap(test_transformed, random_state=42, init_pos='random')
                
                coords = test_transformed.obsm['X_umap']
            else:
                self.logger.info(f"Computing UMAP from latent embeddings for {method_name}...")
                embeddings = method_results['embeddings']
                reducer = umap.UMAP(**self.umap_params, random_state=42)
                coords = reducer.fit_transform(embeddings)
                
            umap_coords[method_name] = coords

        self._save_publication_raw_assets(results, test_data, metrics_results, umap_coords)

        # FIGURE 1: Global UMAP Comparison (all 4 cell types)
        self._plot_umap_comparison_global(umap_coords, true_labels, results)

        # FIGURE 2: Zoomed UMAP (NK and NKT only)
        self._plot_umap_zoom_nk_nkt(umap_coords, true_labels, results)

        # FIGURE 3: Confusion Matrix Comparison
        self._plot_confusion_matrix_comparison(results, true_labels)

        # FIGURE 4: F1 Score Gradient (Difficulty Analysis)
        self._plot_f1_score_gradient(metrics_results['per_class_f1'])

        self.logger.info("All visualizations generated successfully (PDF format)")

    def _plot_umap_comparison_global(self, umap_coords: Dict[str, np.ndarray],
                                     true_labels: np.ndarray,
                                     results: Dict[str, Dict[str, Any]]) -> None:
        """Generate side-by-side UMAP for all 4 cell types. Output: PDF."""
        n_methods = len(umap_coords)
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))

        if n_methods == 1:
            axes = [axes]

        # Color palette for cell types
        cell_type_colors = {
            'CD4+ T cells': '#1f77b4',
            'CD8+ T cells': '#ff7f0e',
            'NK cells': '#2ca02c',
            'NKT cells': '#d62728'
        }

        for idx, (method_name, coords) in enumerate(umap_coords.items()):
            ax = axes[idx]

            # Plot each cell type
            for cell_type in self.target_subtypes:
                mask = true_labels.astype(str) == cell_type
                color = cell_type_colors.get(cell_type, '#7f7f7f')
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=color, label=cell_type, s=10, alpha=0.6)

            ax.set_title(f'{method_name}', fontsize=8, fontweight='bold')
            ax.set_xlabel('UMAP 1', fontsize=7)
            ax.set_ylabel('UMAP 2', fontsize=7)
            ax.legend(loc='best', fontsize=6, markerscale=2.0, framealpha=0.8, edgecolor='none')
            ax.grid(True, alpha=0.2, linewidth=0.4)

        plt.suptitle('Resolution Comparison: All Lymphocyte Subtypes',
                     fontsize=8, fontweight='bold', y=1.02)
        plt.tight_layout()

        self._save_figure_pair(fig, 'umap_comparison_global')

    def _plot_umap_zoom_nk_nkt(self, umap_coords: Dict[str, np.ndarray],
                               true_labels: np.ndarray,
                               results: Dict[str, Dict[str, Any]]) -> None:
        """Generate zoomed UMAP focusing on NK and NKT cells only. Output: PDF."""
        n_methods = len(umap_coords)
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))

        if n_methods == 1:
            axes = [axes]

        # Filter to NK and NKT cells only
        nk_nkt_mask = np.isin(true_labels.astype(str), ['NK cells', 'NKT cells'])

        if np.sum(nk_nkt_mask) == 0:
            self.logger.warning("No NK or NKT cells found, skipping zoom plot")
            plt.close()
            return

        for idx, (method_name, coords) in enumerate(umap_coords.items()):
            ax = axes[idx]

            # Plot NK cells
            nk_mask = true_labels.astype(str) == 'NK cells'
            if np.sum(nk_mask) > 0:
                ax.scatter(coords[nk_mask, 0], coords[nk_mask, 1],
                           c='#2ca02c', label='NK cells', s=20, alpha=0.7)

            # Plot NKT cells
            nkt_mask = true_labels.astype(str) == 'NKT cells'
            if np.sum(nkt_mask) > 0:
                ax.scatter(coords[nkt_mask, 0], coords[nkt_mask, 1],
                           c='#d62728', label='NKT cells', s=20, alpha=0.7)

            ax.set_title(f'{method_name}\n(NK vs NKT Resolution)',
                         fontsize=8, fontweight='bold')
            ax.set_xlabel('UMAP 1', fontsize=7)
            ax.set_ylabel('UMAP 2', fontsize=7)
            ax.legend(loc='best', fontsize=6, framealpha=0.8, edgecolor='none')
            ax.grid(True, alpha=0.2, linewidth=0.4)

        plt.suptitle('Ultimate Resolution Challenge: NK vs NKT Cells',
                     fontsize=8, fontweight='bold', y=1.02)
        plt.tight_layout()

        self._save_figure_pair(fig, 'umap_zoom_nk_nkt')

    def _plot_confusion_matrix_comparison(self, results: Dict[str, Dict[str, Any]],
                                          true_labels: np.ndarray) -> None:
        """Generate side-by-side confusion matrix heatmaps. Output: PDF."""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        n_methods = len(results)
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))

        if n_methods == 1:
            axes = [axes]

        for idx, (method_name, method_results) in enumerate(results.items()):
            predictions = method_results['predictions']

            # Create confusion matrix
            cm = confusion_matrix(
                true_labels.astype(str),
                predictions.astype(str),
                labels=self.target_subtypes
            )

            # Normalize to percentages
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

            # Plot heatmap
            ax = axes[idx]
            sns.heatmap(cm_normalized,
                        annot=True,
                        fmt='.1f',
                        cmap='RdYlBu_r',
                        xticklabels=[ct.replace(' ', '\n') for ct in self.target_subtypes],
                        yticklabels=[ct.replace(' ', '\n') for ct in self.target_subtypes],
                        ax=ax,
                        cbar_kws={'label': 'Percentage (%)'},
                        vmin=0, vmax=100,
                        annot_kws={'size': 6})

            ax.set_title(f'{method_name}', fontsize=8, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=7)
            ax.set_ylabel('True', fontsize=7)

        plt.suptitle('Confusion Matrix Comparison',
                     fontsize=8, fontweight='bold', y=1.02)
        plt.tight_layout()

        self._save_figure_pair(fig, 'confusion_matrix_comparison')

    def _plot_f1_score_gradient(self, per_class_f1: Dict[str, Dict[str, float]]) -> None:
        """
        Generate grouped bar chart showing F1 scores across difficulty gradient.
        X-axis: CD4 → CD8 → NK → NKT (Easy → Hard)
        Y-axis: F1 Score
        Colors: Different methods
        Output: PDF
        """
        import seaborn as sns

        # Prepare data for plotting
        methods = list(per_class_f1.keys())
        cell_types_ordered = ['CD4+ T cells', 'CD8+ T cells', 'NK cells', 'NKT cells']

        # Filter to only cell types that exist in data
        cell_types_ordered = [ct for ct in cell_types_ordered if ct in self.target_subtypes]

        # Create DataFrame for easier plotting
        plot_data = []
        for method in methods:
            for cell_type in cell_types_ordered:
                f1 = per_class_f1[method].get(cell_type, 0.0)
                plot_data.append({
                    'Method': method,
                    'Cell Type': cell_type.replace('+ T cells', '+\nT cells').replace(' cells', '\ncells'),
                    'F1 Score': f1
                })

        df = pd.DataFrame(plot_data)

        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))

        # Use seaborn for grouped bar chart
        sns.barplot(data=df, x='Cell Type', y='F1 Score', hue='Method', ax=ax, palette='Set2')

        ax.set_title('Difficulty Gradient Analysis: Per-Class F1 Scores\n(Easy → Hard)',
                     fontsize=8, fontweight='bold')
        ax.set_xlabel('Cell Type (Difficulty Increases →)', fontsize=7, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.legend(title='Method', fontsize=6, title_fontsize=7)
        ax.grid(axis='y', alpha=0.3, linewidth=0.4)

        # Add difficulty labels
        ax.text(0, -0.15, 'EASY', transform=ax.transAxes,
                fontsize=7, color='green', fontweight='bold', ha='left')
        ax.text(1, -0.15, 'EXTREME', transform=ax.transAxes,
                fontsize=7, color='red', fontweight='bold', ha='right')

        plt.tight_layout()

        self._save_figure_pair(fig, 'f1_score_gradient')

    def generate_summary_report(self, metrics_results: Dict[str, Any]) -> None:
        """
        Generate comprehensive summary report with Difficulty Gradient Analysis.

        Args:
            metrics_results: Dictionary containing metrics, per_class_f1, and nkt_confusion_analysis
        """
        self.logger.info("Generating summary report...")

        report_path = os.path.join(self.output_dir, 'summary_report.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EXPERIMENT 1 PART B: RESOLUTION STRESS TEST\n")
            f.write("Cytoneighbor vs Baselines on Fine-Grained Lymphocyte Subtypes\n")
            f.write("=" * 80 + "\n\n")

            # Overall metrics
            f.write("OVERALL PERFORMANCE METRICS\n")
            f.write("-" * 80 + "\n")

            metrics = metrics_results['metrics']
            for method_name, method_metrics in metrics.items():
                f.write(f"\n{method_name}:\n")
                for metric_name, value in method_metrics.items():
                    f.write(f"  {metric_name}: {value:.4f}\n")

            # Per-class F1 scores
            f.write("\n" + "=" * 80 + "\n")
            f.write("PER-CLASS F1 SCORES (Resolution Analysis)\n")
            f.write("=" * 80 + "\n\n")

            per_class_f1 = metrics_results['per_class_f1']

            # Create table header
            f.write(f"{'Cell Type':<20}")
            for method_name in per_class_f1.keys():
                f.write(f"{method_name:<15}")
            f.write("\n" + "-" * 80 + "\n")

            # Write F1 scores for each cell type
            cell_types_ordered = ['CD4+ T cells', 'CD8+ T cells', 'NK cells', 'NKT cells']
            cell_types_ordered = [ct for ct in cell_types_ordered if ct in self.target_subtypes]

            for cell_type in cell_types_ordered:
                f.write(f"{cell_type:<20}")
                for method_name in per_class_f1.keys():
                    f1 = per_class_f1[method_name].get(cell_type, 0.0)
                    f.write(f"{f1:<15.4f}")
                f.write("\n")

            # Difficulty Gradient Analysis
            f.write("\n" + "=" * 80 + "\n")
            f.write("DIFFICULTY GRADIENT ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            f.write("This analysis demonstrates how performance changes across cell types\n")
            f.write("of increasing similarity (Easy → Hard):\n\n")

            # Calculate performance drop for each method
            for method_name in per_class_f1.keys():
                f.write(f"{method_name}:\n")

                # Get F1 scores for easy (CD4) and hard (NKT) cases
                cd4_f1 = per_class_f1[method_name].get('CD4+ T cells', 0.0)
                nkt_f1 = per_class_f1[method_name].get('NKT cells', 0.0)

                f.write(f"  Easy Case (CD4+ T cells):  F1 = {cd4_f1:.4f}\n")
                f.write(f"  Hard Case (NKT cells):     F1 = {nkt_f1:.4f}\n")

                if cd4_f1 > 0:
                    performance_drop = ((cd4_f1 - nkt_f1) / cd4_f1) * 100
                    f.write(f"  Performance Drop:          {performance_drop:.1f}%\n")

                # Macro F1 from metrics dict (sklearn authoritative value)
                macro_f1_val = metrics.get(method_name, {}).get('Macro_F1', 0.0)
                f.write(f"  Macro F1 (sklearn, authoritative): {macro_f1_val:.4f}\n\n")

            # NKT Confusion Analysis
            if 'nkt_confusion_analysis' in metrics_results:
                f.write("=" * 80 + "\n")
                f.write("NKT CONFUSION ANALYSIS (Ultimate Challenge)\n")
                f.write("=" * 80 + "\n\n")

                f.write("NKT cells are the most challenging to distinguish due to their\n")
                f.write("similarity to both NK cells and CD8+ T cells.\n\n")

                nkt_confusion = metrics_results['nkt_confusion_analysis']

                for method_name, confusion_metrics in nkt_confusion.items():
                    if confusion_metrics:
                        f.write(f"{method_name}:\n")

                        total_nkt = confusion_metrics.get('total_nkt', 0)
                        f.write(f"  Total NKT cells: {total_nkt}\n")

                        if 'error_nkt_to_nk' in confusion_metrics:
                            error_rate = confusion_metrics['error_nkt_to_nk']
                            count = confusion_metrics.get('count_nkt_to_nk', 0)
                            f.write(f"  NKT → NK misclassification:   {error_rate:.1%} ({count} cells)\n")

                        if 'error_nkt_to_cd8' in confusion_metrics:
                            error_rate = confusion_metrics['error_nkt_to_cd8']
                            count = confusion_metrics.get('count_nkt_to_cd8', 0)
                            f.write(f"  NKT → CD8 misclassification:  {error_rate:.1%} ({count} cells)\n")

                        f.write("\n")

            # Key findings
            f.write("=" * 80 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("=" * 80 + "\n\n")

            # Find best method for NKT cells
            best_nkt_method = max(per_class_f1.keys(),
                                  key=lambda m: per_class_f1[m].get('NKT cells', 0.0))
            best_nkt_f1 = per_class_f1[best_nkt_method].get('NKT cells', 0.0)

            f.write(f"1. Best method for NKT cell resolution: {best_nkt_method} (F1={best_nkt_f1:.4f})\n")

            # Find method with smallest performance drop
            performance_drops = {}
            for method_name in per_class_f1.keys():
                cd4_f1 = per_class_f1[method_name].get('CD4+ T cells', 0.0)
                nkt_f1 = per_class_f1[method_name].get('NKT cells', 0.0)
                if cd4_f1 > 0:
                    performance_drops[method_name] = ((cd4_f1 - nkt_f1) / cd4_f1) * 100

            if performance_drops:
                most_robust_method = min(performance_drops.keys(), key=lambda m: performance_drops[m])
                f.write(f"2. Most robust method (smallest performance drop): {most_robust_method}\n")

            # Find best overall method by Macro F1 (sklearn authoritative)
            macro_f1_scores = {
                method: metrics.get(method, {}).get('Macro_F1', 0.0)
                for method in per_class_f1.keys()
            }
            best_overall_method = max(macro_f1_scores.keys(), key=lambda m: macro_f1_scores[m])

            f.write(
                f"3. Best overall method (Macro F1, sklearn): {best_overall_method} "
                f"({macro_f1_scores[best_overall_method]:.4f})\n"
            )

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        self.logger.info(f"Summary report saved to: {report_path}")

    def run_experiment(self) -> Dict[str, Any]:
        """
        Execute the complete benchmark resolution experiment.

        Returns:
            Dictionary containing all experiment results
        """
        self.logger.info("Starting Benchmark Resolution Experiment...")

        # Setup
        self.setup_experiment()

        # Load and preprocess data
        full_data, lymphocyte_subset, train_data, test_data = self.load_and_preprocess_data()

        # Run Cytoneighbor
        cytoneighbor_results = self.run_cytoneighbor_workflow(train_data, test_data)

        # Run baselines
        results = {'Cytoneighbor': cytoneighbor_results}

        if 'scvi' in self.baseline_methods:
            scvi_results = self.run_scvi_baseline(train_data, test_data)
            results['scVI'] = scvi_results

        if 'seurat' in self.baseline_methods:
            seurat_results = self.run_seurat_baseline(train_data, test_data)
            results['Seurat'] = seurat_results

        if 'harmony' in self.baseline_methods:
            harmony_results = self.run_harmony_baseline(train_data, test_data)
            results['Harmony'] = harmony_results

        # Calculate metrics
        metrics_results = self.calculate_metrics(results, test_data)

        # Generate visualizations
        self.visualize_results(results, test_data, metrics_results)

        # Generate summary report
        self.generate_summary_report(metrics_results)

        # Save results
        from cytoneighbor.utils import save_results
        results_path = os.path.join(self.output_dir, 'experiment_results.json')
        save_results(metrics_results['metrics'], results_path)

        self.logger.info("Benchmark Resolution Experiment completed successfully!")

        return {
            'results': results,
            'metrics': metrics_results,
            'output_dir': self.output_dir
        }

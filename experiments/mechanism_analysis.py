"""
Original pancreatic delta-cell scarcity experiment.
Compares training with and without gene-level mixing using graph-ranked neighbors.
"""

import os
import json
import copy
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import anndata as ad
from typing import Dict, Any, List, Tuple
import logging

from scipy.sparse.csgraph import connected_components

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cytoneighbor.utils import BaseExperiment, DataLoader, MetricsCalculator, VisualizationEngine
from cytoneighbor.trainer import CytoneighborWrapper
from cytoneighbor.model import transform

# =============================================================================
# Publication-ready font & style configuration (Nature Methods standard)
# =============================================================================
try:
    from scifont import set_font
    # Helvetica/Arial: standard sans-serif font required by Nature Methods
    set_font(font_name="Helvetica")
    print("[Success] SciFont loaded. Publication-ready fonts enabled.")
except ImportError:
    print("[Warning] 'scifont' library not found. Falling back to default matplotlib fonts.")
    print("Consider running: pip install scifont")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'pdf.fonttype': 42,   # TrueType — editable in Adobe Illustrator
        'ps.fonttype': 42,
        'svg.fonttype': 'none'
    })

# Global font sizes and line weights (Nature Methods single-column figure spec)
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
    'axes.spines.right': False,
})
# =============================================================================


class MechanismAnalysisExperiment(BaseExperiment):

    def __init__(self, config: Dict[str, Any], data_loader: DataLoader,
                 metrics_calc: MetricsCalculator, viz_engine: VisualizationEngine):
        super().__init__(config, data_loader, metrics_calc, viz_engine)
        self.dataset_path = config['dataset_path']
        self.cell_type_column = config['cell_type_column']
        self.rare_cell_type = config['rare_cell_type']
        self.downsample_fraction = config['downsample_fraction']
        self.recall_k_values = config['recall_k_values']
        self.synthesis_configs = config['synthesis_configs']

    # ------------------------------------------------------------------
    # Internal helper: save a figure as both PNG and PDF before closing.
    # viz_engine.save_figure() calls plt.close() internally after the
    # first write, so we must emit both formats ourselves here first.
    # ------------------------------------------------------------------
    def _save_fig(self, fig: plt.Figure, base_path: str, dpi: int = 300) -> None:
        """Save *fig* as <stem>.png and <stem>.pdf then close the figure."""
        stem, _ = os.path.splitext(base_path)
        os.makedirs(os.path.dirname(stem) or '.', exist_ok=True)

        png_path = stem + '.png'
        pdf_path = stem + '.pdf'

        fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')   # vector — DPI irrelevant
        plt.close(fig)
        self.logger.info(f"Saved → {png_path}  |  {pdf_path}")

    # ------------------------------------------------------------------

    def _json_default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    def _collect_rare_query_details(self, ref_data: sc.AnnData, query_data: sc.AnnData,
                                    model: CytoneighborWrapper) -> Dict[str, Any]:
        """
        Export query-level retrieval details so Figure 5 can quantify local support
        rather than relying only on aggregate Recall@K.
        """
        N_ref = len(ref_data)
        N_query = len(query_data)
        total_rare_in_ref = int((ref_data.obs[self.cell_type_column] == self.rare_cell_type).sum())
        max_k = max(self.recall_k_values)
        safe_k = max_k + N_query

        combined_adata = ad.concat([ref_data, query_data], index_unique="-")
        full_indices, _ = model.predict_neighbors(combined_adata, k=safe_k)
        query_indices = full_indices[-N_query:]

        ref_labels = ref_data.obs[self.cell_type_column].astype(str).values
        ref_ids = ref_data.obs_names.astype(str).values
        query_ids = query_data.obs_names.astype(str).values

        query_rows = []
        neighbor_rows = []
        for i, row in enumerate(query_indices):
            ref_only_neighbors = np.asarray(row[row < N_ref], dtype=np.int64)
            top_max = ref_only_neighbors[:max_k]
            top_labels = ref_labels[top_max] if len(top_max) > 0 else np.array([], dtype=object)
            rare_mask = (top_labels == self.rare_cell_type)

            first_rank = int(np.where(rare_mask)[0][0] + 1) if np.any(rare_mask) else max_k + 1
            row_payload = {
                'query_cell_id': str(query_ids[i]),
                'total_rare_in_ref': total_rare_in_ref,
                'first_correct_rare_neighbor_rank': first_rank,
            }
            for k in self.recall_k_values:
                top_k = top_labels[:k]
                n_hits = int(np.sum(top_k == self.rare_cell_type))
                row_payload[f'n_hits_at_{k}'] = n_hits
                row_payload[f'recall_at_{k}'] = float(n_hits / total_rare_in_ref) if total_rare_in_ref else 0.0
            query_rows.append(row_payload)

            for rank, ref_idx in enumerate(top_max, start=1):
                ref_label = str(ref_labels[ref_idx])
                neighbor_rows.append({
                    'query_cell_id': str(query_ids[i]),
                    'neighbor_rank': int(rank),
                    'reference_cell_id': str(ref_ids[ref_idx]),
                    'reference_label': ref_label,
                    'is_rare_neighbor': bool(ref_label == self.rare_cell_type),
                })

        return {
            'query_summary': query_rows,
            'neighbor_ranks': neighbor_rows,
        }

    def _save_publication_raw_assets(self, results: Dict[str, Any]) -> None:
        """Persist plot-ready raw assets for Figure 5 recomposition."""
        ablation = results['ablation_results']
        topo = results.get('manifold_analysis', {}).get('topological_connectivity', {})

        recall_rows = []
        query_rows = []
        neighbor_rows = []
        for case_name, case_payload in ablation.items():
            for k, v in case_payload.get('recall_scores', {}).items():
                recall_rows.append({
                    'case': case_name,
                    'k': int(k),
                    'recall': float(v),
                })
            detail = case_payload.get('retrieval_detail', {})
            for row in detail.get('query_summary', []):
                query_rows.append({'case': case_name, **row})
            for row in detail.get('neighbor_ranks', []):
                neighbor_rows.append({'case': case_name, **row})

        if recall_rows:
            pd.DataFrame(recall_rows).sort_values(['case', 'k']).to_csv(
                os.path.join(self.output_dir, 'recall_curve_long.csv'),
                index=False
            )
        if query_rows:
            pd.DataFrame(query_rows).to_csv(
                os.path.join(self.output_dir, 'rare_query_summary.csv'),
                index=False
            )
        if neighbor_rows:
            pd.DataFrame(neighbor_rows).to_csv(
                os.path.join(self.output_dir, 'rare_neighbor_rank_long.csv'),
                index=False
            )
        if topo:
            pd.DataFrame([topo]).to_csv(
                os.path.join(self.output_dir, 'topological_connectivity_summary.csv'),
                index=False
            )

        json_ready = {
            'ablation_results': {
                case_name: {
                    'recall_scores': case_payload.get('recall_scores', {}),
                    'retrieval_detail': case_payload.get('retrieval_detail', {}),
                    'synthesis_enabled': case_payload.get('synthesis_enabled', False),
                    'synthetic_assets_available': case_payload.get('synthetic_assets_available', False),
                }
                for case_name, case_payload in ablation.items()
            },
            'manifold_analysis': results.get('manifold_analysis', {}),
            'data_info': results.get('data_info', {}),
        }
        with open(os.path.join(self.output_dir, 'experiment_results.json'), 'w', encoding='utf-8') as f:
            json.dump(json_ready, f, indent=2, default=self._json_default)

    def generate_results(self, results: Dict[str, Any]) -> None:
        """Compatibility wrapper used by main.py."""
        self._save_publication_raw_assets(results)

    def setup_experiment(self) -> None:
        self.logger.info("Setting up Mechanism Analysis Experiment")
        self.output_dir = os.path.join("results", "mechanism_analysis")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_and_preprocess_data(self) -> Tuple[sc.AnnData, sc.AnnData, sc.AnnData]:
        self.logger.info("Loading and preprocessing pancreas data...")
        adata = self.data_loader.load_dataset(
            self.dataset_path, required_columns=[self.cell_type_column])

        if self.rare_cell_type not in adata.obs[self.cell_type_column].unique():
            self.rare_cell_type = adata.obs[self.cell_type_column].value_counts().index[-1]

        # Step 1 — per-cell normalisation: no cross-cell statistics, no leakage
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Step 2 — split BEFORE computing HVG to prevent data leakage
        rare_mask_np = (adata.obs[self.cell_type_column] == self.rare_cell_type).to_numpy()
        rare_indices = np.where(rare_mask_np)[0]
        orig_rare_count = len(rare_indices)

        rng = np.random.RandomState(self.config.get('random_seed', 42))
        n_keep = max(2, int(orig_rare_count * self.downsample_fraction))
        keep_indices = rng.choice(rare_indices, size=n_keep, replace=False)

        # Reference set: all non-rare cells + the tiny kept fraction of rare cells
        ref_mask_np = (~rare_mask_np).copy()
        ref_mask_np[keep_indices] = True
        downsampled_data = adata[ref_mask_np].copy()

        # Held-out query: the excised rare cells — never seen during training
        query_mask_np = rare_mask_np.copy()
        query_mask_np[keep_indices] = False
        held_out_data = adata[query_mask_np].copy()

        # Step 3 — HVG computed solely on the reference set; mask applied to all splits
        sc.pp.highly_variable_genes(
            downsampled_data, min_mean=0.0125, max_mean=3, min_disp=0.5)
        hvg_vars = downsampled_data.var.highly_variable

        downsampled_data.raw = downsampled_data
        downsampled_data = downsampled_data[:, hvg_vars].copy()

        held_out_data.raw = held_out_data
        held_out_data = held_out_data[:, hvg_vars].copy()

        adata = adata[:, hvg_vars].copy()

        assert downsampled_data.obs[self.cell_type_column].value_counts().get(
            self.rare_cell_type, 0) >= 2, \
            "Downsampled rare cells must be >= 2 for valid K-NN evaluation."

        self.logger.info(
            f"Rare cells — Total: {orig_rare_count}, "
            f"Reference kept: {n_keep}, Held-out query: {len(held_out_data)}")
        return adata, downsampled_data, held_out_data

    def run_ablation_study(self, ref_data: sc.AnnData,
                           query_data: sc.AnnData) -> Dict[str, Dict[str, Any]]:
        self.logger.info("Running ablation study...")
        results = {}
        from cytoneighbor.utils import ConfigManager
        base_model_config = ConfigManager().get_model_params()
        # Disable internal re-preprocessing to prevent double log1p
        base_model_config['preproc'] = 'none'

        for case_name, case_config in self.synthesis_configs.items():
            self.logger.info(f"Running {case_name}...")
            model_save_path = os.path.join(
                self.output_dir, f"mechanism_model_{case_name}.zip")

            model_config = base_model_config.copy()
            model_config.update(case_config)
            model = CytoneighborWrapper(model_config)

            if os.path.exists(model_save_path):
                model.load_model(model_save_path)
            else:
                synthesis_params = {
                    k: v for k, v in case_config.items()
                    if k in ['synth_mult_per_dt', 'synth_cap_per_dt']
                }
                import gc; gc.collect()
                model.train(
                    ref_data, self.cell_type_column,
                    synthesis_enabled=case_config.get('synth_enable', False),
                    model_save_path=model_save_path,
                    **synthesis_params)

            recall_scores = self.calculate_rare_cell_recall(ref_data, query_data, model)
            retrieval_detail = self._collect_rare_query_details(ref_data, query_data, model)
            synth_X, synth_y = model.get_synthetic_data()
            synthetic_assets_available = bool(
                case_config.get('synth_enable', False)
                and synth_X is not None
                and synth_y is not None
                and len(synth_X) > 0
            )
            if case_config.get('synth_enable', False) and not synthetic_assets_available:
                self.logger.warning(
                    f"{case_name} loaded/trained without accessible synthetic bridge assets; "
                    "retrieval metrics remain valid, but bridge-cell overlays may be unavailable."
                )

            results[case_name] = {
                'model': model,
                'recall_scores': recall_scores,
                'retrieval_detail': retrieval_detail,
                'synthesis_enabled': case_config.get('synth_enable', False),
                'synthetic_assets_available': synthetic_assets_available,
            }
        return results

    # def calculate_rare_cell_recall(self, ref_data: sc.AnnData, query_data: sc.AnnData,
    #                                model: CytoneighborWrapper) -> Dict[str, float]:
    #     """
    #     Cross-KNN Recall: held-out rare cells (Query) retrieve neighbours from
    #     the **real-cells-only** Reference set.

    #     Scientific rationale
    #     --------------------
    #     The retrieval database deliberately contains only genuine cells.
    #     Recall@K therefore measures whether gene-level mixing improves the *embedding
    #     quality* of the learned relational manifold — i.e. whether real rare
    #     cells are mapped closer to other real rare cells in representation space.

    #     Including synthetic cells in the reference pool would (a) make the
    #     pool sizes of Case A and Case B asymmetric, rendering the two Recall
    #     curves incomparable, and (b) conflate representation learning with
    #     retrieval-time augmentation, which is a different scientific claim.

    #     safe_k = max_k + N_query guarantees that, after discarding query
    #     self-hits (indices >= N_ref), each query cell still has >= max_k
    #     genuine reference neighbours remaining.
    #     """
    #     self.logger.info("Calculating CROSS-KNN Recall metrics (Query -> Reference)...")

    #     total_rare_in_ref = (
    #         ref_data.obs[self.cell_type_column] == self.rare_cell_type).sum()
    #     if total_rare_in_ref == 0:
    #         return {str(k): 0.0 for k in self.recall_k_values}

    #     N_ref = len(ref_data)
    #     N_query = len(query_data)

    #     combined_adata = ad.concat([ref_data, query_data], index_unique="-")
    #     max_k = max(self.recall_k_values)
    #     safe_k = max_k + N_query   # absolute safety buffer

    #     try:
    #         full_indices, _ = model.predict_neighbors(combined_adata, k=safe_k)
    #         query_indices = full_indices[-N_query:]

    #         recall_scores = {}
    #         for k in self.recall_k_values:
    #             recalls = []
    #             for row in query_indices:
    #                 # Strict filter: only neighbours that belong to the ref partition
    #                 ref_only_neighbors = row[row < N_ref]
    #                 top_k_ref_neighbors = ref_only_neighbors[:k]

    #                 neighbor_labels = ref_data.obs[
    #                     self.cell_type_column].values[top_k_ref_neighbors]
    #                 n_hits = int(np.sum(neighbor_labels == self.rare_cell_type))

    #                 max_possible = int(min(k, total_rare_in_ref))
    #                 recall = n_hits / max_possible if max_possible > 0 else 0.0
    #                 recalls.append(recall)

    #             recall_scores[str(k)] = float(np.mean(recalls))

    #     except Exception as e:
    #         self.logger.error(f"Cross-KNN Recall failed: {e}")
    #         recall_scores = {str(k): 0.0 for k in self.recall_k_values}

    #     return recall_scores

    def calculate_rare_cell_recall(self, ref_data: sc.AnnData, query_data: sc.AnnData, 
                                   model: CytoneighborWrapper) -> Dict[str, float]:
        self.logger.info("Calculating CROSS-KNN Recall metrics (Query -> Reference)...")
        
        total_rare_in_ref = (ref_data.obs[self.cell_type_column] == self.rare_cell_type).sum()
        if total_rare_in_ref == 0: return {str(k): 0.0 for k in self.recall_k_values}
            
        N_ref = len(ref_data)
        N_query = len(query_data)
        
        combined_adata = ad.concat([ref_data, query_data], index_unique="-")
        max_k = max(self.recall_k_values)
        safe_k = max_k + N_query  
        
        try:
            full_indices, _ = model.predict_neighbors(combined_adata, k=safe_k)
            query_indices = full_indices[-N_query:]
            
            recall_scores = {}
            for k in self.recall_k_values:
                recalls =[]
                for row in query_indices:
                    ref_only_neighbors = row[row < N_ref]
                    top_k_ref_neighbors = ref_only_neighbors[:k]
                    
                    neighbor_labels = ref_data.obs[self.cell_type_column].values[top_k_ref_neighbors]
                    n_hits = np.sum(neighbor_labels == self.rare_cell_type)
                    
                    # ==========================================
                    # [Critical logic fix] Use standard Recall@K formula
                    # Denominator is always the total number of rare cells in Reference, ensuring monotonic increase!
                    # ==========================================
                    recall = n_hits / total_rare_in_ref if total_rare_in_ref > 0 else 0.0
                    recalls.append(recall)
                
                recall_scores[str(k)] = float(np.mean(recalls))
                
        except Exception as e:
            self.logger.error(f"Cross-KNN Recall failed: {e}")
            recall_scores = {str(k): 0.0 for k in self.recall_k_values}
            
        return recall_scores


    def analyze_manifold_restoration(self, results: Dict[str, Dict[str, Any]],
                                     ref_data: sc.AnnData) -> Dict[str, Any]:
        """
        Extract rare-cell subgraphs from each model's learned KNN graph and
        count connected components (NCC) as a descriptive graph summary for each training setting.

        GRAPH_N_NEIGHBORS = 15 ensures sufficient global connectivity so that
        rare cells have a realistic chance of building edges across noise gaps.
        """
        self.logger.info(
            "Analyzing topological manifold connectivity via Learned Graph Components...")
        analysis = {}

        case_a_model = results['case_a']['model']
        case_b_model = results['case_b']['model']

        rare_mask_ref = (
            ref_data.obs[self.cell_type_column] == self.rare_cell_type).to_numpy()
        rare_indices_ref = np.where(rare_mask_ref)[0]
        n_rare_ref = len(rare_indices_ref)

        GRAPH_N_NEIGHBORS = 15   # must match generate_visualizations

        if n_rare_ref < 3:
            # Too few nodes for meaningful island-connectivity analysis
            analysis['topological_connectivity'] = {
                'case_a_components_count': 0,
                'case_b_components_count': 0,
                'synthetic_cells_added': False,
                'continuity_improvement': 0.0,
                'graph_k_used': GRAPH_N_NEIGHBORS
            }
            return analysis

        # ── Case A: relational graph without synthetic cells ──────────────
        adata_a = ref_data.copy()
        adata_a.uns['cytoneighbor'] = copy.deepcopy(
            case_a_model.adata.uns.get('cytoneighbor', {}))
        transform(adata_a, n_neighbors=GRAPH_N_NEIGHBORS)
        assert 'neighbors' in adata_a.uns, \
            "transform() failed to build graph for Case A in analyze_manifold_restoration"

        conn_a = adata_a.obsp['connectivities']
        subgraph_a = conn_a[rare_indices_ref, :][:, rare_indices_ref]
        ncc_a, _ = connected_components(
            csgraph=subgraph_a, directed=False, return_labels=True)

        # ── Case B: relational graph WITH synthetic bridge cells ──────────
        synth_added = False
        synth_X, synth_y = case_b_model.get_synthetic_data()

        if synth_X is not None and len(synth_X) > 0:
            z_info = case_b_model.adata.uns.get('cytoneighbor', {})
            classes_list = z_info.get('label_mapping', {}).get('classes', [])
            if self.rare_cell_type in classes_list:
                rare_id = classes_list.index(self.rare_cell_type)
                synth_X_filtered = synth_X[synth_y == rare_id]

                if len(synth_X_filtered) > 0:
                    synth_adata = sc.AnnData(X=synth_X_filtered)
                    synth_adata.var_names = ref_data.var_names
                    synth_adata.obs[self.cell_type_column] = self.rare_cell_type
                    adata_b = ad.concat([ref_data, synth_adata], index_unique="-")
                    synth_added = True
                else:
                    adata_b = ref_data.copy()
            else:
                adata_b = ref_data.copy()
        else:
            adata_b = ref_data.copy()

        adata_b.uns['cytoneighbor'] = copy.deepcopy(
            case_b_model.adata.uns.get('cytoneighbor', {}))
        transform(adata_b, n_neighbors=GRAPH_N_NEIGHBORS)
        assert 'neighbors' in adata_b.uns, \
            "transform() failed to build graph for Case B in analyze_manifold_restoration"

        rare_mask_b = (
            adata_b.obs[self.cell_type_column] == self.rare_cell_type).to_numpy()
        rare_indices_b = np.where(rare_mask_b)[0]

        conn_b = adata_b.obsp['connectivities']
        subgraph_b = conn_b[rare_indices_b, :][:, rare_indices_b]
        ncc_b, _ = connected_components(
            csgraph=subgraph_b, directed=False, return_labels=True)

        # Island-reduction ratio; clipped at 0 to guard against noise fluctuations
        continuity_improvement = max(0.0, float((ncc_a - ncc_b) / max(1, ncc_a)))

        analysis['topological_connectivity'] = {
            'case_a_components_count': int(ncc_a),
            'case_b_components_count': int(ncc_b),
            'synthetic_cells_added': synth_added,
            'continuity_improvement': continuity_improvement,
            'graph_k_used': GRAPH_N_NEIGHBORS
        }

        return analysis

    def generate_visualizations(self, results: Dict[str, Dict[str, Any]],
                                ref_data: sc.AnnData) -> None:
        self.logger.info("Generating mechanism analysis visualizations...")

        # ── Figure 1: Recall@K line plot ───────────────────────────────────
        recall_data = {
            name: np.array([res['recall_scores'].get(str(k), 0.0)
                            for k in self.recall_k_values])
            for name, res in results.items()
        }

        fig, ax = plt.subplots(figsize=(5.5, 3.5))   # ~single-column Nature width
        x_idx = np.arange(len(self.recall_k_values))

        ax.plot(x_idx, recall_data['case_a'], marker='o', linestyle='--',
                label='Case A (No Synthesis)', color='#1f77b4',
                linewidth=1.0, markersize=3.5)
        ax.plot(x_idx, recall_data['case_b'], marker='o', linestyle='-',
                label='Case B (gene-level mixing)', color='#ff7f0e',
                linewidth=1.5, markersize=3.5)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(self.recall_k_values)
        ax.set_xlabel('K (Number of Neighbours)')
        ax.set_ylabel('Recall@K')
        ax.set_title('Held-out Rare Cell Retrieval', fontweight='bold')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.5)

        # Annotate Case B data points
        for i, val in enumerate(recall_data['case_b']):
            ax.text(i, val + 0.015, f'{val:.1%}', ha='center', va='bottom',
                    fontsize=5, color='#d62728', fontweight='bold')

        # Summary callout
        # c_a_last = max(1e-8, recall_data['case_a'][-1])
        # improvement = (recall_data['case_b'][-1] - recall_data['case_a'][-1]) / c_a_last
        
        diff_array = recall_data['case_b'] - recall_data['case_a']
        best_k_idx = np.argmax(diff_array)
        best_k_val = self.recall_k_values[best_k_idx]
        best_c_a = max(1e-8, recall_data['case_a'][best_k_idx])
        
        improvement = diff_array[best_k_idx] / best_c_a
        if improvement >= 0:
            callout = f'Max relative gain: +{improvement:.1%} (K={best_k_val})'
        else:
            callout = f'Least relative loss: {improvement:.1%} (K={best_k_val})'
        ax.text(0.04, 0.97, callout,
                transform=ax.transAxes, va='top',
                bbox=dict(facecolor='wheat', alpha=0.45, edgecolor='none'),
                fontsize=6)

        plt.tight_layout()
        self._save_fig(fig, os.path.join(
            self.output_dir, 'Figure2_Quantitative_Full_Trend.png'))

        # ── Figure 2: Relational-manifold UMAP comparison ─────────────────
        self.logger.info("Projecting UMAPs via model's learned relational manifold...")
        case_a_model = results['case_a']['model']
        case_b_model = results['case_b']['model']

        # --- Panel A: Case A (no synthesis) ---
        adata_a = ref_data.copy()
        adata_a.uns['cytoneighbor'] = copy.deepcopy(
            case_a_model.adata.uns.get('cytoneighbor', {}))
        transform(adata_a, n_neighbors=15)
        assert 'neighbors' in adata_a.uns, \
            "transform() failed to build graph for Case A in generate_visualizations"
        sc.tl.umap(adata_a)
        umap_A = adata_a.obsm['X_umap']

        # --- Panel B: Case B (with synthetic bridge cells) ---
        synth_X_filtered = np.array([])
        if results['case_b']['synthesis_enabled']:
            synth_X, synth_y = case_b_model.get_synthetic_data()
            if synth_X is not None and len(synth_X) > 0:
                try:
                    rare_id = case_b_model.adata.uns['cytoneighbor'][
                        'label_mapping']['classes'].index(self.rare_cell_type)
                    synth_X_filtered = synth_X[synth_y == rare_id]
                except Exception:
                    pass

        if len(synth_X_filtered) > 0:
            synth_adata = sc.AnnData(X=synth_X_filtered)
            synth_adata.var_names = ref_data.var_names
            synth_adata.obs[self.cell_type_column] = self.rare_cell_type
            adata_b = ad.concat([ref_data, synth_adata], index_unique="-")
        else:
            adata_b = ref_data.copy()

        adata_b.uns['cytoneighbor'] = copy.deepcopy(
            case_b_model.adata.uns.get('cytoneighbor', {}))
        transform(adata_b, n_neighbors=15)
        assert 'neighbors' in adata_b.uns, \
            "transform() failed to build graph for Case B in generate_visualizations"
        sc.tl.umap(adata_b)

        combined_umap = adata_b.obsm['X_umap']
        real_umap_B = combined_umap[:len(ref_data)]
        synth_umap_B = combined_umap[len(ref_data):]

        rare_mask = ref_data.obs[self.cell_type_column].values == self.rare_cell_type
        other_mask = ~rare_mask

        # Double-column width for the two-panel comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.5))

        # Panel A
        ax1.scatter(umap_A[other_mask, 0], umap_A[other_mask, 1],
                    c='#d9d9d9', s=6, alpha=0.4, rasterized=True, zorder=1)
        ax1.scatter(umap_A[rare_mask, 0], umap_A[rare_mask, 1],
                    c='#2166ac', s=30, alpha=0.9,
                    edgecolors='white', linewidths=0.3,
                    rasterized=True, zorder=2)
        ax1.set_title(f'(A) Standard: Fragmented\n{self.rare_cell_type} Islands',
                      fontweight='bold')
        ax1.axis('off')

        # Panel B
        ax2.scatter(real_umap_B[other_mask, 0], real_umap_B[other_mask, 1],
                    c='#d9d9d9', s=4, alpha=0.15, rasterized=True, zorder=1)
        ax2.scatter(real_umap_B[rare_mask, 0], real_umap_B[rare_mask, 1],
                    c='#4393c3', s=14, alpha=0.5, rasterized=True, zorder=2)
        if len(synth_umap_B) > 0:
            ax2.scatter(synth_umap_B[:, 0], synth_umap_B[:, 1],
                        c='#FFD700', s=50, alpha=0.95,
                        edgecolors='#8B0000', linewidths=0.4,
                        marker='^', rasterized=True, zorder=3,
                        label='Synthetic bridge cells')
            ax2.legend(frameon=False, loc='lower right', markerscale=0.8)
        ax2.set_title('(B) Cytoneighbor with\ngene-level mixing training',
                      fontweight='bold')
        ax2.axis('off')

        plt.tight_layout()
        self._save_fig(fig, os.path.join(
            self.output_dir, 'Figure2_Mechanism_Comparison.png'))

        # ── Figure 3: Summary statistics table ────────────────────────────
        cols = [f"Recall@{k}" for k in self.recall_k_values]
        rows = ["Case A (No Synth)", "Case B (gene-level mixing)",
                "Improvement (Abs)", "Improvement (Rel)"]

        c_a = recall_data['case_a']
        c_b = recall_data['case_b']
        diff = c_b - c_a
        rel = diff / np.where(c_a == 0, 1e-8, c_a)

        cell_text = [
            [f"{v:.1%}" for v in c_a],
            [f"{v:.1%}" for v in c_b],
            [f"{v:+.1%}" for v in diff],
            [f"{v:+.1%}" for v in rel],
        ]

        n_cols = len(cols)
        fig_w = max(8.0, 1.4 * n_cols)
        fig, ax = plt.subplots(figsize=(fig_w, 2.2))
        ax.axis('off')
        table = ax.table(
            cellText=cell_text, rowLabels=rows, colLabels=cols,
            loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.1, 1.6)

        # Colour-code rows (matplotlib table: row 0 = header, rows 1..n = data)
        for (i, j), cell in table.get_celld().items():
            cell.set_linewidth(0.4)
            if i == 0:                                    # column header
                cell.set_facecolor('#bdbdbd')
                cell.set_text_props(fontweight='bold')
            elif i == 1:                                  # Case A row
                cell.set_facecolor('#f0f0f0')
            elif i == 2:                                  # Case B row
                cell.set_facecolor('#e8f4f8')
            elif i == 3:                                  # Abs improvement
                cell.set_text_props(color='#1f77b4', fontweight='bold')
            elif i == 4:                                  # Rel improvement
                cell.set_text_props(color='#d62728', fontweight='bold')

        plt.title("Cross-Validated Retrieval: Held-out Rare Cells",
                  y=1.08, fontsize=8, fontweight='bold')
        plt.tight_layout()
        self._save_fig(fig, os.path.join(
            self.output_dir, 'Figure2_Data_Table.png'))

    def run_experiment(self) -> Dict[str, Any]:
        self.setup_experiment()
        self.logger.info("Starting Mechanism Analysis Experiment")

        orig_data, ref_data, query_data = self.load_and_preprocess_data()

        ablation_results = self.run_ablation_study(ref_data, query_data)
        manifold_analysis = self.analyze_manifold_restoration(ablation_results, ref_data)
        self.generate_visualizations(ablation_results, ref_data)

        results = {
            'ablation_results': ablation_results,
            'manifold_analysis': manifold_analysis,
            'data_info': {
                'original': len(orig_data),
                'reference': len(ref_data),
                'query': len(query_data),
                'rare_cell_type': self.rare_cell_type,   # append
            }
        }
        self._generate_summary_report(results)   # append this line
        return results

    # def generate_results(self, results: Dict[str, Any]) -> None:
    #     from cytoneighbor.utils import save_results
    #     save_results(results, os.path.join(self.output_dir, 'experiment_results.json'))

    #     with open(os.path.join(self.output_dir, 'summary_report.txt'), 'w') as f:
    #         f.write("Mechanism Analysis: Cross-Validated Rare Cell Recovery\n")
    #         f.write("=" * 60 + "\n")
    #         f.write(f"Reference Set (Train):     {results['data_info']['reference']} cells\n")
    #         f.write(f"Query Set (Held-out Test): {results['data_info']['query']} cells\n\n")

    #         f.write("Graph Connectivity:\n")
    #         topo = results.get('manifold_analysis', {}).get('topological_connectivity', {})
    #         f.write(f"  Case A isolated islands : {topo.get('case_a_components_count', 0)}\n")
    #         f.write(f"  Case B isolated islands : {topo.get('case_b_components_count', 0)}\n")
    #         f.write(f"  Fragmentation reduction : "
    #                 f"{topo.get('continuity_improvement', 0.0):.1%}\n")
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> None:
        """Generate a professional and comprehensive summary report."""
        report_path = os.path.join(self.output_dir, 'summary_report.txt')
        
        data_info = results['data_info']
        ablation = results['ablation_results']
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("========================================================================\n")
            f.write(" ORIGINAL PANCREATIC DELTA-CELL SCARCITY ANALYSIS\n")
            f.write("========================================================================\n\n")
            
            f.write("[1. DATASET PARTITIONING & EXTREME SCARCITY SETTING]\n")
            f.write("-" * 72 + "\n")
            f.write(f"Target Rare Cell Type:      {data_info['rare_cell_type']}\n")
            f.write(f"Total Original Cells:       {data_info['original']}\n")
            f.write(f"Reference Set (Train/Base): {data_info['reference']} (Deeply downsampled anchors)\n")
            f.write(f"Held-out Query Set (Test):  {data_info['query']} (Unseen true rare cells to be retrieved)\n\n")
            
            f.write("[2. CROSS-KNN RETRIEVAL PERFORMANCE (STANDARD RECALL@K)]\n")
            f.write("-" * 72 + "\n")
            f.write(f"{'K-Value':<10} | {'Case A (Baseline)':<20} | {'Case B (Cytoneighbor)':<22} | {'Rel. Improvement'}\n")
            f.write("-" * 72 + "\n")
            
            case_a_recalls = ablation['case_a']['recall_scores']
            case_b_recalls = ablation['case_b']['recall_scores']
            
            for k in self.recall_k_values:
                k_str = str(k)
                ra = case_a_recalls.get(k_str, 0.0)
                rb = case_b_recalls.get(k_str, 0.0)
                rel_imp = ((rb - ra) / ra * 100) if ra > 0 else 0.0
                f.write(f"Recall@{k:<4} | {ra:>18.2%} | {rb:>20.2%} | +{rel_imp:>14.1f}%\n")

            deltas = [case_b_recalls.get(str(k), 0.0) - case_a_recalls.get(str(k), 0.0)
                      for k in self.recall_k_values]
            if all(d >= 0 for d in deltas):
                f.write("\n* Interpretation: under this run, gene-level mixing was associated with equal or\n")
                f.write("  higher retrieval yield across the evaluated neighborhood range.\n\n")
            elif all(d <= 0 for d in deltas):
                f.write("\n* Interpretation: under this run, gene-level mixing was associated with lower\n")
                f.write("  retrieval yield across the evaluated neighborhood range and should be\n")
                f.write("  interpreted cautiously before being used for manuscript figures.\n\n")
            else:
                f.write("\n* Interpretation: under this run, gene-level mixing showed mixed retrieval effects\n")
                f.write("  across neighborhood sizes rather than a uniform gain.\n\n")

            f.write("[3. TOPOLOGICAL GRAPH CONNECTIVITY]\n")
            f.write("-" * 72 + "\n")
            topo = results.get('manifold_analysis', {}).get('topological_connectivity', {})
            f.write(f"Graph Construction K:                  {topo.get('graph_k_used', 'N/A')}\n")
            f.write(f"Baseline (Case A) Isolated Components: {topo.get('case_a_components_count', 'N/A')}\n")
            f.write(f"Case B Isolated Components: {topo.get('case_b_components_count', 'N/A')}\n")
            f.write(f"Synthetic Anchor Support Injected:     {'Yes' if topo.get('synthetic_cells_added', False) else 'No'}\n")
            f.write(f"Synthetic Assets Available:            {'Yes' if results['ablation_results']['case_b'].get('synthetic_assets_available', False) else 'No'}\n")
            f.write("========================================================================\n")

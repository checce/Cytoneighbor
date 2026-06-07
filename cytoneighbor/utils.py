"""
Utility functions for the Cytoneighbor Experimental Framework.
Provides data loading, preprocessing, metrics calculation, and visualization capabilities.
"""

import os
import yaml
import json
import logging
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import f1_score
import warnings

# Configure scanpy
sc.settings.verbosity = 1
warnings.filterwarnings('ignore')


# Project root resolution for robust path handling
def _get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent

class ConfigManager:
    """Manages YAML configuration files and parameter access."""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = str(_get_project_root() / "configs" / "config.yaml")
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
        except UnicodeDecodeError as e:
            # Try with different encoding
            try:
                with open(self.config_path, 'r', encoding='gbk') as f:
                    config = yaml.safe_load(f)
                return config
            except Exception:
                raise ValueError(f"Configuration file encoding error: {e}")
    
    def get_experiment_config(self, experiment_name: str) -> Dict[str, Any]:
        """Get configuration for specific experiment."""
        if experiment_name not in self.config.get('experiments', {}):
            raise KeyError(f"Experiment '{experiment_name}' not found in configuration")
        return self.config['experiments'][experiment_name]
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.config.get('model_params', {})
    
    def get_global_config(self) -> Dict[str, Any]:
        """Get global configuration."""
        return self.config.get('global', {})


class DataLoader:
    """Handles data loading, validation, and preprocessing for experiments."""
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = str(_get_project_root() / "data")
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)
    
    def load_dataset(self, dataset_path: str, required_columns: Optional[List[str]] = None) -> ad.AnnData:
        """
        Load dataset from .h5ad file with validation.
        
        Args:
            dataset_path: Path to .h5ad file (relative to base_path)
            required_columns: List of required columns in .obs
            
        Returns:
            AnnData object
        """
        full_path = self.base_path / dataset_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Dataset not found: {full_path}")
        
        try:
            adata = sc.read_h5ad(full_path)
            self.logger.info(f"Loaded dataset: {adata.n_obs} cells, {adata.n_vars} genes")
            
            # Check if expression matrix is empty or has wrong dimensions
            if adata.X is None or adata.X.size == 0 or len(adata.X.shape) == 0:
                self.logger.warning("Main expression matrix (adata.X) is empty or invalid")
                
                # Try to find expression data in layers
                if adata.layers and len(adata.layers) > 0:
                    self.logger.info(f"Available layers: {list(adata.layers.keys())}")
                    
                    # Common layer names for expression data
                    expression_layer_candidates = ['counts', 'raw', 'X', 'data', 'expression']
                    
                    for layer_name in expression_layer_candidates:
                        if layer_name in adata.layers:
                            self.logger.info(f"Using layer '{layer_name}' as main expression matrix")
                            adata.X = adata.layers[layer_name].copy()
                            break
                    else:
                        # Use the first available layer
                        first_layer = list(adata.layers.keys())[0]
                        self.logger.info(f"Using first available layer '{first_layer}' as main expression matrix")
                        adata.X = adata.layers[first_layer].copy()
                else:
                    raise ValueError("No valid expression data found in adata.X or layers")
            
            # Validate expression matrix dimensions
            if len(adata.X.shape) != 2:
                raise ValueError(f"Expression matrix has invalid shape: {adata.X.shape}")
            
            if adata.X.shape[0] != adata.n_obs or adata.X.shape[1] != adata.n_vars:
                self.logger.warning(f"Expression matrix shape {adata.X.shape} doesn't match expected ({adata.n_obs}, {adata.n_vars})")
            
            self.logger.info(f"Final expression matrix shape: {adata.X.shape}")
            
            if required_columns:
                self.validate_columns(adata, required_columns)
            
            return adata
            
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {full_path}: {e}")
    
    def validate_columns(self, adata: ad.AnnData, required_columns: List[str]) -> bool:
        """Validate presence of required columns in adata.obs."""
        missing_columns = [col for col in required_columns if col not in adata.obs.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return True
    
    def extract_cell_subgroups(self, adata: ad.AnnData, cell_type_column: str, 
                              target_types: List[str]) -> ad.AnnData:
        """Extract specific cell type subgroups with robust error handling."""
        if cell_type_column not in adata.obs.columns:
            raise KeyError(f"Cell type column '{cell_type_column}' not found")
        
        # Get available cell types
        available_types = set(adata.obs[cell_type_column].unique())
        requested_types = set(target_types)
        
        # Check which types are missing
        missing_types = requested_types - available_types
        existing_types = requested_types & available_types
        
        if missing_types:
            self.logger.warning(f"Warning: Requested types {list(missing_types)} not found in dataset. "
                              f"Available types are: {list(available_types)}")
        
        if not existing_types:
            raise ValueError(f"No cells found for any target types: {target_types}. "
                           f"Available types: {list(available_types)}")
        
        # Only extract existing types
        mask = adata.obs[cell_type_column].isin(list(existing_types))
        subset = adata[mask].copy()
        
        self.logger.info(f"Extracted {subset.n_obs} cells from {len(existing_types)} cell types: {list(existing_types)}")
        
        return subset
    
    def downsample_rare_cells(self, adata: ad.AnnData, cell_type_column: str, 
                             rare_cell_type: str, target_fraction: float) -> ad.AnnData:
        """Downsample rare cells to simulate search failure."""
        if rare_cell_type not in adata.obs[cell_type_column].values:
            raise ValueError(f"Rare cell type '{rare_cell_type}' not found")
        
        rare_mask = adata.obs[cell_type_column] == rare_cell_type
        rare_indices = np.where(rare_mask)[0]
        
        n_keep = max(1, int(len(rare_indices) * target_fraction))
        keep_indices = np.random.choice(rare_indices, size=n_keep, replace=False)
        
        # Create new mask: keep all non-rare cells + downsampled rare cells
        final_mask = ~rare_mask
        final_mask[keep_indices] = True
        
        downsampled = adata[final_mask].copy()
        self.logger.info(f"Downsampled {rare_cell_type}: {len(rare_indices)} -> {n_keep} cells")
        
        return downsampled
    
    def split_reference_query(self, adata: ad.AnnData, condition_column: str, 
                             healthy_condition: str = "healthy", 
                             disease_condition: str = "disease") -> Tuple[ad.AnnData, ad.AnnData]:
        """Split data into reference (healthy) and query (disease) sets."""
        if condition_column not in adata.obs.columns:
            raise KeyError(f"Condition column '{condition_column}' not found")
        
        healthy_mask = adata.obs[condition_column] == healthy_condition
        disease_mask = adata.obs[condition_column] == disease_condition
        
        if not healthy_mask.any():
            raise ValueError(f"No healthy samples found with condition '{healthy_condition}'")
        if not disease_mask.any():
            raise ValueError(f"No disease samples found with condition '{disease_condition}'")
        
        reference = adata[healthy_mask].copy()
        query = adata[disease_mask].copy()
        
        self.logger.info(f"Split data: {reference.n_obs} reference, {query.n_obs} query cells")
        
        return reference, query


class MetricsCalculator:
    """Calculates evaluation metrics for single-cell analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_ilisi(self, adata: ad.AnnData, batch_key: str, 
                       neighbors_key: str = 'neighbors', k: int = 30) -> float:
        """
        Calculate integration Local Inverse Simpson Index (iLISI).
        Higher values indicate better integration.
        """
        # try:
        #     # Use scanpy's implementation if available, otherwise implement manually
        #     if hasattr(sc.metrics, 'ilisi'):
        #         return sc.metrics.ilisi(adata, batch_key=batch_key, type_='embed', k0=k)
        #     else:
        #         return self._calculate_lisi(adata, batch_key, neighbors_key, k, metric_type='integration')
        # except Exception as e:
        #     self.logger.warning(f"iLISI calculation failed: {e}")
        #     return 0.0
        try:
            return self._calculate_lisi(adata, batch_key, neighbors_key, k, metric_type='integration')
        except Exception as e:
            self.logger.warning(f"iLISI calculation failed: {e}")
            return 0.0
    
    def calculate_clisi(self, adata: ad.AnnData, label_key: str, 
                       neighbors_key: str = 'neighbors', k: int = 30) -> float:
        """
        Calculate cell-type Local Inverse Simpson Index (cLISI).
        Higher values indicate worse cell type mixing (good for cell type preservation).
        """
        # try:
        #     if hasattr(sc.metrics, 'clisi'):
        #         return sc.metrics.clisi(adata, label_key=label_key, type_='embed', k0=k)
        #     else:
        #         return self._calculate_lisi(adata, label_key, neighbors_key, k, metric_type='clustering')
        # except Exception as e:
        #     self.logger.warning(f"cLISI calculation failed: {e}")
        #     return 0.0
        try:
            return self._calculate_lisi(adata, label_key, neighbors_key, k, metric_type='clustering')
        except Exception as e:
            self.logger.warning(f"cLISI calculation failed: {e}")
            return 0.0
    
    def _calculate_lisi(self, adata: ad.AnnData, key: str, neighbors_key: str, 
                       k: int, metric_type: str) -> float:
        """Manual LISI calculation implementation with improved robustness."""
        # Check for neighbor connectivity matrix
        conn_key = f"{neighbors_key}_connectivities"
        if conn_key not in adata.obsp:
            # Try alternative keys
            possible_keys = ['connectivities', 'neighbors_connectivities', 'X_neighbors_connectivities']
            conn_key = None
            for key_name in possible_keys:
                if key_name in adata.obsp:
                    conn_key = key_name
                    break
            
            if conn_key is None:
                raise ValueError(f"Connectivity matrix not found. Available keys: {list(adata.obsp.keys())}")
        
        conn = adata.obsp[conn_key]
        labels = adata.obs[key].values
        
        # Convert to dense if sparse
        if hasattr(conn, 'toarray'):
            conn_dense = conn.toarray()
        else:
            conn_dense = conn
        
        lisi_scores = []
        
        for i in range(adata.n_obs):
            # Get connectivity row for cell i
            row = conn_dense[i]
            
            # Get top k neighbors (excluding self if present)
            if row[i] > 0:  # Self-connection exists
                row_copy = row.copy()
                row_copy[i] = 0  # Remove self-connection
                top_k_indices = np.argsort(row_copy)[-k:]
            else:
                top_k_indices = np.argsort(row)[-k:]
            
            # Get neighbor labels
            neighbor_labels = labels[top_k_indices]
            
            # Calculate Simpson index
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            if len(unique_labels) == 0:
                lisi_scores.append(1.0)
                continue
                
            proportions = counts / len(neighbor_labels)
            simpson_index = np.sum(proportions ** 2)
            lisi_score = 1 / simpson_index if simpson_index > 0 else 1.0
            lisi_scores.append(lisi_score)
        
        return np.mean(lisi_scores)
    
    def calculate_recall_at_k(self, similarity_matrix: np.ndarray, 
                             true_labels: np.ndarray, k: int = 10) -> float:
        """
        Calculate Recall@k for neighbor retrieval.
        
        Args:
            similarity_matrix: [n_queries, n_candidates] similarity scores
            true_labels: [n_total] true labels for all cells
            k: Number of top neighbors to consider
            
        Returns:
            Average recall@k across all queries
        """
        if similarity_matrix.shape[0] == 0:
            return 0.0
        
        n_queries = similarity_matrix.shape[0]
        recalls = []
        
        for i in range(n_queries):
            # Get top k neighbors
            top_k_indices = np.argsort(similarity_matrix[i])[-k:]
            
            # Count how many have the same label as query
            query_label = true_labels[i]
            same_label_count = np.sum(true_labels[top_k_indices] == query_label)
            
            # Calculate recall (excluding self if present)
            total_same_label = np.sum(true_labels == query_label) - 1  # Exclude self
            recall = same_label_count / max(1, total_same_label)
            recalls.append(min(1.0, recall))  # Cap at 1.0
        
        return np.mean(recalls)
    
    def calculate_f1_score(self, predictions: np.ndarray, true_labels: np.ndarray) -> float:
        """Calculate F1 score for classification."""
        try:
            return f1_score(true_labels, predictions, average='weighted')
        except Exception as e:
            self.logger.warning(f"F1 score calculation failed: {e}")
            return 0.0
    
    def calculate_confidence_scores(self, model_outputs: Dict, 
                                   reference_embeddings: np.ndarray) -> np.ndarray:
        """Calculate confidence scores from model outputs."""
        # Extract confidence scores from model outputs
        if 'confidence' in model_outputs:
            return model_outputs['confidence']
        elif 'probabilities' in model_outputs:
            # Use max probability as confidence
            return np.max(model_outputs['probabilities'], axis=1)
        else:
            # Default: uniform confidence
            return np.ones(len(reference_embeddings))
    
    def calculate_confidence_interval(self, scores: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for a set of scores.
        
        Args:
            scores: Array of scores
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(scores) == 0:
            return (0.0, 0.0)
        
        alpha = 1 - confidence
        n = len(scores)
        mean = np.mean(scores)
        std_err = stats.sem(scores)  # Standard error of the mean
        
        # Use t-distribution for small samples, normal for large samples
        if n < 30:
            t_val = stats.t.ppf(1 - alpha/2, n - 1)
            margin_error = t_val * std_err
        else:
            z_val = stats.norm.ppf(1 - alpha/2)
            margin_error = z_val * std_err
        
        return (mean - margin_error, mean + margin_error)
    
    def perform_statistical_test(self, scores1: np.ndarray, scores2: np.ndarray, 
                                test_type: str = 'ttest') -> Dict[str, float]:
        """
        Perform statistical significance test between two sets of scores.
        
        Args:
            scores1: First set of scores
            scores2: Second set of scores
            test_type: Type of test ('ttest', 'mannwhitney', 'wilcoxon')
            
        Returns:
            Dictionary with test statistics and p-value
        """
        if len(scores1) == 0 or len(scores2) == 0:
            return {'statistic': 0.0, 'p_value': 1.0, 'test_type': test_type}
        
        try:
            if test_type == 'ttest':
                # Independent t-test
                statistic, p_value = stats.ttest_ind(scores1, scores2)
            elif test_type == 'mannwhitney':
                # Mann-Whitney U test (non-parametric)
                statistic, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
            elif test_type == 'wilcoxon':
                # Wilcoxon signed-rank test (paired)
                if len(scores1) != len(scores2):
                    raise ValueError("Wilcoxon test requires equal length arrays")
                statistic, p_value = stats.wilcoxon(scores1, scores2)
            else:
                raise ValueError(f"Unknown test type: {test_type}")
            
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'test_type': test_type,
                'significant': p_value < 0.05
            }
        except Exception as e:
            self.logger.warning(f"Statistical test failed: {e}")
            return {'statistic': 0.0, 'p_value': 1.0, 'test_type': test_type, 'significant': False}
    
    def identify_eroded_cells(self, reference_confidence: np.ndarray, 
                             query_confidence: np.ndarray, 
                             threshold: float = 0.1) -> np.ndarray:
        """
        Identify cells with eroded identity based on confidence score comparison.
        
        Args:
            reference_confidence: Confidence scores from reference (healthy) data
            query_confidence: Confidence scores from query (disease) data  
            threshold: Threshold for identifying erosion (confidence drop)
            
        Returns:
            Boolean array indicating which query cells have eroded identity
        """
        if len(query_confidence) == 0:
            return np.array([], dtype=bool)
        
        # Calculate reference baseline (mean confidence)
        reference_baseline = np.mean(reference_confidence) if len(reference_confidence) > 0 else 0.8
        
        # Identify eroded cells as those with confidence significantly below baseline
        confidence_drop = reference_baseline - query_confidence
        eroded_mask = confidence_drop > threshold
        
        self.logger.info(f"Identified {np.sum(eroded_mask)} eroded cells out of {len(query_confidence)} "
                        f"({np.mean(eroded_mask):.1%}) using threshold {threshold}")
        
        return eroded_mask


class VisualizationEngine:
    """Generates publication-ready visualizations for experimental results."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.figure_size = self.config.get('figure_size', [10, 8])
        self.dpi = self.config.get('dpi', 300)
        self.color_palette = self.config.get('color_palette', 'Set1')
        self.font_size = self.config.get('font_size', 12)
        
        # Set matplotlib parameters
        plt.rcParams.update({
            'font.size': self.font_size,
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'figure.figsize': self.figure_size
        })
    
    def _get_colors(self, n_colors: int) -> np.ndarray:
        """Get colors from colormap with compatibility for different Matplotlib versions."""
        try:
            # Try new API first (Matplotlib 3.5+)
            cmap = plt.colormaps[self.color_palette]
        except (AttributeError, KeyError):
            try:
                # Fallback to older API
                cmap = plt.cm.get_cmap(self.color_palette)
            except AttributeError:
                # Final fallback to pyplot
                cmap = plt.get_cmap(self.color_palette)
        
        return cmap(np.linspace(0, 1, n_colors))
    
    def plot_umap_with_boundaries(self, embeddings: np.ndarray, 
                                 labels: np.ndarray, title: str) -> plt.Figure:
        """Create UMAP visualization with clear cell type boundaries."""
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Create scatter plot
        unique_labels = np.unique(labels)
        try:
            # Try new API first (Matplotlib 3.5+)
            cmap = plt.colormaps[self.color_palette]
        except (AttributeError, KeyError):
            # Fallback to older API
            cmap = plt.cm.get_cmap(self.color_palette)
        colors = cmap(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                      c=[colors[i]], label=str(label), alpha=0.7, s=20)
        
        ax.set_title(title, fontsize=self.font_size + 2)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def plot_metric_comparison_with_ci(self, metrics_data: Dict[str, Dict], 
                                      statistical_tests: Dict[str, Dict]) -> plt.Figure:
        """Generate comparative metric plots with confidence intervals and significance."""
        # Prepare data for plotting
        methods = list(metrics_data.keys())
        metric_names = []
        
        # Find common metrics across methods
        for method_metrics in metrics_data.values():
            for metric in method_metrics.keys():
                if not metric.endswith('_CI') and metric not in metric_names:
                    metric_names.append(metric)
        
        # Filter to main metrics
        main_metrics = [m for m in metric_names if m in ['Accuracy', 'F1_Score', 'ASW', 'cLISI', 'iLISI']]
        
        fig, axes = plt.subplots(1, len(main_metrics), figsize=(len(main_metrics) * 5, 6))
        if len(main_metrics) == 1:
            axes = [axes]
        
        colors = self._get_colors(len(methods))
        
        for i, metric_name in enumerate(main_metrics):
            ax = axes[i]
            
            # Collect values and CIs for this metric
            values = []
            ci_lower = []
            ci_upper = []
            method_labels = []
            
            for method in methods:
                if metric_name in metrics_data[method]:
                    values.append(metrics_data[method][metric_name])
                    method_labels.append(method)
                    
                    # Get confidence interval if available
                    ci_key = f"{metric_name}_CI"
                    if ci_key in metrics_data[method]:
                        ci_low, ci_high = metrics_data[method][ci_key]
                        ci_lower.append(values[-1] - ci_low)
                        ci_upper.append(ci_high - values[-1])
                    else:
                        ci_lower.append(0)
                        ci_upper.append(0)
            
            if not values:
                continue
            
            # Create bar plot with error bars
            x_pos = np.arange(len(method_labels))
            bars = ax.bar(x_pos, values, yerr=[ci_lower, ci_upper], 
                         capsize=5, color=colors[:len(method_labels)], alpha=0.7)
            
            ax.set_title(f'{metric_name}')
            ax.set_ylabel('Score')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(method_labels, rotation=45, ha='right')
            
            # Add value labels on bars
            for j, (bar, value) in enumerate(zip(bars, values)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci_upper[j] + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            # Add significance annotations
            if metric_name.lower() in statistical_tests:
                y_max = max([v + ci_u for v, ci_u in zip(values, ci_upper)])
                y_offset = y_max * 0.1
                
                comparisons = statistical_tests[metric_name.lower()]
                annotation_y = y_max + y_offset
                
                for comparison, test_result in comparisons.items():
                    if test_result['significant']:
                        method1, method2 = comparison.split('_vs_')
                        if method1 in method_labels and method2 in method_labels:
                            idx1 = method_labels.index(method1)
                            idx2 = method_labels.index(method2)
                            
                            # Draw significance line
                            ax.plot([idx1, idx2], [annotation_y, annotation_y], 'k-', linewidth=1)
                            ax.plot([idx1, idx1], [annotation_y - y_offset*0.1, annotation_y], 'k-', linewidth=1)
                            ax.plot([idx2, idx2], [annotation_y - y_offset*0.1, annotation_y], 'k-', linewidth=1)
                            
                            # Add significance marker
                            p_val = test_result['p_value']
                            if p_val < 0.001:
                                sig_text = '***'
                            elif p_val < 0.01:
                                sig_text = '**'
                            elif p_val < 0.05:
                                sig_text = '*'
                            else:
                                sig_text = 'ns'
                            
                            ax.text((idx1 + idx2) / 2, annotation_y + y_offset*0.1, 
                                   sig_text, ha='center', va='bottom', fontsize=12, fontweight='bold')
                            
                            annotation_y += y_offset * 0.5
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_recall_curves(self, recall_data: Dict[str, np.ndarray]) -> plt.Figure:
        """Plot recall curves for different k values."""
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        for method_name, recall_values in recall_data.items():
            k_values = np.arange(1, len(recall_values) + 1)
            ax.plot(k_values, recall_values, marker='o', label=method_name, linewidth=2)
        
        ax.set_xlabel('k (Number of Neighbors)')
        ax.set_ylabel('Recall@k')
        ax.set_title('Recall@k Performance Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_confidence_distributions(self, healthy_scores: np.ndarray, 
                                    disease_scores: np.ndarray) -> plt.Figure:
        """Plot confidence score distributions for healthy vs disease cells."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram comparison
        ax1.hist(healthy_scores, bins=30, alpha=0.7, label='Healthy', color='green', density=True)
        ax1.hist(disease_scores, bins=30, alpha=0.7, label='Disease', color='red', density=True)
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Density')
        ax1.set_title('Confidence Score Distributions')
        ax1.legend()
        
        # Box plot comparison
        ax2.boxplot([healthy_scores, disease_scores], labels=['Healthy', 'Disease'])
        ax2.set_ylabel('Confidence Score')
        ax2.set_title('Confidence Score Comparison')
        
        plt.tight_layout()
        return fig
    
    def plot_spatial_predictions(self, coordinates: np.ndarray, 
                               predictions: np.ndarray) -> plt.Figure:
        """Plot spatial predictions on tissue coordinates."""
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Create scatter plot with predictions as colors
        unique_preds = np.unique(predictions)
        colors = self._get_colors(len(unique_preds))
        
        for i, pred in enumerate(unique_preds):
            mask = predictions == pred
            ax.scatter(coordinates[mask, 0], coordinates[mask, 1], 
                      c=[colors[i]], label=str(pred), alpha=0.8, s=50)
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Spatial Cell Type Predictions')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def save_figure(self, figure: plt.Figure, output_path: str, dpi: int = 300) -> None:
        """Save figure to file with specified DPI."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        figure.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(figure)


class BaseExperiment:
    """Base class for all experiments with common functionality."""
    
    def __init__(self, config: Dict[str, Any], data_loader: DataLoader, 
                 metrics_calc: MetricsCalculator, viz_engine: VisualizationEngine):
        self.config = config
        self.data_loader = data_loader
        self.metrics_calc = metrics_calc
        self.viz_engine = viz_engine
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results = {}
        
    def setup_experiment(self) -> None:
        """Setup experiment-specific configurations."""
        pass
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the experiment and return results."""
        raise NotImplementedError("Subclasses must implement run_experiment")
    
    def generate_results(self, results: Dict[str, Any]) -> None:
        """Generate visualizations and save results."""
        pass
    
    def cleanup(self) -> None:
        """Cleanup resources after experiment."""
        pass


# Utility functions for working with Cytoneighbor model
def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('cytoneighbor_experiments.log')
        ]
    )


def ensure_output_dir(output_dir: str) -> None:
    """Ensure output directory exists."""
    os.makedirs(output_dir, exist_ok=True)


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save experiment results to file."""
    ensure_output_dir(os.path.dirname(output_path))
    
    # Convert numpy arrays and other non-serializable types to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_for_json(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def load_results(input_path: str) -> Dict[str, Any]:
    """Load experiment results from file."""
    with open(input_path, 'r') as f:
        return json.load(f)
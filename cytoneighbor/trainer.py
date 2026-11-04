"""
Training utilities and wrapper for the Cytoneighbor model.
Provides a clean interface to the existing Cytoneighbor implementation.
"""

import os
import copy  # [FIX Issue #7] Used in predict_neighbors to avoid shared nn.Module references
import logging
import numpy as np
import pandas as pd
import anndata as ad
from typing import Dict, Any, Optional, Tuple, List
import torch

# Import the existing Cytoneighbor model functions
from cytoneighbor.model import fit, transform, annotate, load_data, save_model, load_model


class CytoneighborWrapper:
    """
    Wrapper class for the Cytoneighbor model providing a clean interface
    to the existing implementation without modification.
    """

    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the Cytoneighbor wrapper.

        Args:
            model_config: Configuration dictionary containing model parameters
        """
        self.config = model_config
        self.logger = logging.getLogger(__name__)
        self.adata = None
        self.is_trained = False

        # ===[CRITICAL FIX START] å¨ææåææåæ°ï¼é²æ­¢ Override è¢«æ¦æ?===
        self.model_params = model_config.copy()

        # 1. å¤çç¹æ®çé®åæ å°?(å¤é¨éç½®æä»¶å?learning_rateï¼åºå±?model.py éè¦?lr)
        if 'learning_rate' in self.model_params:
            self.model_params['lr'] = float(self.model_params.pop('learning_rate'))

        # 2. [å³é®ä¿®å¤] å¤ç batch_size çé®åæ å°?(éç½®æä»¶å?batch_sizeï¼åºå±å« batch_size_override)
        if 'batch_size' in self.model_params:
            self.model_params['batch_size_override'] = int(self.model_params.pop('batch_size'))

        # 3. ç¡®ä¿åºç¡æ´æ°ç±»åæ­£ç¡®
        if 'd_model' in self.model_params: self.model_params['d_model'] = int(self.model_params['d_model'])
        if 'depth' in self.model_params: self.model_params['depth'] = int(self.model_params['depth'])
        if 'hidden' in self.model_params: self.model_params['hidden'] = int(self.model_params['hidden'])
        if 'epochs' in self.model_params: self.model_params['epochs'] = int(self.model_params['epochs'])

        # 4. [æ°å¢ä¿®å¤] å¼ºå¶è½¬æ¢æµ®ç¹æ°ç±»åï¼é²æ­¢ YAML å°?'1e-4' éè¯¯è§£æä¸ºå­ç¬¦ä¸²
        float_keys = ['weight_decay', 'dropout', 'rank_margin', 'lambda_rank', 'neg_cache_pick_p']
        for k in float_keys:
            if k in self.model_params:
                self.model_params[k] = float(self.model_params[k])

        # 5. [å®å¨ååº] è¿æ»¤æä»»ä½?fit() ä¸è®¤è¯çåæ°ï¼é²æ­¢éç½®åéå¼ååºå±æ¥é?
        import inspect
        from cytoneighbor.model import fit
        valid_keys = inspect.signature(fit).parameters.keys()
        self.model_params = {k: v for k, v in self.model_params.items() if k in valid_keys}
        # === [CRITICAL FIX END] ===

    def prepare_data(self, adata: ad.AnnData, cell_type_column: str) -> ad.AnnData:
        """
        Prepare AnnData object for training.

        Args:
            adata: AnnData object containing single-cell data
            cell_type_column: Column name containing cell type labels

        Returns:
            Prepared AnnData object
        """
        if cell_type_column not in adata.obs.columns:
            raise KeyError(f"Cell type column '{cell_type_column}' not found in adata.obs")

        # Create a copy to avoid modifying original data
        adata_copy = adata.copy()

        # Extract expression data and labels
        X = adata_copy.X
        if hasattr(X, 'toarray'):  # Handle sparse matrices
            X = X.toarray()

        labels = adata_copy.obs[cell_type_column].values

        # Load data into the Cytoneighbor format
        load_data(adata_copy, data=X, labels=labels)

        self.adata = adata_copy
        self.logger.info(f"Data prepared: {adata_copy.n_obs} cells, {adata_copy.n_vars} genes")

        return adata_copy

    def train(self, adata: ad.AnnData, cell_type_column: str,
              synthesis_enabled: bool = True, model_save_path: str = None,
              **synthesis_params) -> None:
        """
        Train the Cytoneighbor model.

        Args:
            adata: AnnData object containing training data
            cell_type_column: Column name containing cell type labels
            synthesis_enabled: Whether to enable gene-level mixing synthesis
            model_save_path: Path to save/load the model
            **synthesis_params: Additional synthesis parameters (synth_mult_per_dt, synth_cap_per_dt, etc.)
        """
        # Only try to load existing model if model_save_path is explicitly provided
        if model_save_path and os.path.exists(model_save_path):
            self.logger.info(f"Found existing model at {model_save_path}, loading...")
            try:
                # Prepare data first (needed for model structure)
                if self.adata is None:
                    self.prepare_data(adata, cell_type_column)

                # Load the pre-trained model
                self.adata = load_model(self.adata, model_save_path)
                self.is_trained = True
                self.logger.info("Pre-trained model loaded successfully, skipping training")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load existing model: {e}")
                self.logger.info("Proceeding with fresh training...")

        # Prepare data if not already done
        if self.adata is None:
            self.prepare_data(adata, cell_type_column)

        # Update synthesis setting
        self.model_params['synth_enable'] = synthesis_enabled

        # Add synthesis parameters to model params
        for param_name, param_value in synthesis_params.items():
            if param_name in ['synth_mult_per_dt', 'synth_cap_per_dt', 'synth_overgen_factor',
                              'synth_ratio_range', 'synth_p_three_parent', 'synth_parts_three',
                              'synth_alpha', 'synth_refresh_every']:
                self.model_params[param_name] = param_value
                self.logger.info(f"Set synthesis parameter {param_name} = {param_value}")

        self.logger.info("Starting Cytoneighbor training...")
        self.logger.info(f"Synthesis enabled: {synthesis_enabled}")
        if synthesis_params:
            self.logger.info(f"Synthesis parameters: {synthesis_params}")

        # -----------------------------------------------------------------------
        # [FIX Issue #4] Set an experiment-specific best-model checkpoint path.
        #
        # model.py fit() defaults save_best_model_path to "cynb_best.zip" in the
        # current working directory when None is passed. If multiple experiments or
        # hyperparameter searches run concurrently (or sequentially without clearing
        # the directory), their best-epoch snapshots silently overwrite one another.
        #
        # We derive a deterministic path from model_save_path so each experiment
        # writes to its own unambiguous checkpoint file. The key is removed from
        # model_params after the call (via finally) to keep the dict reusable.
        # -----------------------------------------------------------------------
        _best_model_path_key = 'save_best_model_path'
        if _best_model_path_key not in self.model_params:
            _best_path = (
                model_save_path.replace('.zip', '_best.zip')
                if model_save_path else None
            )
            if _best_path:
                self.model_params[_best_model_path_key] = _best_path
                self.logger.info(
                    f"[Fix Issue#4] Best-model checkpoint path set to: {_best_path}"
                )
        # -----------------------------------------------------------------------

        # Train the model using the existing fit function
        try:          
            self.adata = fit(
                self.adata,
                **self.model_params
            )
            self.is_trained = True
            self.logger.info("Training completed successfully")

            # Save the trained model only if path is provided
            if model_save_path:
                save_model(self.adata, model_save_path)
                self.logger.info(f"Model saved to {model_save_path}")
            else:
                self.logger.info("No save path provided, model not saved to disk")

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            # [FIX Issue #4] Remove the temporary key so model_params stays clean
            # for any subsequent calls (e.g. re-training with a different save path).
            # self.model_params.pop(_best_model_path_key, None)
            # åªæ¸é¤æ¬æ¬?train() å¨ææ³¨å¥çé®ï¼ä¸è§¦ç¢°ç¨æ·åå§éç½®ä¸­çé®
            if _best_path:  # åªææä»¬è®¾ç½®äºï¼æè´è´£æ¸é?
                self.model_params.pop(_best_model_path_key, None)

    def predict_neighbors(self, query_data: ad.AnnData, k: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict neighbors for query data.

        Args:
            query_data: AnnData object containing query cells
            k: Number of neighbors to return

        Returns:
            Tuple of (neighbor_indices, distances)
        """
        if not self.is_trained:
            raise RuntimeError("Operation failed: model state invalid")

        # Create a copy of query data and transfer model information
        query_copy = query_data.copy()

        if 'cytoneighbor' not in query_copy.uns:
            query_copy.uns['cytoneighbor'] = {}

        # [FIX Issue #7] Use copy.deepcopy so that transform()'s internal
        # model.to(device) / model.to('cpu') calls do not mutate the nn.Module
        # held by self.adata â?which would cause a device mismatch on any
        # subsequent forward pass through the original model object.
        cytoneighbor_info = self.adata.uns.get('cytoneighbor', {})
        query_copy.uns['cytoneighbor'] = copy.deepcopy({
            'model': cytoneighbor_info.get('model'),
            'feature_names': cytoneighbor_info.get('feature_names'),
            'ann': cytoneighbor_info.get('ann'),
            'preproc': cytoneighbor_info.get('preproc', {"mode": "none"}),
            'label_mapping': cytoneighbor_info.get('label_mapping'),
            'env_key': cytoneighbor_info.get('env_key')
        })

        try:
            # Use transform to build neighbor graph
            transform(query_copy, n_neighbors=k)

            # Extract neighbor information from the connectivity matrix
            if 'connectivities' in query_copy.obsp:
                conn_matrix = query_copy.obsp['connectivities']

                # Convert to neighbor indices and distances
                neighbor_indices = []
                distances = []

                for i in range(query_copy.n_obs):
                    # è·åè¯¥ç»èçç¨ å¯è¿æ¥è¡?
                    row_values = conn_matrix[i].toarray().flatten()

                    # [æ ¸å¿ä¿®å¤]ï¼åªåéé¶è¿éåº¦çææé»å±ï¼æç»"åæ°"åé»å±?
                    valid_mask = row_values > 0
                    valid_indices = np.where(valid_mask)[0]
                    valid_values = row_values[valid_mask]

                    # å¦ææ²¡æææé»å±ï¼æèææé»å±æ°å°äºè¯·æ±ç?k
                    if len(valid_indices) == 0:
                        top_k_idx = np.array([], dtype=np.int64)
                        top_k_distances = np.array([], dtype=np.float32)
                    else:
                        # ç¡®å®å®éè¦åçæ°é?
                        actual_k = min(k, len(valid_indices))
                        # å¨ææå¼åæåºï¼ä»å°å°å¤§ï¼ï¼åæå?actual_k ä¸ªï¼åç¿»è½¬ï¼ä»å¤§å°å°ï¼?
                        local_sort_idx = np.argsort(valid_values)[-actual_k:][::-1]

                        top_k_idx = valid_indices[local_sort_idx]
                        # è·ç¦» = 1.0 - è¿éåº¦
                        top_k_distances = 1.0 - valid_values[local_sort_idx]

                    neighbor_indices.append(top_k_idx)
                    distances.append(top_k_distances)

                # [æ³¨æ] ç±äºæ¯ä¸ªç»èçé»å±æ°å¯è½ä¸åï¼æäºä¸å?k ä¸ªï¼ï¼?
                # å¿é¡»è¿åå¯¹è±¡æ°ç» (dtype=object) çå½¢å¼ï¼å¦åè½¬ä¸º np.array ä¼æ¥ Ragged å¼å¸¸ã?
                # å¨ä¸æ¸¸æºå¶åæä»£ç ä¸­ï¼å¾ªç¯æ¯ for rare_idx in rare_indicesï¼éåçæ¯æ¯è¡åè¡¨ï¼
                # ä¸éè¿ if rare_idx < len(neighbor_indices) å¤æ­äºé¿åº¦ï¼æä»¥å¼å®¹ã?
                return np.array(neighbor_indices, dtype=object), np.array(distances, dtype=object)
            else:
                raise RuntimeError("Neighbor graph not found after transform")

        except Exception as e:
            self.logger.error(f"Neighbor prediction failed: {e}")
            n_cells = query_data.n_obs
            # ä¿æä¸æ­£å¸¸è·¯å¾ä¸è´ç dtype=object æ ¼å¼
            dummy_row = np.arange(min(k, n_cells), dtype=np.int64)
            dummy_dist_row = np.ones(min(k, n_cells), dtype=np.float32)
            dummy_indices = np.array([dummy_row.copy() for _ in range(n_cells)], dtype=object)
            dummy_distances = np.array([dummy_dist_row.copy() for _ in range(n_cells)], dtype=object)
            return dummy_indices, dummy_distances

    def get_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """æåæ¨¡åè®­ç»æ?gene-level mixing äº§ççåæç»è?""
        if not self.is_trained:
            return None, None
        z = self.adata.uns.get('cytoneighbor', {})
        return z.get('synthetic_data', None), z.get('synthetic_labels', None)

    # def get_embeddings(self, adata: ad.AnnData) -> np.ndarray:
    #     """
    #     Get embeddings for the given data.
    #
    #     Args:
    #         adata: AnnData object
    #
    #     Returns:
    #         Embedding matrix
    #     """
    #     if not self.is_trained:
    #         raise RuntimeError("Model must be trained before getting embeddings")
    #
    #     # For Cytoneighbor, we use the ANN projection space as embeddings
    #     z = self.adata.uns.get('cytoneighbor', {})
    #     P_ann = z.get('ann', {}).get('P_ann', None)
    #
    #     if P_ann is None:
    #         raise RuntimeError("ANN projection matrix not found")
    #
    #     # Align features and get embeddings
    #     from .utils import DataLoader
    #     data_loader = DataLoader()
    #
    #     # Get feature names from trained model
    #     feature_names = z.get('feature_names', [])
    #
    #     # Align query data features with training features
    #     X = np.zeros((adata.n_obs, len(feature_names)), dtype=np.float32)
    #
    #     for i, gene in enumerate(feature_names):
    #         if gene in adata.var_names:
    #             gene_idx = adata.var_names.get_loc(gene)
    #             if hasattr(adata.X, 'toarray'):
    #                 X[:, i] = adata.X[:, gene_idx].toarray().flatten()
    #             else:
    #                 X[:, i] = adata.X[:, gene_idx]
    #
    #     # Apply preprocessing
    #     preproc_mode = z.get('preproc', {}).get('mode', 'none')
    #     if preproc_mode == 'log1p':
    #         X = np.log1p(np.maximum(X, 0.0))
    #     elif preproc_mode == 'sqrt':
    #         X = np.sqrt(np.maximum(X, 0.0))
    #
    #     # Project to ANN space
    #     embeddings = X @ P_ann.T
    #
    #     return embeddings.astype(np.float32)

    def get_embeddings(self, adata: ad.AnnData) -> np.ndarray:
        """
        Get embeddings for the given data.
        Args:
            adata: AnnData object
        Returns:
            Embedding matrix (High-dimensional gene expression)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting embeddings")

        # For Cytoneighbor, we use the ANN projection space as embeddings
        z = self.adata.uns.get('cytoneighbor', {})
        # P_ann = z.get('ann', {}).get('P_ann', None) # [å·²åºå¼] ä¸åéè¦æå½±ç©é?

        # [é»è¾ä¿æä¸å] Align features
        feature_names = z.get('feature_names', [])
        X = np.zeros((adata.n_obs, len(feature_names)), dtype=np.float32)

        for i, gene in enumerate(feature_names):
            if gene in adata.var_names:
                gene_idx = adata.var_names.get_loc(gene)
                if hasattr(adata.X, 'toarray'):
                    X[:, i] = adata.X[:, gene_idx].toarray().flatten()
                else:
                    X[:, i] = adata.X[:, gene_idx]

        # [é»è¾ä¿æä¸å] Apply preprocessing
        preproc_mode = z.get('preproc', {}).get('mode', 'none')
        if preproc_mode == 'log1p':
            X = np.log1p(np.maximum(X, 0.0))
        elif preproc_mode == 'sqrt':
            X = np.sqrt(np.maximum(X, 0.0))

        # [å³é®ä¿®å¤ç¹]
        # åéè¯¯ä»£ç ?
        # # embeddings = X @ P_ann.T
        # return embeddings.astype(np.float32)

        # ä¿®æ­£ä¸? ç´æ¥è¿åé¢å¤çåçåºå è¡¨è¾¾ç©é?X
        return X.astype(np.float32)

    def predict_cell_types(self, query_data: ad.AnnData,
                           reference_data: ad.AnnData,
                           reference_cell_type_column: str,
                           temperature: float = 1.0) -> np.ndarray:
        """
        Predict cell types for query data using reference data.

        Args:
            query_data: AnnData object containing query cells
            reference_data: AnnData object containing reference cells
                Note [Issue #6]: This parameter is currently unused. Annotation is
                performed using self.adata (the trained AnnData), which contains the
                training data and its cell type labels from prepare_data(). self.adata
                is the canonical reference for the fitted model. The parameter is
                retained in the signature for API compatibility and future extension.
            reference_cell_type_column: Column name for cell types in reference
            temperature: Temperature parameter for softmax (higher = softer distribution)

        Returns:
            Predicted cell type labels
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        # Use the trained reference data (self.adata) for annotation
        # Annotate directly on query_data to preserve probabilities

        # annotate(
        #     reference=self.adata,  # Use the trained model data
        #     reference_obs_key=reference_cell_type_column,
        #     query=query_data,  # Annotate directly on input data
        #     temperature=temperature,  # Pass temperature parameter
        #     write_diagnostics=True  # Ensure diagnostic metrics are written
        # )

        annotate(
            reference=self.adata,  
            reference_obs_key=reference_cell_type_column,
            query=query_data, 
            # enable_unknown=False,  
            enable_unknown=True,          
            score_mode="logit",            # <--- ä½¿ç¨æ å Logit
            temperature=temperature,       # <--- é»è®¤ä¸?1.0
            write_diagnostics=True  
        )

        # Extract predictions
        pred_key = 'cytoneighbor_prediction'
        if pred_key in query_data.obs.columns:
            return query_data.obs[pred_key].values
        else:
            raise RuntimeError("Predictions not found after annotation")

    def enable_synthesis(self, enable: bool = True) -> None:
        """
        Enable or disable gene-level mixing synthesis.

        Args:
            enable: Whether to enable synthesis
        """
        self.model_params['synth_enable'] = enable
        self.logger.info(f"Synthesis {'enabled' if enable else 'disabled'}")

    def get_confidence_scores(self, predictions: np.ndarray) -> np.ndarray:
        """
        Get confidence scores for predictions.

        Args:
            predictions: Prediction results from annotate

        Returns:
            Confidence scores
        """
        # This would typically be extracted from the annotate results
        # For now, return uniform confidence
        return np.ones(len(predictions))

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.

        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("No trained model to save")

        save_model(self.adata, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load a pre-trained model.

        Args:
            filepath: Path to the saved model
        """
        if self.adata is None:
            # Create empty AnnData object for loading
            self.adata = ad.AnnData(np.zeros((1, 1)))

        self.adata = load_model(self.adata, filepath)
        self.is_trained = True
        self.logger.info(f"Model loaded from {filepath}")


class TrainingManager:
    """
    Manages the training process with logging and checkpointing.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def train_model(self, adata: ad.AnnData, cell_type_column: str,
                    output_dir: str = "models") -> CytoneighborWrapper:
        """
        Train a Cytoneighbor model with full logging and checkpointing.

        Args:
            adata: Training data
            cell_type_column: Cell type column name
            output_dir: Directory to save model checkpoints

        Returns:
            Trained model wrapper
        """
        os.makedirs(output_dir, exist_ok=True)

        # Initialize model
        model = CytoneighborWrapper(self.config)

        # Train model
        model.train(adata, cell_type_column)

        # Save model
        model_path = os.path.join(output_dir, "cytoneighbor_model.zip")
        model.save_model(model_path)

        return model

    def evaluate_model(self, model: CytoneighborWrapper,
                       test_data: ad.AnnData,
                       reference_data: ad.AnnData,
                       cell_type_column: str) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            model: Trained model
            test_data: Test dataset
            reference_data: Reference dataset for annotation
            cell_type_column: Cell type column name

        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        predictions = model.predict_cell_types(
            test_data, reference_data, cell_type_column
        )

        # Calculate metrics (simplified)
        true_labels = test_data.obs[cell_type_column].values
        accuracy = np.mean(predictions == true_labels)

        return {
            'accuracy': accuracy,
            'n_predictions': len(predictions)
        }
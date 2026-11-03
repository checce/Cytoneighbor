import os
from pathlib import Path
# --- Environment variable configuration (compatibility from scArches scripts) ---
os.environ["SCIPY_ARRAY_API"] = "1"
os.environ["SKLEARN_ENABLE_ARRAY_API"] = "0"

import logging
import pandas as pd
import numpy as np
import anndata
import scanpy as sc
import torch  # For GPU memory management
import gc
import time
import sys
import warnings
from anndata import concat
from sklearn.neighbors import KNeighborsClassifier

# --- Attempt to import scvi ---
try:
    import scvi
except ImportError:
    print("Error: scvi-tools is not installed. Please run 'pip install scvi-tools'")
    sys.exit(1)

from sklearn.metrics import (
    accuracy_score, f1_score, adjusted_rand_score,
    adjusted_mutual_info_score, normalized_mutual_info_score
)

# --- Global configuration ---
# Modify paths according to your setup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
H5AD_PATH = os.environ.get("CYTONEIGHBOR_DATA", str(PROJECT_ROOT / "data" / "Zebrafish embryonic cells.h5ad"))
OUTPUT_BASE_DIR = str(PROJECT_ROOT / "results" / "CV_Perfomace")
DATASET_NAME = Path(H5AD_PATH).stem

BATCH_KEY = "batch"
CELL_TYPE_KEY = "cell_type"
N_HVG = 3000
N_NEIGHBORS = 15  # scArches KNN parameter (aligned with Cytoneighbor)
SCVI_LATENT_DIM = 30
SEED = 42

# Directory preparation
DATASET_CLEAN_NAME = DATASET_NAME.replace(" ", "_")
RESULTS_DIR = os.path.join(OUTPUT_BASE_DIR, "CV_scVI_Ecosystem_Results", DATASET_CLEAN_NAME)
MODEL_CACHE_DIR = os.path.join(OUTPUT_BASE_DIR, "model_cache", DATASET_CLEAN_NAME, "scVI_Ecosystem")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
scvi.settings.seed = SEED

# ==============================================================================
# 1. Helper functions
# ==============================================================================

def evaluate_predictions(true_labels, pred_labels, fold_info, method_name):
    """Compute 7 core metrics"""
    if true_labels is None or pred_labels is None:
        return {**fold_info, 'method': method_name, 'error': 'Missing labels'}
    try:
        true_labels = np.array(true_labels).astype(str)
        pred_labels = np.array(pred_labels).astype(str)
        all_labels = list(set(true_labels) | set(pred_labels))
        
        return {
            **fold_info,
            'method': method_name,
            'accuracy': accuracy_score(true_labels, pred_labels),
            'f1_micro': f1_score(true_labels, pred_labels, labels=all_labels, average='micro', zero_division=0),
            'f1_macro': f1_score(true_labels, pred_labels, labels=all_labels, average='macro', zero_division=0),
            'f1_weighted': f1_score(true_labels, pred_labels, labels=all_labels, average='weighted', zero_division=0),
            'ari': adjusted_rand_score(true_labels, pred_labels),
            'ami': adjusted_mutual_info_score(true_labels, pred_labels),
            'nmi': normalized_mutual_info_score(true_labels, pred_labels)
        }
    except Exception as e:
        logger.error(f" failed: {method_name}  failed: {e}")
        return {**fold_info, 'method': method_name, 'error': str(e)}

def run_joint_training(adata_ref, adata_query, fold_info):
    """
    Joint training core logic:
    1. Train Reference scVI
    2. Branch A: scVI -> Query Mapping -> Latent -> KNN (i.e. scArches)
    3. Branch B: scVI -> convert to scANVI -> Query Mapping -> Predict (i.e. scANVI)
    """
    results = []
    
    # -----------------------------------------------------------
    # Step 1: Train shared base model (Reference scVI)
    # -----------------------------------------------------------
    logger.info("  [Step 1] Training shared Reference scVI model...")
    
    # scVI Setup
    scvi.model.SCVI.setup_anndata(
        adata_ref, 
        layer="counts", 
        batch_key=BATCH_KEY, 
        labels_key=CELL_TYPE_KEY
    )
    
    model_ref_scvi = scvi.model.SCVI(adata_ref, n_latent=SCVI_LATENT_DIM)
    
    # Train scVI (early stopping, no epoch limit)
    model_ref_scvi.train(
        early_stopping=True, 
        early_stopping_patience=10, 
        train_size=0.9
    )
    
    # Save Reference scVI
    ref_scvi_path = os.path.join(MODEL_CACHE_DIR, f"ref_scvi_fold_{fold_info['fold']}")
    model_ref_scvi.save(ref_scvi_path, overwrite=True, save_anndata=False)

    # -----------------------------------------------------------
    # Step 2: Branch A - scArches (scVI + KNN)
    # -----------------------------------------------------------
    try:
        logger.info("  [Branch A] Running scArches (scVI Mapping + KNN)...")
        start_time = time.time()
        
        # Load Query into scVI (Arches mode)
        model_query_scvi = scvi.model.SCVI.load_query_data(
            adata_query,
            model_ref_scvi
        )
        model_query_scvi.train(
            max_epochs=200, 
            plan_kwargs=dict(weight_decay=0.0), 
            early_stopping=True
        )
        
        # Extract latent vectors
        latent_ref = model_ref_scvi.get_latent_representation()
        latent_query = model_query_scvi.get_latent_representation()
        
        # KNN classification
        knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, n_jobs=-1)
        knn.fit(latent_ref, adata_ref.obs[CELL_TYPE_KEY].astype(str))
        pred_labels_knn = knn.predict(latent_query)
        
        # Evaluate
        true_labels = adata_query.obs[CELL_TYPE_KEY].values
        metrics_sca = evaluate_predictions(true_labels, pred_labels_knn, fold_info, "scArches")
        metrics_sca['runtime_seconds'] = time.time() - start_time
        results.append(metrics_sca)
        
        logger.info(f"    scArches done: Macro-F1 = {metrics_sca.get('f1_macro', 0):.4f}")
        
        # Clean up Branch A memory
        del model_query_scvi, latent_ref, latent_query, knn
        gc.collect()
        
    except Exception as e:
        logger.error(f"scArches branch failed: {e}", exc_info=True)
        results.append({**fold_info, 'method': 'scArches', 'error': str(e)})

    # -----------------------------------------------------------
    # Step 3: Branch B - scANVI (Semi-supervised Transfer)
    # -----------------------------------------------------------
    try:
        logger.info("  [Branch B] Running scANVI (fine-tuning from scVI)...")
        start_time = time.time()
        
        # Initialize scANVI from trained scVI (Transfer Learning)
        model_ref_scanvi = scvi.model.SCANVI.from_scvi_model(
            model_ref_scvi,
            unlabeled_category="Unknown",
            labels_key=CELL_TYPE_KEY
        )
        
        # Fine-tune scANVI (20-50 epochs usually sufficient)
        model_ref_scanvi.train(
            max_epochs=50, 
            early_stopping=True, 
            early_stopping_patience=5
        )
        
        # Save Reference scANVI
        ref_scanvi_path = os.path.join(MODEL_CACHE_DIR, f"ref_scanvi_fold_{fold_info['fold']}")
        model_ref_scanvi.save(ref_scanvi_path, overwrite=True, save_anndata=False)
        
        # Load Query into scANVI (Arches mode)
        model_query_scanvi = scvi.model.SCANVI.load_query_data(
            adata_query,
            model_ref_scanvi
        )
        model_query_scanvi.train(
            max_epochs=100, 
            plan_kwargs=dict(weight_decay=0.0), 
            early_stopping=True
        )
        
        # scANVI direct prediction
        pred_labels_scanvi = model_query_scanvi.predict()
        
        # Evaluate
        true_labels = adata_query.obs[CELL_TYPE_KEY].values
        metrics_scanvi = evaluate_predictions(true_labels, pred_labels_scanvi, fold_info, "scANVI")
        metrics_scanvi['runtime_seconds'] = time.time() - start_time
        results.append(metrics_scanvi)
        
        logger.info(f"    scANVI done: Macro-F1 = {metrics_scanvi.get('f1_macro', 0):.4f}")
        
        # Clean up Branch B memory
        del model_ref_scanvi, model_query_scanvi
        
    except Exception as e:
        logger.error(f"scANVI branch failed: {e}", exc_info=True)
        results.append({**fold_info, 'method': 'scANVI', 'error': str(e)})

    # Clean up shared model
    del model_ref_scvi
    gc.collect()
    
    return results

# ==============================================================================
# 2. Main execution workflow
# ==============================================================================

def main():
    logger.info("=== Starting scVI Ecosystem (scArches & scANVI) Joint Cross-Validation ===")
    
    # 1. Data loading (using robust loading logic from the scANVI script)
    logger.info(f"Loading data: {H5AD_PATH}")
    try:
        adata_full = sc.read_h5ad(H5AD_PATH)
    except Exception as e:
        logger.error(f"Loading failed: {e}")
        return

    # [Key Fix] Check and restore .X
    if adata_full.X is None:
        logger.warning("Detected adata.X is None, attempting to restore...")
        if 'counts' in adata_full.layers:
            adata_full.X = adata_full.layers['counts'].copy()
        elif adata_full.raw is not None:
            adata_full.X = adata_full.raw.X.copy()
        else:
            logger.error("Unable to restore data (X/layers/raw are all empty)")
            return

    # Ensure counts layer exists (required for scVI)
    if 'counts' not in adata_full.layers:
        logger.info("Backing up raw counts to layers['counts']")
        adata_full.layers['counts'] = adata_full.X.copy()

    # Sparsify to save memory
    from scipy import sparse
    if not sparse.issparse(adata_full.X):
        adata_full.X = sparse.csr_matrix(adata_full.X)
    if not sparse.issparse(adata_full.layers['counts']):
        adata_full.layers['counts'] = sparse.csr_matrix(adata_full.layers['counts'])

    # Global filtering
    sc.pp.filter_genes(adata_full, min_cells=20)
    
    # Precompute log1p (for subsequent HVG calculation, does not affect counts layer)
    if 'log1p' not in adata_full.uns:
        sc.pp.normalize_total(adata_full, target_sum=1e4)
        sc.pp.log1p(adata_full)

    # 2. Split batches
    unique_batches = adata_full.obs[BATCH_KEY].unique().tolist()
    logger.info(f"Found {len(unique_batches)} batches in total.")
    
    adatas_for_cv = [adata_full[adata_full.obs[BATCH_KEY] == b].copy() for b in unique_batches]
    slice_names_for_cv = [f"{DATASET_CLEAN_NAME}_batch_{str(b).replace(' ', '_')}" for b in unique_batches]
    
    del adata_full
    gc.collect()

    # 3. Cross-validation loop
    all_cv_results = []
    
    for i, query_adata_orig in enumerate(adatas_for_cv):
        adata_ref_merged = None
        adata_query = None
        
        try:
            query_slice_name = slice_names_for_cv[i]
            logger.info(f"\n>>>>> Fold {i + 1}/{len(adatas_for_cv)}: Query = {query_slice_name} <<<<<")
            
            # Build reference set
            ref_list = [ad for idx, ad in enumerate(adatas_for_cv) if idx != i]
            adata_ref_merged = concat(ref_list, join='outer', merge='same')
            adata_query = query_adata_orig.copy()
            
            # [Key] Calculate HVG inside the loop (prevent leakage)
            logger.info(f"  Calculating HVG (Top {N_HVG}, based on Reference)...")
            sc.pp.highly_variable_genes(
                adata_ref_merged,
                n_top_genes=N_HVG,
                subset=True,
                flavor="seurat",
                batch_key=BATCH_KEY
            )
            hvg_genes = adata_ref_merged.var_names
            adata_query = adata_query[:, hvg_genes].copy()
            logger.info(f"  Feature alignment completed. Dimensions: {adata_ref_merged.n_vars}")
            
            # [Key] Label cleaning & GT Remapping (Open-Set fairness)
            for ad_obj in [adata_ref_merged, adata_query]:
                # Clean whitespace, etc.
                ad_obj.obs[CELL_TYPE_KEY] = ad_obj.obs[CELL_TYPE_KEY].astype(str).str.strip()
            
            ref_labels = set(adata_ref_merged.obs[CELL_TYPE_KEY].unique())
            query_labels = adata_query.obs[CELL_TYPE_KEY].values
            
            # Set labels in Query not present in Reference to Unknown
            is_unknown = ~np.isin(query_labels, list(ref_labels))
            if is_unknown.any():
                logger.info(f"  [Open-Set] Marking {is_unknown.sum()} unseen-class cells as 'Unknown'")
                query_labels[is_unknown] = "Unknown"
                adata_query.obs[CELL_TYPE_KEY] = query_labels # Update in-memory labels for failed:

            # --- Call joint training ---
            fold_info = {'fold': i + 1, 'query_slice': query_slice_name}
            fold_results = run_joint_training(
                adata_ref_merged, 
                adata_query, 
                fold_info
            )
            all_cv_results.extend(fold_results)
            
        except Exception as e:
            logger.error(f"Fold {i+1} critical error: {e}", exc_info=True)
            
        finally:
            # Memory cleanup
            del adata_ref_merged, adata_query
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("  GPU memory cleaned")

    # 4. Results summary
    if all_cv_results:
        results_df = pd.DataFrame(all_cv_results)
        
        # Save detailed results
        long_path = os.path.join(RESULTS_DIR, f"{DATASET_CLEAN_NAME}_scVI_Ecosystem_detailed.csv")
        results_df.to_csv(long_path, index=False)
        
        # Print and save summary
        summary = results_df.groupby('method')[['f1_macro', 'accuracy', 'ari', 'nmi']].agg(['mean', 'std'])
        summary_path = os.path.join(RESULTS_DIR, f"{DATASET_CLEAN_NAME}_scVI_Ecosystem_summary.csv")
        summary.to_csv(summary_path)
        
        print("\n=== Final Results Summary ===")
        print(summary)
        logger.info(f"Results saved to: {RESULTS_DIR}")
    else:
        logger.error("No valid results generated.")

if __name__ == "__main__":
    # Disable multi-process loading to prevent deadlock
    scvi.settings.dl_num_workers = 0
    main()

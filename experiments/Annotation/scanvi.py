import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import os
import sys
import warnings
import gc
import logging
import time
import torch  # For GPU memory management
from anndata import concat

# Attempt to import scvi-tools
try:
    import scvi
except ImportError:
    print("Error: scvi-tools is not installed. Please run 'pip install scvi-tools'")
    sys.exit(1)

from sklearn.metrics import (
    accuracy_score, f1_score, adjusted_rand_score,
    adjusted_mutual_info_score, normalized_mutual_info_score
)

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. Helper functions
# ==============================================================================

def evaluate_predictions(true_labels, pred_labels, fold_info, method_name):
    """Compute 7 core metrics"""
    if true_labels is None or pred_labels is None:
        return {**fold_info, 'method': method_name, 'error': 'Missing labels'}
    try:
        # Convert to string to prevent category encoding issues
        true_labels = np.array(true_labels).astype(str)
        pred_labels = np.array(pred_labels).astype(str)
        
        all_labels = list(set(true_labels) | set(pred_labels))
        
        accuracy = accuracy_score(true_labels, pred_labels)
        f1_micro = f1_score(true_labels, pred_labels, labels=all_labels, average='micro', zero_division=0)
        f1_macro = f1_score(true_labels, pred_labels, labels=all_labels, average='macro', zero_division=0)
        f1_weighted = f1_score(true_labels, pred_labels, labels=all_labels, average='weighted', zero_division=0)
        ari = adjusted_rand_score(true_labels, pred_labels)
        ami = adjusted_mutual_info_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        
        return {
            **fold_info,
            'method': method_name,
            'accuracy': accuracy, 
            'f1_micro': f1_micro, 
            'f1_macro': f1_macro, 
            'f1_weighted': f1_weighted,
            'ari': ari, 'ami': ami, 'nmi': nmi
        }
    except Exception as e:
        logger.error(f"Error evaluating method {method_name} failed: {e}", exc_info=True)
        return {**fold_info, 'method': method_name, 'error': str(e)}

def run_scANVI_annotation(adata_ref, adata_query, batch_key, cell_type_key, fold_info, model_dir):
    """Train scANVI in one cross-validation fold (Arches mode)"""
    method_name = "scANVI"
    start_time = time.time()
    
    model_ref = None
    model_query = None

    try:
        # --- 1. Train Reference ---
        logger.info("  1. Setting up and training reference scANVI model...")
        # scVI recommends using raw counts
        scvi.model.SCANVI.setup_anndata(
            adata_ref,
            layer="counts",
            batch_key=batch_key,
            labels_key=cell_type_key,
            unlabeled_category="Unknown"
        )
        
        # Use default parameters for fair comparison (latent=30 is standard)
        model_ref = scvi.model.SCANVI(adata_ref, n_latent=30)
        
        # [Modified] Remove max_epochs limit, use early_stopping to ensure convergence
        model_ref.train(
            early_stopping=True,
            train_size=0.9,
            early_stopping_patience=10,
            datasplitter_kwargs={'drop_last': True}
        )
        
        # Save Reference model so Query can use Arches load
        ref_model_path = os.path.join(model_dir, f"scANVI_ref_fold_{fold_info['fold']}")
        model_ref.save(ref_model_path, overwrite=True, save_anndata=False)

        # --- 2. Train Query (Arches) ---
        logger.info("  2. Mapping query set with Arches framework...")
        
        # Load Query data into Reference model architecture
        model_query = scvi.model.SCANVI.load_query_data(
            adata_query,
            ref_model_path,
        )
        
        # Train Query (semi-supervised/unsupervised adaptation)
        model_query.train(
            max_epochs=100, # Arches adaptation usually converges quickly; generous upper bound
            plan_kwargs=dict(weight_decay=0.0),
            early_stopping=True,
            early_stopping_patience=10
        )
        
        # Predict
        pred_labels = model_query.predict()
        
        # --- 3. Evaluate ---
        # Obtain ground-truth labels for evaluation (includes Unknown remapping)
        true_labels = adata_query.obs[cell_type_key].values
        
        results = evaluate_predictions(true_labels, pred_labels, fold_info, method_name)
        results['runtime_seconds'] = time.time() - start_time
        
        logger.info(f"  scANVI evaluation done: Macro-F1 = {results.get('f1_macro', 0.0):.4f}")
        return results

    except Exception as e:
        logger.error(f"scANVI failed in Fold {fold_info['fold']} failed: {e}", exc_info=True)
        return {**fold_info, 'method': method_name, 'error': str(e)}

    finally:
        # [Critical] Local GPU memory cleanup
        del model_ref
        del model_query
        gc.collect()

# ==============================================================================
# 2. Main workflow
# ==============================================================================

def main():
    print("--- scANVI Cross-Validation Script (Optimized) ---")
    warnings.filterwarnings('ignore')

    # --- Configuration parameters ---
    single_full_h5ad_input = "/content/drive/MyDrive/Cell_datasets/Zebrafish embryonic cells.h5ad"
    batch_obs_key = 'batch'
    cell_type_key = 'cell_type'
    
    # Number of highly variable genes (recommended 2000-4000)
    N_HVG = 3000 

    output_base_dir = "/content/drive/MyDrive/"
    dataset_name = os.path.splitext(os.path.basename(single_full_h5ad_input))[0].replace(" ", "_")
    scanvi_cv_results_dir = os.path.join(output_base_dir, "CV_scANVI_Results", dataset_name)
    model_save_dir = os.path.join(output_base_dir, "model_cache", dataset_name, "scANVI")

    os.makedirs(scanvi_cv_results_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Results path: {scanvi_cv_results_dir}")

    # --- Data loading and global preprocessing ---
    logger.info("Loading data...")
    try:
        adata_full = sc.read_h5ad(single_full_h5ad_input)
    except Exception as e:
        logger.error(f"Loading failed: {e}")
        sys.exit(1)

        # ================= [Fix Start] =================
    # Fix logic: ensure adata.X is not empty, otherwise filter_genes will raise an error
    if adata_full.X is None:
        logger.warning("Detected adata.X is None, attempting to restore data from other locations...")
        if 'counts' in adata_full.layers:
            logger.info("Restoring adata.X from adata.layers['counts']")
            adata_full.X = adata_full.layers['counts'].copy()
        elif adata_full.raw is not None:
            logger.info("Restoring adata.X from adata.raw.X")
            adata_full.X = adata_full.raw.X.copy()
        else:
            logger.error("Critical error: adata.X, layers['counts'], and raw.X are all empty, unable to process data.")
            sys.exit(1)

    # 1. Ensure 'counts' layer exists (required for scVI training)
    if 'counts' not in adata_full.layers:
        logger.info("layers['counts'] not found, backing up .X (raw counts) to layers.")
        adata_full.layers['counts'] = adata_full.X.copy()
    
    # Ensure X and layers['counts'] are both csr_matrix (saves memory and speeds up)
    from scipy import sparse
    if not sparse.issparse(adata_full.X):
        adata_full.X = sparse.csr_matrix(adata_full.X)
    if not sparse.issparse(adata_full.layers['counts']):
        adata_full.layers['counts'] = sparse.csr_matrix(adata_full.layers['counts'])
    # ================= [Fix End] =================
    
    # 2. Global basic filtering
    sc.pp.filter_genes(adata_full, min_cells=10)
    
    # 3. Precompute log1p (for HVG calculation, be careful not to overwrite the counts layer)
    # This is for fast HVG calculation inside the loop later
    if 'log1p' not in adata_full.uns:
        target_sum = 1e4
        sc.pp.normalize_total(adata_full, target_sum=target_sum)
        sc.pp.log1p(adata_full)
    
    # 4. Split object by batch
    unique_batches = adata_full.obs[batch_obs_key].unique()
    adatas_for_cv = [adata_full[adata_full.obs[batch_obs_key] == b].copy() for b in unique_batches]
    slice_names_for_cv = [f"{dataset_name}_{str(b).replace(' ', '_')}" for b in unique_batches]
    
    # Release full dataset memory
    del adata_full
    gc.collect()

    logger.info(f"Data splitting completed, total {len(adatas_for_cv)} batches.")

    # --- Cross-validation loop ---
    all_cv_results = []
    
    for i_cv_fold, query_adata_orig in enumerate(adatas_for_cv):
        adata_ref_merged = None
        adata_query = None
        
        try:
            query_slice_name = slice_names_for_cv[i_cv_fold]
            logger.info(f"\n===== CV Fold {i_cv_fold + 1}/{len(adatas_for_cv)}: Query = {query_slice_name} =====")

            # 1. Build reference set
            ref_adatas_list = [ad for idx, ad in enumerate(adatas_for_cv) if idx != i_cv_fold]
            adata_ref_merged = concat(ref_adatas_list, join='outer', merge='same')
            adata_query = query_adata_orig.copy()

            # 2. [Key optimization] Calculate highly variable genes (HVG) based on reference set
            # This significantly reduces scANVI GPU memory usage and improves accuracy
            logger.info(f"  Calculating highly variable genes (Top {N_HVG})...")
            
            # Calculate HVG using seurat method (requires log1p data, which we prepared in global preprocessing)
            sc.pp.highly_variable_genes(
                adata_ref_merged,
                n_top_genes=N_HVG,
                subset=True, # Directly slice reference set in-place
                flavor="seurat",
                batch_key=batch_obs_key # Consider batch effects within the reference set
            )
            hvg_genes = adata_ref_merged.var_names
            
            # Align query set to the same HVG
            adata_query = adata_query[:, hvg_genes].copy()
            logger.info(f"  Feature alignment completed. Training features: {len(hvg_genes)}")

            # 3. Label cleaning and GT remapping (open-set logic)
            # Clean whitespace
            for ad_obj in [adata_ref_merged, adata_query]:
                ad_obj.obs[cell_type_key] = ad_obj.obs[cell_type_key].astype(str).str.strip()
            
            # Check unseen classes
            ref_labels = set(adata_ref_merged.obs[cell_type_key].unique())
            query_labels = adata_query.obs[cell_type_key].astype(str).values
            
            # Rename categories in Query not present in Reference to 'Unknown'
            # scANVI is a closed-set model; it does not know these classes during training and will force-classify them, leading to errors.
            # The renaming here is so that during metric calculation, GT=Unknown and Pred=KnownClass,
            # thus correctly recorded as False Positive/Negative, reflecting the model's poor robustness to new classes.
            is_unknown = ~np.isin(query_labels, list(ref_labels))
            if is_unknown.any():
                logger.info(f"  [Open-Set] Marking {is_unknown.sum()} unseen-class cells as 'Unknown'")
                # We modify query labels in memory for evaluation, not modifying original data
                query_labels[is_unknown] = "Unknown"
                adata_query.obs[cell_type_key] = query_labels

            # 4. Run scANVI
            fold_info = {'fold': i_cv_fold + 1, 'query_slice': query_slice_name}
            scANVI_results = run_scANVI_annotation(
                adata_ref=adata_ref_merged,
                adata_query=adata_query,
                batch_key=batch_obs_key,
                cell_type_key=cell_type_key,
                fold_info=fold_info,
                model_dir=model_save_dir
            )
            
            all_cv_results.append(scANVI_results)

        except Exception as e:
            logger.error(f"  Fold {i_cv_fold+1} encountered a serious error: {e}", exc_info=True)
        
        finally:
            # 5. [Key optimization] Deep GPU memory cleanup
            logger.info("  Cleaning GPU memory...")
            del adata_ref_merged
            del adata_query
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"  CUDA cache cleared. Current memory usage: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

    # --- Results summary ---
    print("\n--- Results Summary ---")
    if all_cv_results:
        results_df = pd.DataFrame(all_cv_results)
        
        # Save long format
        long_path = os.path.join(scanvi_cv_results_dir, f"{dataset_name}_scANVI_detailed.csv")
        results_df.to_csv(long_path, index=False)
        
        # Generate summary
        metric_cols = ['accuracy', 'f1_macro', 'f1_weighted', 'ari', 'nmi']
        available_cols = [c for c in metric_cols if c in results_df.columns]
        
        if available_cols:
            summary = results_df.groupby('method')[available_cols].agg(['mean', 'std'])
            print(summary)
            summary_path = os.path.join(scanvi_cv_results_dir, f"{dataset_name}_scANVI_summary.csv")
            summary.to_csv(summary_path)
            logger.info(f"Summary saved to: {summary_path}")
    else:
        logger.error("No results generated.")

    print("\n--- Script Finished ---")

if __name__ == '__main__':
    # Disable multi-process loading to prevent deadlock
    scvi.settings.dl_num_workers = 0
    # Set random seed
    scvi.settings.seed = 42
    main()
import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import sys
import warnings
import gc # Import garbage collection module

# Import scikit-learn evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    adjusted_rand_score,
    adjusted_mutual_info_score, # AMI
    normalized_mutual_info_score # NMI
)
import matplotlib.pyplot as plt # Import plotting library
import seaborn as sns
# Import anndata concat function
from anndata import concat
# Import celltypist library
import celltypist
import importlib # Import importlib for module reloading in interactive environments
import glob # For finding existing files
import logging # Import logging module
import time # For timing (if needed)
import scipy.sparse # For handling sparse matrices (e.g. value inspection)
# Import pandas api types for dtype checking
from pandas.api.types import is_categorical_dtype, is_numeric_dtype
from pandas import CategoricalDtype # Import for creating categorical dtype

# --- Configure logging ---
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# (filter_and_split_batches and other helpers omitted; keep as-is)
# Only the core evaluation logic modified in main() is shown
# --- Helper function: extended evaluation ---
# (this function is unchanged)
def extended_evaluation(true_labels, pred_labels, categories):
    """Detailed classification evaluation with reports and confusion matrices."""
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

    processed_categories = list(categories)
    unique_all_labels = set(true_labels) | set(pred_labels)

    for label in ['Unknown', 'unknown', 'Unassigned', 'unassigned', 'nan', 'none', 'N/A']:
        if label in unique_all_labels and label not in processed_categories:
            processed_categories.append(label)

    unique_all_labels_str = {str(lab) for lab in unique_all_labels}
    final_categories_for_report = sorted([str(cat) for cat in processed_categories if str(cat) in unique_all_labels_str])

    if not final_categories_for_report:
        final_categories_for_report = sorted(list(unique_all_labels_str))

    if not final_categories_for_report:
        logger.warning("No valid categories available for classification report generation.")
        return

    print("\n完整分类报告 (包括 'Unknown' 等):")
    try:
        print(classification_report(
            np.asarray(true_labels).astype(str),
            np.asarray(pred_labels).astype(str),
            labels=final_categories_for_report,
            target_names=final_categories_for_report,
            zero_division=0
        ))
    except ValueError as e_cr:
        logger.error(f"Error generating full classification report: {e_cr}")
    except Exception as e_other_cr:
         logger.error(f"Unknown error generating full classification report: {e_other_cr}")

    unknown_like_predictions = {'Unknown', 'unknown', 'Unassigned', 'unassigned', 'nan', 'none', 'N/A'}
    pred_labels_str = np.asarray(pred_labels).astype(str)
    true_labels_str = np.asarray(true_labels).astype(str)

    known_mask = ~np.isin(pred_labels_str, list(unknown_like_predictions)) & \
                 ~np.isin(true_labels_str, list(unknown_like_predictions))

    filtered_true = true_labels_str[known_mask]
    filtered_pred = pred_labels_str[known_mask]

    if len(filtered_true) > 0 and len(filtered_pred) > 0:
        print("\nKnown-sample classification report (excluding cells with true/predicted Unknown/unassigned):")
        known_cats_for_cm_report = sorted(list(set(filtered_true) | set(filtered_pred)))
        if not known_cats_for_cm_report:
             print("After filtering out 'Unknown', there are no valid known categories left for reporting.")
        else:
            try:
                print(classification_report(
                    filtered_true, filtered_pred,
                    labels=known_cats_for_cm_report,
                    target_names=known_cats_for_cm_report,
                    zero_division=0
                ))
                plt.figure(figsize=(max(8, len(known_cats_for_cm_report)*0.6),
                                   max(6, len(known_cats_for_cm_report)*0.5)))
                try:
                    cm = confusion_matrix(
                        filtered_true, filtered_pred,
                        labels=known_cats_for_cm_report
                    )
                    disp = ConfusionMatrixDisplay(
                        confusion_matrix=cm,
                        display_labels=known_cats_for_cm_report
                    )
                    disp.plot(include_values=True, cmap='Blues', ax=plt.gca(), colorbar=False)
                    plt.title("Confusion matrix (known true -> known predicted labels)")
                    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
                    plt.tight_layout(); plt.show(); plt.close('all')
                except ValueError as e_cm:
                    logger.error(f"Error plotting confusion matrix: {e_cm}")
                    plt.close('all')
                except Exception as e_other_cm:
                     logger.error(f"Unknown error plotting confusion matrix: {e_other_cm}")
                     plt.close('all')
            except ValueError as e_cr_filtered:
                 logger.error(f"Error generating filtered classification report: {e_cr_filtered}")
            except Exception as e_other_cr_filtered:
                 logger.error(f"Unknown error generating filtered classification report: {e_other_cr_filtered}")
    else:
        print("\nAfter filtering out labels such as 'Unknown', there are not enough known samples to generate a classification report or confusion matrix for the known categories.")


# --- filter_and_split_batches 函数 ---
# (this function is unchanged)
def filter_and_split_batches(
    input_h5ad_path: str,
    output_dir: str,
    batch_obs_key: str,
    min_cells_for_gene_filter: int,
    counts_layer_key: str = 'counts'
) -> list:
    """
    Load a single AnnData object, perform gene filtering on the full dataset, split by batch key,
    ensure .X is numeric, save each batch to output_dir, and return a list of AnnData objects.
    """
    logger.info(f"Loading, filtering and splitting file: {input_h5ad_path}...")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Split cache files will be saved to: {output_dir}")

    try:
        logger.info("Loading full AnnData object...")
        adata_full = sc.read_h5ad(input_h5ad_path)
        logger.info(f"Successfully loaded {input_h5ad_path}。Initial dimensions: {adata_full.shape}")
    except Exception as e:
        logger.error(f"加载 {input_h5ad_path} 失败", exc_info=True)
        return []

    logger.info(f"Applying gene filter on the full dataset (min_cells={min_cells_for_gene_filter})...")
    if counts_layer_key and counts_layer_key in adata_full.layers:
        logger.info(f"Using  a.layers['{counts_layer_key}']  for gene filtering...")
        adata_for_filtering = ad.AnnData(X=adata_full.layers[counts_layer_key], var=adata_full.var.copy())
    else:
        logger.info(f"Using  a.X  for gene filtering...")
        adata_for_filtering = ad.AnnData(X=adata_full.X, var=adata_full.var.copy())
    n_genes_before_filter = adata_for_filtering.n_vars
    sc.pp.filter_genes(adata_for_filtering, min_cells=min_cells_for_gene_filter)
    genes_to_keep = adata_for_filtering.var_names
    logger.info(f"Global gene filtering complete: {n_genes_before_filter} -> {len(genes_to_keep)}  genes.")
    del adata_for_filtering
    gc.collect()
    adata_full_filtered = adata_full[:, genes_to_keep].copy()
    logger.info(f"Final dataset created from filtered gene list. Current dimensions: {adata_full_filtered.shape}")
    del adata_full
    gc.collect()

    if batch_obs_key not in adata_full_filtered.obs.columns:
        logger.error(f"Batch key '{batch_obs_key}' not found in adata.obs. Available columns: {list(adata_full_filtered.obs.columns)}")
        del adata_full_filtered; gc.collect()
        return []

    unique_batches = adata_full_filtered.obs[batch_obs_key].unique()
    logger.info(f"Found {len(unique_batches)} unique batches for splitting: {list(unique_batches)}")
    base_filename = os.path.splitext(os.path.basename(input_h5ad_path))[0].replace(" ", "_").replace("/", "-").replace("\\", "-")

    processed_batches_list = []
    for i, batch_id in enumerate(unique_batches):
        batch_id_str = str(batch_id)
        logger.info(f"--- Processing batch {i+1}/{len(unique_batches)}: '{batch_id_str}' ---")
        adata_batch = None
        try:
            logger.debug(f"Subsetting AnnData for batch '{batch_id_str}'.")
            adata_batch = adata_full_filtered[adata_full_filtered.obs[batch_obs_key] == batch_id].copy()
            logger.info(f"批次 '{batch_id_str}'  subsetting complete. Dimensions: {adata_batch.shape}")
            gc.collect()
            logger.info("Performing preliminary processing (loading counts from layer, ensuring numeric X)...")
            if counts_layer_key and counts_layer_key in adata_batch.layers:
                logger.info(f"从 adata_batch.layers['{counts_layer_key}']  into .X...")
                layer_data = adata_batch.layers[counts_layer_key]
                if scipy.sparse.issparse(layer_data):
                    layer_data = layer_data.tocsr()
                adata_batch.X = layer_data.copy()
            else:
                 logger.warning(f"未Found '{counts_layer_key}'  not found, assuming .X contains raw counts.")
                 if adata_batch.X is None:
                    raise ValueError(f"未Found '{counts_layer_key}'  not found, and .X is empty.")
            if hasattr(adata_batch.X, 'dtype') and not np.issubdtype(adata_batch.X.dtype, np.number):
                try: adata_batch.X = adata_batch.X.astype(np.float32)
                except: continue
            elif hasattr(adata_batch.X, 'dtype') and adata_batch.X.dtype != np.float32:
                 try: adata_batch.X = adata_batch.X.astype(np.float32)
                 except: pass
            if adata_batch.n_obs == 0 or adata_batch.n_vars == 0:
                logger.warning(f"Batch '{batch_id_str}'  is empty, skipped.")
                continue
            safe_batch_id_fn = batch_id_str.replace(" ", "_").replace("/", "-").replace("\\", "-")
            slice_name = f"{base_filename}_batch_{safe_batch_id_fn}"
            adata_batch.uns['slice_name'] = slice_name
            if 'slice_name' not in adata_batch.obs.columns:
                adata_batch.obs['slice_name'] = slice_name
            output_batch_path = os.path.join(output_dir, f"{slice_name}.h5ad")
            try:
                logger.info(f"Saving split data (globally gene-filtered) to {output_batch_path}")
                temp_adata_to_save = ad.AnnData(X=adata_batch.X, obs=adata_batch.obs.copy(), var=adata_batch.var.copy())
                if 'slice_name' in adata_batch.uns: temp_adata_to_save.uns['slice_name'] = adata_batch.uns['slice_name']
                temp_adata_to_save.write_h5ad(output_batch_path)
                logger.info(f"Cached file saved: {slice_name}")
                processed_batches_list.append(adata_batch)
            except Exception as e_save_batch:
                logger.warning(f"Cannot save batch '{batch_id_str}'  to {output_batch_path}: {e_save_batch}。")
                processed_batches_list.append(adata_batch)
        except Exception as e_batch_proc:
            logger.error(f"Error processing batch '{batch_id_str}'  error: {e_batch_proc}", exc_info=True)
        finally:
            if 'adata_batch' in locals() and adata_batch is not None:
                del adata_batch
            gc.collect()
            logger.info(f"--- Completed Error processing batch '{batch_id_str}' ---")
    if 'adata_full_filtered' in locals() and adata_full_filtered is not None:
        del adata_full_filtered
    gc.collect()
    logger.info("All batches have been split and cached.")
    return processed_batches_list

def main():
    print("--- CellTypist Cross-validation script ---")
    warnings.filterwarnings('ignore')

    # --- 1. Configuration parameters ---
    print("Step 1: Configure parameters")
    # Input file path
    single_full_h5ad_input = '/home/ioag/yyq/Cell_datasets/Mouse_Kidney_mod1.h5ad'
    # Batch and cell-type column names
    batch_obs_key_for_splitting = 'batch'
    cell_type_col = 'cell_type'
    
    # Output directory
    output_base_dir = "/home/ioag/yyq/"
    cv_results_parent_subdir = "CV_Celltpist_Results"
    dataset_name_subdir = os.path.splitext(os.path.basename(single_full_h5ad_input))[0].replace(" ", "_")
    final_output_dir = os.path.join(output_base_dir, cv_results_parent_subdir, dataset_name_subdir)
    # raw_processed_splits_dir = os.path.join(output_base_dir, "processing_cache", dataset_name_subdir) # 建议子目录加上数据集名称以防冲突
    raw_processed_splits_dir = os.path.join(output_base_dir, "processing_cache")
    
    # Create directories
    os.makedirs(raw_processed_splits_dir, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)
    
    logger.info(f"All results from this run will be saved to: {final_output_dir}")

    # --- Key parameter definitions ---
    MIN_CELLS_FOR_GENE_FILTER = 10  # Gene filter: min cells threshold
    COUNTS_LAYER_KEY = 'counts'     # Raw counts layer name
    
    # --- CellTypist training parameters ---
    # Use officially recommended defaults
    # n_jobs=-1 uses all available CPU cores for acceleration
    # check_expression=True validates input data format (e.g. log1p check)
    # feature_selection: if n_genes > 300, CellTypist performs feature selection by default. Set dynamically in-loop.
    celltypist_base_train_args = {
        'check_expression': True,
        'n_jobs': -1,
        # 'max_iter': 1000 
    }

    # --- 2. Data loading and subset preparation ---
    print("\nStep 2: Data loading and subset preparation...")
    adatas_for_cv = []
    safe_base_input_filename = os.path.splitext(os.path.basename(single_full_h5ad_input))[0].replace(" ", "_").replace("/", "-").replace("\\", "-")
    split_file_pattern = f"{safe_base_input_filename}_batch_*.h5ad"
    potential_split_files = sorted(glob.glob(os.path.join(raw_processed_splits_dir, split_file_pattern)))
    
    if len(potential_split_files) >= 2:
        print(f"   in directory '{raw_processed_splits_dir}' 中Found {len(potential_split_files)} potential cache files. Attempting to load...")
        loaded_count = 0
        temp_adatas_raw = []
        for path in potential_split_files:
            try:
                ad_item = sc.read_h5ad(path)
                s_name = os.path.splitext(os.path.basename(path))[0]
                s_name_cleaned = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in s_name)
                if not s_name_cleaned: s_name_cleaned = f"unnamed_slice_{len(temp_adatas_raw)}"
                ad_item.uns['slice_name'] = s_name_cleaned
                if 'slice_name' not in ad_item.obs.columns: ad_item.obs['slice_name'] = s_name_cleaned
                if ad_item.n_obs > 0 and ad_item.n_vars > 0:
                    temp_adatas_raw.append(ad_item)
                    print(f"    Loaded cached file: {s_name_cleaned} ({ad_item.n_obs}x{ad_item.n_vars})")
                    loaded_count += 1
                else:
                    warnings.warn(f"    Cached file {path} 加载后 is empty, skipped.")
                    if 'ad_item' in locals(): del ad_item; gc.collect()
            except Exception as e:
                warnings.warn(f"    警告: 加载Cached file {path}  failed: {e}")
        if loaded_count >= 2:
            print(f"  Successfully loaded {loaded_count} cached files. Skipping direct splitting.")
            adatas_for_cv = temp_adatas_raw
        else:
            print(f"  Only {loaded_count} 个Cached file (少于 CV 所需的 2 个)。继续分割完整数据集。")
    
    if len(adatas_for_cv) < 2:
        print(f"  Insufficient cached files found. Calling global filter and split logic on '{single_full_h5ad_input}'...")
        adatas_for_cv = filter_and_split_batches(
            input_h5ad_path=single_full_h5ad_input,
            output_dir=raw_processed_splits_dir,
            batch_obs_key=batch_obs_key_for_splitting,
            min_cells_for_gene_filter=MIN_CELLS_FOR_GENE_FILTER,
            counts_layer_key=COUNTS_LAYER_KEY
        )
    
    if not adatas_for_cv or len(adatas_for_cv) < 2:
        print("Error: fewer than two valid subsets remain after preparation. Cannot perform cross-validation.")
        sys.exit(1)
        
    valid_slice_names_for_cv = [ad.uns.get('slice_name', f'UnnamedSplit_{i}') for i, ad in enumerate(adatas_for_cv)]
    print(f"Using  {len(adatas_for_cv)} valid subsets for cross-validation.")
    
    # --- 3. Start cross-validation ---
    print("\nStep 3: Start cross-validation...")
    cv_results_list =[]
    
    for i_query_cv in range(len(adatas_for_cv)):
        query_adata_original = adatas_for_cv[i_query_cv].copy()
        query_s_name_cv = query_adata_original.uns.get('slice_name', f'fold_{i_query_cv}')
        logger.info(f"\n--- CV Fold {i_query_cv + 1}/{len(adatas_for_cv)}: query set = {query_s_name_cv} ---")
        
        # Build training set
        train_adatas_this_fold_raw =[ad.copy() for idx, ad in enumerate(adatas_for_cv) if idx != i_query_cv]
        adata_train_merged_raw = concat(train_adatas_this_fold_raw, join='inner')

        valid_train_mask = ~pd.isna(adata_train_merged_raw.obs[cell_type_col])
        num_dropped = sum(~valid_train_mask)
        if num_dropped > 0:
            logger.warning(f"  [数据清洗] 从 Reference 中剔除了 {num_dropped} 个标签为 NaN 的细胞。")
            adata_train_merged_raw = adata_train_merged_raw[valid_train_mask].copy()
        
        adata_train_merged_raw.obs[cell_type_col] = adata_train_merged_raw.obs[cell_type_col].astype(str)
 
        if sp.issparse(adata_train_merged_raw.X):
            adata_train_merged_raw.X.data = np.nan_to_num(adata_train_merged_raw.X.data, nan=0.0)
        else:
            adata_train_merged_raw.X = np.nan_to_num(adata_train_merged_raw.X, nan=0.0)
            
        sc.pp.normalize_total(adata_train_merged_raw, target_sum=1e4)
        sc.pp.log1p(adata_train_merged_raw)
        

        train_args = celltypist_base_train_args.copy()

        if adata_train_merged_raw.n_vars >= 300:
            train_args['feature_selection'] = True
        else:
            train_args['feature_selection'] = False
            
        logger.info(f"训练 CellTypist 模型 (Using 默认参数, feature_selection={train_args['feature_selection']})...")
        
        try:
           
            ct_model = celltypist.train(adata_train_merged_raw, labels=cell_type_col, **train_args)
            
        
            query_adata_for_pred = query_adata_original.copy()
            sc.pp.normalize_total(query_adata_for_pred, target_sum=1e4)
            sc.pp.log1p(query_adata_for_pred)
            
            # Predict
            predictions = celltypist.annotate(query_adata_for_pred, model=ct_model, majority_voting=False)
            pred_labels_cv = predictions.predicted_labels['predicted_labels'].astype(str)
            

            true_labels_cv = query_adata_original.obs[cell_type_col].astype(str).copy()
            ref_classes = set(adata_train_merged_raw.obs[cell_type_col].astype(str).unique())
            unseen_mask = ~true_labels_cv.isin(ref_classes)
            
            if unseen_mask.any():
                logger.info(f"Query 中发现未见类型，将 {unseen_mask.sum()} 个细胞映射为 'Unknown'")
                true_labels_cv[unseen_mask] = "Unknown"
            
 
            acc_cv = accuracy_score(true_labels_cv, pred_labels_cv)
            unique_labels = list(set(true_labels_cv) | set(pred_labels_cv))
            f1_micro = f1_score(true_labels_cv, pred_labels_cv, labels=unique_labels, average='micro', zero_division=0)
            f1_macro = f1_score(true_labels_cv, pred_labels_cv, labels=unique_labels, average='macro', zero_division=0)
            f1_weighted = f1_score(true_labels_cv, pred_labels_cv, labels=unique_labels, average='weighted', zero_division=0)
            ari_cv = adjusted_rand_score(true_labels_cv, pred_labels_cv)
            ami_cv = adjusted_mutual_info_score(true_labels_cv, pred_labels_cv)
            nmi_cv = normalized_mutual_info_score(true_labels_cv, pred_labels_cv)
            

            cv_results_list.append({
                'fold_query_slice': query_s_name_cv, 
                'accuracy': acc_cv, 
                'f1_micro': f1_micro, 
                'f1_macro': f1_macro, 
                'f1_weighted': f1_weighted, 
                'ari': ari_cv, 
                'ami': ami_cv, 
                'nmi': nmi_cv, 
                'status': 'Success'
            })
            
            logger.info(f"Fold {i_query_cv + 1} 成功: Accuracy={acc_cv:.4f}, Macro-F1={f1_macro:.4f}")

        except Exception as e:
            logger.error(f"Fold {i_query_cv + 1}  failed: {e}", exc_info=True)
            cv_results_list.append({
                'fold_query_slice': query_s_name_cv,
                'status': 'Failed',
                'error': str(e)
            })

        del query_adata_original, train_adatas_this_fold_raw, adata_train_merged_raw
        if 'ct_model' in locals(): del ct_model
        if 'query_adata_for_pred' in locals(): del query_adata_for_pred
        gc.collect()

    # --- 4. Summary and save logic ---
    print("\nStep 4: Summarizing results...")
    if cv_results_list:
        results_df = pd.DataFrame(cv_results_list)
        
        # Save detailed results
        results_csv_path = os.path.join(final_output_dir, "celltypist_summary.csv")
        results_df.to_csv(results_csv_path, index=False)
        logger.info(f"CellTypist cross-validation detailed results saved to: {results_csv_path}")
        
        # Computing and printing mean values (successful folds only)
        success_df = results_df[results_df['status'] == 'Success']
        if not success_df.empty:
            mean_metrics = success_df[['accuracy', 'f1_macro', 'f1_weighted', 'ari', 'nmi']].mean()
            print("\nAverage performance metrics:")
            print(mean_metrics)
    else:
        logger.warning("No results generated.")

    logger.info("CellTypist cross-validation complete.")

if __name__ == '__main__':
    main()
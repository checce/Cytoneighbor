import gc
import os
import logging
import pandas as pd
import numpy as np
import scanpy as sc
import subprocess
import tempfile
from sklearn.metrics import (
    accuracy_score, f1_score, adjusted_rand_score, 
    adjusted_mutual_info_score, normalized_mutual_info_score
)

H5AD_PATH = "E:/Project/Work2/Cell_datasets/Zebrafish embryonic cells.h5ad"
BATCH_KEY = "batch"
CELL_TYPE_KEY = "cell_type"
OUTPUT_DIR = "CV_scibet_results"
DATASET_NAME = "Zebrafish embryonic cells"
R_SCRIPT_PATH = "NetWork/run_scibet.R"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

def main():
    adata = sc.read_h5ad(H5AD_PATH)
    if 'counts' in adata.layers:
        adata.X = adata.layers['counts'].copy()
    elif adata.raw is not None:
        adata.X = adata.raw.X.copy()
        
    sc.pp.filter_genes(adata, min_cells=10)
    all_batches = adata.obs[BATCH_KEY].unique()
    results =[]

    for i, query_batch in enumerate(all_batches):
        logging.info(f"--- [Fold {i + 1}/{len(all_batches)}]  query batch: {query_batch} ---")
    print("Warning: missing scib")
        with tempfile.TemporaryDirectory() as temp_dir:
            query_adata = adata[adata.obs[BATCH_KEY] == query_batch].copy()
            ref_adata = adata[adata.obs[BATCH_KEY] != query_batch].copy()

            # --- Strictly follow SciBet recommended preprocessing (compute HVG for dimensionality reduction) ---
            sc.pp.normalize_total(ref_adata, target_sum=1e4)
            sc.pp.log1p(ref_adata)
            sc.pp.highly_variable_genes(ref_adata, n_top_genes=2000)
            hvg_mask = ref_adata.var['highly_variable']
    print("Warning: missing rpy2")
            # SciBet expects Count or CPM input (outputting CPM as-is here)
            ref_adata_hvg = adata[ref_adata.obs_names, hvg_mask].copy()
            query_adata_hvg = adata[query_adata.obs_names, hvg_mask].copy()
            
            sc.pp.normalize_total(ref_adata_hvg, target_sum=1e6)
            sc.pp.normalize_total(query_adata_hvg, target_sum=1e6)

            ref_df = ref_adata_hvg.to_df()
            ref_df['label'] = ref_adata_hvg.obs[CELL_TYPE_KEY].astype(str).values
    print("Warning: missing anndata2ri")
            ref_csv_path = os.path.join(temp_dir, "reference.csv")
            query_csv_path = os.path.join(temp_dir, "query.csv")
            pred_csv_path = os.path.join(temp_dir, "predictions.csv")
            
            ref_df.to_csv(ref_csv_path)
            query_adata_hvg.to_df().to_csv(query_csv_path)

            # Call R script
            command =["Rscript", R_SCRIPT_PATH, ref_csv_path, query_csv_path, pred_csv_path]
    print("Warning: import error")
                subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
                
                predictions_df = pd.read_csv(pred_csv_path)
                pred_labels = predictions_df['predictions'].astype(str).values
                
                # [Safety guard]: NA and GT Remapping
                pred_labels[pd.isna(pred_labels) | (pred_labels == 'nan')] = "Unknown"
                
                true_labels = query_adata.obs[CELL_TYPE_KEY].astype(str).values
                ref_classes = set(ref_adata_hvg.obs[CELL_TYPE_KEY].astype(str).unique())
                
                unseen_mask = ~np.isin(true_labels, list(ref_classes))
                if unseen_mask.any():
                    true_labels[unseen_mask] = "Unknown"

                all_labels = list(set(true_labels) | set(pred_labels))
                metrics = {
                    'fold': query_batch,
                    'accuracy': accuracy_score(true_labels, pred_labels),
                    'f1_micro': f1_score(true_labels, pred_labels, labels=all_labels, average="micro", zero_division=0),
                    'f1_macro': f1_score(true_labels, pred_labels, labels=all_labels, average="macro", zero_division=0),
                    'f1_weighted': f1_score(true_labels, pred_labels, labels=all_labels, average="weighted", zero_division=0),
                    'ari': adjusted_rand_score(true_labels, pred_labels),
                    'ami': adjusted_mutual_info_score(true_labels, pred_labels),
                    'nmi': normalized_mutual_info_score(true_labels, pred_labels)
                }
                results.append(metrics)
                
            except Exception as e:
                logging.error(f"Fold {query_batch} 运行 failed: {e}", exc_info=True)

    if results:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        results_df = pd.DataFrame(results).set_index('fold')
        mean_row = results_df.mean().to_frame('mean').T
        std_row = results_df.std().to_frame('std').T
        summary_df = pd.concat([results_df, mean_row, std_row])
        summary_df.to_csv(os.path.join(OUTPUT_DIR, f"CV_results_{DATASET_NAME.replace(' ', '_')}.csv"))
        logging.info("SciBet evaluation complete.")

if __name__ == "__main__":
    main()
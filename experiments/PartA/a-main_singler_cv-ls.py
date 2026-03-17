import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import os
import sys
import warnings
import gc
import logging
import subprocess
import tempfile
from anndata import concat
from sklearn.metrics import (
    accuracy_score, f1_score, adjusted_rand_score,
    normalized_mutual_info_score, adjusted_mutual_info_score
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def main():
    print("--- SingleR 交叉验证脚本 ---")
    warnings.filterwarnings('ignore')

    base_project_dir = "E:/Project/Work2"
    single_full_h5ad_input = os.path.join(base_project_dir, 'Cell_datasets', 'Zebrafish embryonic cells.h5ad')
    BATCH_KEY = 'batch'
    CELL_TYPE_KEY = 'cell_type'
    R_SCRIPT_PATH = os.path.join(base_project_dir, 'NetWork', 'run_singler.R')
    dataset_name = os.path.splitext(os.path.basename(single_full_h5ad_input))[0].replace(" ", "_")
    singler_cv_results_dir = os.path.join(base_project_dir, "CV_SingleR_Results", dataset_name)
    os.makedirs(singler_cv_results_dir, exist_ok=True)

    adata = sc.read_h5ad(single_full_h5ad_input)
    if 'counts' in adata.layers:
        adata.X = adata.layers['counts'].copy()
    
    sc.pp.filter_genes(adata, min_cells=10)
    
    all_batches = adata.obs[BATCH_KEY].unique().tolist()
    all_results =[]

    for i, query_batch_id in enumerate(all_batches):
        logger.info(f"\n===== CV Fold {i + 1}/{len(all_batches)}: 查询集 = {query_batch_id} =====")
        query_adata = adata[adata.obs[BATCH_KEY] == query_batch_id].copy()
        reference_adata = adata[adata.obs[BATCH_KEY] != query_batch_id].copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # 【核心修复 1】：必须进行 library size 归一化，否则 SingleR 的相关性度量会失效
                for ad_obj in [reference_adata, query_adata]:
                    sc.pp.normalize_total(ad_obj, target_sum=1e4)
                    sc.pp.log1p(ad_obj)

                ref_path = os.path.join(temp_dir, "ref_data.h5ad")
                query_path = os.path.join(temp_dir, "query_data.h5ad")
                output_path = os.path.join(temp_dir, "predictions.csv")

                reference_adata.write_h5ad(ref_path)
                query_adata.write_h5ad(query_path)

                command =['Rscript', R_SCRIPT_PATH, ref_path, query_path, output_path, CELL_TYPE_KEY]
                subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')

                predictions_df = pd.read_csv(output_path)
                pred_labels = predictions_df['predicted_label'].astype(str).values

                # 【GT Remapping 与 NA 填充】
                pred_labels[pd.isna(pred_labels) | (pred_labels == 'nan')] = "Unknown"
                true_labels = query_adata.obs[CELL_TYPE_KEY].astype(str).values
                ref_classes = set(reference_adata.obs[CELL_TYPE_KEY].astype(str).unique())
                
                unseen_mask = ~np.isin(true_labels, list(ref_classes))
                if unseen_mask.any():
                    true_labels[unseen_mask] = "Unknown"

                # 【核心修复 2】：使用字符串原生支持的 scikit-learn 评估，避免 cat.codes 错乱
                all_labels = list(set(true_labels) | set(pred_labels))
                metrics = {
                    'fold': query_batch_id,
                    'accuracy': accuracy_score(true_labels, pred_labels),
                    'f1_micro': f1_score(true_labels, pred_labels, labels=all_labels, average='micro', zero_division=0),
                    'f1_macro': f1_score(true_labels, pred_labels, labels=all_labels, average='macro', zero_division=0),
                    'f1_weighted': f1_score(true_labels, pred_labels, labels=all_labels, average='weighted', zero_division=0),
                    'ari': adjusted_rand_score(true_labels, pred_labels),
                    'ami': adjusted_mutual_info_score(true_labels, pred_labels),
                    'nmi': normalized_mutual_info_score(true_labels, pred_labels)
                }
                all_results.append(metrics)

            except Exception as e:
                logger.error(f"处理 Fold {query_batch_id} 失败: {e}", exc_info=True)

    if all_results:
        results_df = pd.DataFrame(all_results).set_index('fold')
        mean_row = results_df.mean().to_frame('mean').T
        std_row = results_df.std().to_frame('std').T
        summary_df = pd.concat([results_df, mean_row, std_row])
        summary_df.to_csv(os.path.join(singler_cv_results_dir, f"{dataset_name}_summary.csv"))
        logger.info("SingleR 评估完成。")

if __name__ == '__main__':
    main()
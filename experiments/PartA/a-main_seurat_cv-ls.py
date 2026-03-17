import os
import logging
import subprocess
import sys
import pandas as pd
import numpy as np
import anndata
import scanpy as sc
from scipy.io import mmwrite
from scipy.sparse import csr_matrix
from sklearn.metrics import (
    accuracy_score, f1_score, adjusted_rand_score,
    normalized_mutual_info_score, adjusted_mutual_info_score
)

# ==============================================================================
# 配置与环境设置
# ==============================================================================
H5AD_PATH = "E:/Project/Work2/Cell_datasets/Zebrafish embryonic cells.h5ad"
BATCH_KEY = "batch"
CELL_TYPE_KEY = "cell_type"
OUTPUT_DIR = "CV_seurat_results"
DATASET_NAME = "Diabetic Kidney Disease"
TOOL_NAME = "Seurat_TransferData"
R_SCRIPT_PATH = os.path.join("NetWork", "run_seurat.R") # 请确保路径拼写正确

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, f"log_{DATASET_NAME}.txt"), mode='w', encoding='utf-8')
    ]
)

def export_adata_to_csv_mtx(adata: anndata.AnnData, output_dir: str, prefix: str):
    os.makedirs(output_dir, exist_ok=True)
    counts_matrix = adata.X.T
    if not isinstance(counts_matrix, csr_matrix):
        counts_matrix = csr_matrix(counts_matrix)
    mmwrite(os.path.join(output_dir, f"{prefix}_counts.mtx"), counts_matrix)

    obs_df = adata.obs.copy()
    obs_df.index.name = 'cell_barcode'
    obs_df.to_csv(os.path.join(output_dir, f"{prefix}_meta.csv"))

    pd.DataFrame(adata.obs_names).to_csv(os.path.join(output_dir, f"{prefix}_barcodes.csv"), index=False, header=False)
    if prefix == "ref":
        pd.DataFrame(adata.var_names).to_csv(os.path.join(output_dir, "features.csv"), index=False, header=False)

def main():
    if not os.path.exists(R_SCRIPT_PATH):
        logging.error(f"R脚本未找到: {R_SCRIPT_PATH}")
        sys.exit(1)

    logging.info("加载完整的AnnData对象...")
    adata = sc.read_h5ad(H5AD_PATH)
    
    # 构建唯一的 barcode index
    if not all('-' in str(i) for i in adata.obs.index[:10]):
        if not adata.obs.index.is_unique:
            adata.obs.reset_index(drop=True, inplace=True)
        adata.obs['cell_barcode'] = adata.obs.index.astype(str)
        adata.obs.set_index('cell_barcode', inplace=True)

    if 'counts' in adata.layers:
        adata.X = adata.layers['counts'].copy()
    elif adata.raw is not None:
        adata.X = adata.raw.X.copy()

    sc.pp.filter_genes(adata, min_cells=3)
    
    all_batches = adata.obs[BATCH_KEY].unique().tolist()
    adata_by_batch = {batch: adata[adata.obs[BATCH_KEY] == batch].copy() for batch in all_batches}
    all_results =[]

    for i, query_batch_id in enumerate(all_batches):
        fold_num = i + 1
        logging.info(f"=====[折叠 {fold_num}/{len(all_batches)}] 查询批次: {query_batch_id} =====")

        query_adata = adata_by_batch[query_batch_id]
        ref_batch_ids = [b for b in all_batches if b != query_batch_id]
        reference_adata = anndata.concat([adata_by_batch[b] for b in ref_batch_ids], join='inner')

        temp_dir = os.path.join(OUTPUT_DIR, f"temp_fold_{fold_num}")
        try:
            export_adata_to_csv_mtx(reference_adata, temp_dir, "ref")
            export_adata_to_csv_mtx(query_adata, temp_dir, "query")
            predictions_path = os.path.join(temp_dir, 'predictions.csv')

            # 运行 Seurat R 脚本
            command =[
                "Rscript", os.path.abspath(R_SCRIPT_PATH).replace('\\', '/'),
                "--data_dir", os.path.abspath(temp_dir).replace('\\', '/'),
                "--celltype_key", CELL_TYPE_KEY,
                "--out", os.path.abspath(predictions_path).replace('\\', '/')
            ]
            subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')

            # 读取预测结果
            predictions_df = pd.read_csv(predictions_path, dtype={'cell_barcode': str}).set_index('cell_barcode')
            merged_df = query_adata.obs.join(predictions_df, how='left')

            pred_labels = merged_df['predicted.id'].astype(str)
            # 【核心修复】：对于预测失败的NA填充为 Unknown，公平惩罚
            pred_labels[pred_labels.str.lower().isin(['nan', 'na', '<na>'])] = "Unknown"

            # 【GT Remapping】：开放集公平评估
            true_labels = merged_df[CELL_TYPE_KEY].astype(str).copy()
            ref_classes = set(reference_adata.obs[CELL_TYPE_KEY].astype(str).unique())
            unseen_mask = ~true_labels.isin(ref_classes)
            if unseen_mask.any():
                true_labels[unseen_mask] = "Unknown"

            # 计算指标
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
            logging.info(f"评估完成: Macro F1 = {metrics['f1_macro']:.4f}")

        except Exception as e:
            logging.error(f"折叠 {fold_num} 执行失败: {e}", exc_info=True)

    if all_results:
        results_df = pd.DataFrame(all_results).set_index('fold')
        mean_row = results_df.mean().to_frame('mean').T
        std_row = results_df.std().to_frame('std').T
        summary_df = pd.concat([results_df, mean_row, std_row])
        output_csv_path = os.path.join(OUTPUT_DIR, f"benchmark_results_{DATASET_NAME}.csv")
        summary_df.to_csv(output_csv_path)
        logging.info(f"基准测试完成！已保存至: {output_csv_path}")

if __name__ == "__main__":
    main()
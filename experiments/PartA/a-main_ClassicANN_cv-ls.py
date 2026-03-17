import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import os
import sys
import warnings
import gc
import logging
from anndata import concat
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    # 确保路径能够找到 ClassicANN
    sys.path.append(os.path.join(os.path.dirname(__file__), 'NetWork')) # 根据你的实际结构微调
    import ClassicANN as ca
except ImportError:
    print("错误：无法导入 'ClassicANN.py'。请确保路径正确。")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def split_and_preprocess_batches_classic(adata_full_filtered, output_dir, batch_obs_key, base_filename):
    """分割批次的辅助函数 (简化版，无逻辑变动)"""
    os.makedirs(output_dir, exist_ok=True)
    unique_batches = adata_full_filtered.obs[batch_obs_key].unique()
    processed_batches_list = []
    for batch_id in unique_batches:
        adata_batch = adata_full_filtered[adata_full_filtered.obs[batch_obs_key] == batch_id].copy()
        if adata_batch.n_obs == 0: continue
        slice_name = f"{base_filename}_batch_{str(batch_id).replace(' ', '_')}"
        adata_batch.obs['slice_name'] = slice_name
        output_batch_path = os.path.join(output_dir, f"{slice_name}.h5ad")
        adata_batch.write_h5ad(output_batch_path)
        processed_batches_list.append(adata_batch)
    return processed_batches_list

def main():
    print("--- 经典机器学习方法交叉验证脚本 (PCA Shared) ---")
    warnings.filterwarnings('ignore')

    # 配置参数
    single_full_h5ad_input = 'E:/Project/Work2/Cell_datasets/Kamath_NatNeurosci_2022_35513515.h5ad' # 使用您的路径
    batch_obs_key_for_splitting = 'cause_of_death'
    cell_type_col_name = 'authors_main_cell_type'
    output_base_dir = "E:/Project/Work2"
    
    dataset_name = os.path.splitext(os.path.basename(single_full_h5ad_input))[0].replace(" ", "_")
    classic_cv_results_dir = os.path.join(output_base_dir, "CV_ClassicANN_Results", dataset_name)
    raw_processed_splits_dir = os.path.join(output_base_dir, "processing_cache", dataset_name)
    os.makedirs(classic_cv_results_dir, exist_ok=True)
    all_folds_filepath = os.path.join(classic_cv_results_dir, f"{dataset_name}_all_folds_long_format.csv")

    N_PCS_CLASSIC = 50
    METHODS_TO_RUN =['kNN', 'SVM', 'RandomForest', 'LogisticRegression', 'NaiveBayes', 'XGBoost', 'MLP']

    # 1. 数据加载与全局过滤
    logger.info("加载完整数据集与过滤...")
    adata_full = sc.read_h5ad(single_full_h5ad_input)
    if 'counts' in adata_full.layers:
        adata_full.X = adata_full.layers['counts'].copy()
    sc.pp.filter_genes(adata_full, min_cells=10)
    
    adatas_for_classic_cv = split_and_preprocess_batches_classic(
        adata_full, raw_processed_splits_dir, batch_obs_key_for_splitting, dataset_name
    )
    del adata_full; gc.collect()

    if os.path.exists(all_folds_filepath):
        os.remove(all_folds_filepath) # 每次运行刷新结果

    # 2. 交叉验证循环
    for i_cv_fold in range(len(adatas_for_classic_cv)):
        query_adata = adatas_for_classic_cv[i_cv_fold].copy()
        query_slice_name = query_adata.obs['slice_name'].iloc[0]
        ref_adatas_list =[ad.copy() for idx, ad in enumerate(adatas_for_classic_cv) if idx != i_cv_fold]
        
        logger.info(f"\n===== CV Fold {i_cv_fold + 1}/{len(adatas_for_classic_cv)}: Query = {query_slice_name} =====")
        
        adata_ref_merged = concat(ref_adatas_list, join='inner')
        common_genes = adata_ref_merged.var_names.intersection(query_adata.var_names)
        adata_ref_merged = adata_ref_merged[:, common_genes].copy()
        query_adata = query_adata[:, common_genes].copy()

        # 【核心修复 1】：统一预处理基准（标准化 + 对数化）
        for obj in [adata_ref_merged, query_adata]:
            sc.pp.normalize_total(obj, target_sum=1e4)
            sc.pp.log1p(obj)

        # PCA 降维特征提取 (拟合于 Reference，应用到 Query，非常标准)
        scaler_ref = StandardScaler(with_mean=False)
        pca_ref = PCA(n_components=N_PCS_CLASSIC, random_state=42)
        
        X_ref_scaled = scaler_ref.fit_transform(adata_ref_merged.X)
        X_ref_features = pca_ref.fit_transform(X_ref_scaled)
        
        X_query_scaled = scaler_ref.transform(query_adata.X)
        X_query_features = pca_ref.transform(X_query_scaled)

        # 提取文本标签
        labels_ref_str = adata_ref_merged.obs[cell_type_col_name].astype(str).values
        true_labels_query_str = query_adata.obs[cell_type_col_name].astype(str).values

        # 【核心修复 2】：开放集 (Open-set) GT Remapping
        ref_classes = set(labels_ref_str)
        unseen_mask = ~np.isin(true_labels_query_str, list(ref_classes))
        if unseen_mask.any():
            logger.info(f"    [GT Remapping] Query 包含未见类，已将 {unseen_mask.sum()} 个细胞映射为 'Unknown'")
            true_labels_query_str[unseen_mask] = "Unknown"

        # 执行机器学习评估
        fold_info = {'fold': query_slice_name}
        results_this_fold = ca.run_and_evaluate_all_classic_methods(
            X_ref_features=X_ref_features,
            X_query_features=X_query_features,
            labels_ref_str=labels_ref_str,
            true_labels_query_str=true_labels_query_str,
            fold_info=fold_info,
            methods_to_run=METHODS_TO_RUN
        )
        
        if results_this_fold:
            pd.DataFrame(results_this_fold).to_csv(all_folds_filepath, mode='a', header=not os.path.exists(all_folds_filepath), index=False)
        
        del adata_ref_merged, query_adata, X_ref_features, X_query_features; gc.collect()

    # 3. 汇总保存
    if os.path.exists(all_folds_filepath):
        results_df = pd.read_csv(all_folds_filepath)
        metric_cols =['accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'ari', 'ami', 'nmi']
        
        pivoted_df = results_df.pivot_table(index='fold', columns='method', values=metric_cols)
        pivoted_df = pivoted_df.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
        mean_row = pivoted_df.mean().to_frame('mean').T
        std_row = pivoted_df.std().to_frame('std').T
        
        final_summary_table = pd.concat([pivoted_df, mean_row, std_row])
        final_summary_filename = os.path.join(classic_cv_results_dir, f"{dataset_name}_summary.csv")
        final_summary_table.to_csv(final_summary_filename, index=True)
        print(f"\n✅ 格式化摘要表已保存到: {final_summary_filename}")

if __name__ == '__main__':
    main()
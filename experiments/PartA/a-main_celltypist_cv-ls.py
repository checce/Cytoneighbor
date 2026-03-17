import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import sys
import warnings
import gc # 导入垃圾回收模块

# 导入 scikit-learn 评估指标
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    adjusted_rand_score,
    adjusted_mutual_info_score, # AMI
    normalized_mutual_info_score # NMI
)
import matplotlib.pyplot as plt # 导入绘图库
import seaborn as sns
# 导入 anndata 的 concat 函数
from anndata import concat
# 导入 celltypist 库
import celltypist
import importlib # 导入 importlib 用于在交互式环境中重载模块
import glob # 用于查找现有文件
import logging # 导入 logging 模块
import time # 用于计时（如果需要）
import scipy.sparse # 用于处理稀疏矩阵（例如检查值）
# 导入 pandas api types 以检查 dtype
from pandas.api.types import is_categorical_dtype, is_numeric_dtype
from pandas import CategoricalDtype # 导入 for creating categorical dtype

# --- 配置日志 ---
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# (省略 filter_and_split_batches 等辅助函数，保持原样即可)
# 这里仅展示 main() 中被修改的核心评估逻辑
# --- 辅助函数: 扩展评估 ---
# (此函数保持不变)
def extended_evaluation(true_labels, pred_labels, categories):
    """更详细的分类评估，包括报告和混淆矩阵。"""
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
        logger.warning("没有有效的类别可用于生成分类报告。")
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
        logger.error(f"生成完整分类报告时出错: {e_cr}")
    except Exception as e_other_cr:
         logger.error(f"生成完整分类报告时发生未知错误: {e_other_cr}")

    unknown_like_predictions = {'Unknown', 'unknown', 'Unassigned', 'unassigned', 'nan', 'none', 'N/A'}
    pred_labels_str = np.asarray(pred_labels).astype(str)
    true_labels_str = np.asarray(true_labels).astype(str)

    known_mask = ~np.isin(pred_labels_str, list(unknown_like_predictions)) & \
                 ~np.isin(true_labels_str, list(unknown_like_predictions))

    filtered_true = true_labels_str[known_mask]
    filtered_pred = pred_labels_str[known_mask]

    if len(filtered_true) > 0 and len(filtered_pred) > 0:
        print("\n已知样本分类报告 (排除了真实或预测为Unknown/unassigned的样本):")
        known_cats_for_cm_report = sorted(list(set(filtered_true) | set(filtered_pred)))
        if not known_cats_for_cm_report:
             print("过滤 'Unknown' 后没有有效的已知类别进行报告。")
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
                    plt.title("混淆矩阵 (已知真实标签 -> 已知预测标签)")
                    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
                    plt.tight_layout(); plt.show(); plt.close('all')
                except ValueError as e_cm:
                    logger.error(f"绘制混淆矩阵时出错: {e_cm}")
                    plt.close('all')
                except Exception as e_other_cm:
                     logger.error(f"绘制混淆矩阵时发生未知错误: {e_other_cm}")
                     plt.close('all')
            except ValueError as e_cr_filtered:
                 logger.error(f"生成过滤后的分类报告时出错: {e_cr_filtered}")
            except Exception as e_other_cr_filtered:
                 logger.error(f"生成过滤后的分类报告时发生未知错误: {e_other_cr_filtered}")
    else:
        print("\n过滤 'Unknown' 等标签后，没有足够的已知样本进行已知类别的分类报告或混淆矩阵绘制。")


# --- filter_and_split_batches 函数 ---
# (此函数保持不变)
def filter_and_split_batches(
    input_h5ad_path: str,
    output_dir: str,
    batch_obs_key: str,
    min_cells_for_gene_filter: int,
    counts_layer_key: str = 'counts'
) -> list:
    """
    加载单个AnnData对象,在整个数据集上执行基因过滤,然后按批次键分割,
    确保.X为数值类型,保存每个批次到output_dir,并返回AnnData对象列表。
    """
    logger.info(f"正在加载、过滤和分割文件: {input_h5ad_path}...")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"分割后的缓存文件将保存到: {output_dir}")

    try:
        logger.info("加载完整的 AnnData 对象...")
        adata_full = sc.read_h5ad(input_h5ad_path)
        logger.info(f"成功加载 {input_h5ad_path}。初始维度: {adata_full.shape}")
    except Exception as e:
        logger.error(f"加载 {input_h5ad_path} 失败", exc_info=True)
        return []

    logger.info(f"在整个数据集上应用基因过滤 (min_cells={min_cells_for_gene_filter})...")
    if counts_layer_key and counts_layer_key in adata_full.layers:
        logger.info(f"使用 a.layers['{counts_layer_key}'] 进行基因过滤...")
        adata_for_filtering = ad.AnnData(X=adata_full.layers[counts_layer_key], var=adata_full.var.copy())
    else:
        logger.info(f"使用 a.X 进行基因过滤...")
        adata_for_filtering = ad.AnnData(X=adata_full.X, var=adata_full.var.copy())
    n_genes_before_filter = adata_for_filtering.n_vars
    sc.pp.filter_genes(adata_for_filtering, min_cells=min_cells_for_gene_filter)
    genes_to_keep = adata_for_filtering.var_names
    logger.info(f"全局基因过滤完成: {n_genes_before_filter} -> {len(genes_to_keep)} 基因。")
    del adata_for_filtering
    gc.collect()
    adata_full_filtered = adata_full[:, genes_to_keep].copy()
    logger.info(f"已基于过滤后的基因列表创建最终数据集。当前维度: {adata_full_filtered.shape}")
    del adata_full
    gc.collect()

    if batch_obs_key not in adata_full_filtered.obs.columns:
        logger.error(f"批次键 '{batch_obs_key}' 在 adata.obs 中未找到。可用列: {list(adata_full_filtered.obs.columns)}")
        del adata_full_filtered; gc.collect()
        return []

    unique_batches = adata_full_filtered.obs[batch_obs_key].unique()
    logger.info(f"找到 {len(unique_batches)} 个唯一批次用于分割: {list(unique_batches)}")
    base_filename = os.path.splitext(os.path.basename(input_h5ad_path))[0].replace(" ", "_").replace("/", "-").replace("\\", "-")

    processed_batches_list = []
    for i, batch_id in enumerate(unique_batches):
        batch_id_str = str(batch_id)
        logger.info(f"--- 处理批次 {i+1}/{len(unique_batches)}: '{batch_id_str}' ---")
        adata_batch = None
        try:
            logger.debug(f"子集化 AnnData for 批次 '{batch_id_str}'.")
            adata_batch = adata_full_filtered[adata_full_filtered.obs[batch_obs_key] == batch_id].copy()
            logger.info(f"批次 '{batch_id_str}' 子集化完成。维度: {adata_batch.shape}")
            gc.collect()
            logger.info("执行初步处理 (从 layer 加载 counts, 确保数值型 X)...")
            if counts_layer_key and counts_layer_key in adata_batch.layers:
                logger.info(f"从 adata_batch.layers['{counts_layer_key}'] 读取数据到 .X...")
                layer_data = adata_batch.layers[counts_layer_key]
                if scipy.sparse.issparse(layer_data):
                    layer_data = layer_data.tocsr()
                adata_batch.X = layer_data.copy()
            else:
                 logger.warning(f"未找到 '{counts_layer_key}' 层, 假设 .X 包含原始计数。")
                 if adata_batch.X is None:
                    raise ValueError(f"未找到 '{counts_layer_key}' 层, 且 .X 为空。")
            if hasattr(adata_batch.X, 'dtype') and not np.issubdtype(adata_batch.X.dtype, np.number):
                try: adata_batch.X = adata_batch.X.astype(np.float32)
                except: continue
            elif hasattr(adata_batch.X, 'dtype') and adata_batch.X.dtype != np.float32:
                 try: adata_batch.X = adata_batch.X.astype(np.float32)
                 except: pass
            if adata_batch.n_obs == 0 or adata_batch.n_vars == 0:
                logger.warning(f"批次 '{batch_id_str}' 为空，已跳过。")
                continue
            safe_batch_id_fn = batch_id_str.replace(" ", "_").replace("/", "-").replace("\\", "-")
            slice_name = f"{base_filename}_batch_{safe_batch_id_fn}"
            adata_batch.uns['slice_name'] = slice_name
            if 'slice_name' not in adata_batch.obs.columns:
                adata_batch.obs['slice_name'] = slice_name
            output_batch_path = os.path.join(output_dir, f"{slice_name}.h5ad")
            try:
                logger.info(f"正在将分割后的数据 (基因已全局过滤) 保存到 {output_batch_path}")
                temp_adata_to_save = ad.AnnData(X=adata_batch.X, obs=adata_batch.obs.copy(), var=adata_batch.var.copy())
                if 'slice_name' in adata_batch.uns: temp_adata_to_save.uns['slice_name'] = adata_batch.uns['slice_name']
                temp_adata_to_save.write_h5ad(output_batch_path)
                logger.info(f"已保存缓存文件: {slice_name}")
                processed_batches_list.append(adata_batch)
            except Exception as e_save_batch:
                logger.warning(f"无法将批次 '{batch_id_str}' 保存到 {output_batch_path}: {e_save_batch}。")
                processed_batches_list.append(adata_batch)
        except Exception as e_batch_proc:
            logger.error(f"处理批次 '{batch_id_str}' 时发生错误: {e_batch_proc}", exc_info=True)
        finally:
            if 'adata_batch' in locals() and adata_batch is not None:
                del adata_batch
            gc.collect()
            logger.info(f"--- 已完成处理批次 '{batch_id_str}' ---")
    if 'adata_full_filtered' in locals() and adata_full_filtered is not None:
        del adata_full_filtered
    gc.collect()
    logger.info("所有批次已分割并缓存完成。")
    return processed_batches_list

def main():
    print("--- CellTypist 交叉验证脚本 ---")
    warnings.filterwarnings('ignore')

    # --- 1. 配置参数 ---
    print("步骤 1: 配置参数")
    # 输入文件路径
    single_full_h5ad_input = '/home/ioag/yyq/Cell_datasets/Mouse_Kidney_mod1.h5ad'
    # 批次和细胞类型列名
    batch_obs_key_for_splitting = 'batch'
    cell_type_col = 'cell_type'
    
    # 输出目录
    output_base_dir = "/home/ioag/yyq/"
    cv_results_parent_subdir = "CV_Celltpist_Results"
    dataset_name_subdir = os.path.splitext(os.path.basename(single_full_h5ad_input))[0].replace(" ", "_")
    final_output_dir = os.path.join(output_base_dir, cv_results_parent_subdir, dataset_name_subdir)
    # raw_processed_splits_dir = os.path.join(output_base_dir, "processing_cache", dataset_name_subdir) # 建议子目录加上数据集名称以防冲突
    raw_processed_splits_dir = os.path.join(output_base_dir, "processing_cache")
    
    # 创建目录
    os.makedirs(raw_processed_splits_dir, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)
    
    logger.info(f"本次运行的所有结果将保存到: {final_output_dir}")

    # --- 关键参数定义 (修复漏洞) ---
    MIN_CELLS_FOR_GENE_FILTER = 10  # 基因过滤阈值
    COUNTS_LAYER_KEY = 'counts'     # 原始计数层名称
    
    # --- CellTypist 训练参数 ---
    # 使用官方推荐的默认参数
    # n_jobs = -1 使用所有可用CPU核心加速
    # check_expression = True 检查输入数据格式（如是否为log1p）
    # feature_selection: 如果基因数 > 300，CellTypist 默认会做特征选择。我们在循环中动态设置。
    celltypist_base_train_args = {
        'check_expression': True,
        'n_jobs': -1,
        # 'max_iter': 1000 # 默认是 1000，通常不需要显式设置，除非想限制时间。为了公平，保持默认。
    }

    # --- 2. 数据加载与子集准备 ---
    print("\n步骤 2: 数据加载与子集准备...")
    adatas_for_cv = []
    safe_base_input_filename = os.path.splitext(os.path.basename(single_full_h5ad_input))[0].replace(" ", "_").replace("/", "-").replace("\\", "-")
    split_file_pattern = f"{safe_base_input_filename}_batch_*.h5ad"
    potential_split_files = sorted(glob.glob(os.path.join(raw_processed_splits_dir, split_file_pattern)))
    
    if len(potential_split_files) >= 2:
        print(f"  在目录 '{raw_processed_splits_dir}' 中找到 {len(potential_split_files)} 个潜在的现有缓存文件。正在尝试加载...")
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
                    print(f"    已加载缓存文件: {s_name_cleaned} ({ad_item.n_obs}x{ad_item.n_vars})")
                    loaded_count += 1
                else:
                    warnings.warn(f"    缓存文件 {path} 加载后为空，已跳过。")
                    if 'ad_item' in locals(): del ad_item; gc.collect()
            except Exception as e:
                warnings.warn(f"    警告: 加载缓存文件 {path} 失败: {e}")
        if loaded_count >= 2:
            print(f"  成功加载 {loaded_count} 个缓存文件。跳过直接分割。")
            adatas_for_cv = temp_adatas_raw
        else:
            print(f"  只加载了 {loaded_count} 个缓存文件 (少于 CV 所需的 2 个)。继续分割完整数据集。")
    
    if len(adatas_for_cv) < 2:
        print(f"  没有找到足够的现有缓存文件。正在调用全局过滤和分割逻辑处理 '{single_full_h5ad_input}'...")
        adatas_for_cv = filter_and_split_batches(
            input_h5ad_path=single_full_h5ad_input,
            output_dir=raw_processed_splits_dir,
            batch_obs_key=batch_obs_key_for_splitting,
            min_cells_for_gene_filter=MIN_CELLS_FOR_GENE_FILTER,
            counts_layer_key=COUNTS_LAYER_KEY
        )
    
    if not adatas_for_cv or len(adatas_for_cv) < 2:
        print("错误：准备阶段后剩余的有效子集少于两个。无法执行交叉验证。")
        sys.exit(1)
        
    valid_slice_names_for_cv = [ad.uns.get('slice_name', f'UnnamedSplit_{i}') for i, ad in enumerate(adatas_for_cv)]
    print(f"使用 {len(adatas_for_cv)} 个有效子集进行交叉验证。")
    
    # --- 3. 开始交叉验证 ---
    print("\n步骤 3: 开始交叉验证...")
    cv_results_list =[]
    
    for i_query_cv in range(len(adatas_for_cv)):
        query_adata_original = adatas_for_cv[i_query_cv].copy()
        query_s_name_cv = query_adata_original.uns.get('slice_name', f'fold_{i_query_cv}')
        logger.info(f"\n--- CV Fold {i_query_cv + 1}/{len(adatas_for_cv)}: 查询集 = {query_s_name_cv} ---")
        
        # 构建训练集
        train_adatas_this_fold_raw =[ad.copy() for idx, ad in enumerate(adatas_for_cv) if idx != i_query_cv]
        adata_train_merged_raw = concat(train_adatas_this_fold_raw, join='inner')

        # 🚨【核心修复 1】：剔除 Reference 中标签为空 (NaN) 的细胞
        # CellTypist 依赖 sklearn SGDClassifier，绝不接受 NaN 作为 target(y)
        valid_train_mask = ~pd.isna(adata_train_merged_raw.obs[cell_type_col])
        num_dropped = sum(~valid_train_mask)
        if num_dropped > 0:
            logger.warning(f"  [数据清洗] 从 Reference 中剔除了 {num_dropped} 个标签为 NaN 的细胞。")
            adata_train_merged_raw = adata_train_merged_raw[valid_train_mask].copy()
        
        # 强制转为字符串，防止后续类型漂移
        adata_train_merged_raw.obs[cell_type_col] = adata_train_merged_raw.obs[cell_type_col].astype(str)

        # 🚨【核心修复 2】：清空表达矩阵 X 中的 NaN 
        if sp.issparse(adata_train_merged_raw.X):
            adata_train_merged_raw.X.data = np.nan_to_num(adata_train_merged_raw.X.data, nan=0.0)
        else:
            adata_train_merged_raw.X = np.nan_to_num(adata_train_merged_raw.X, nan=0.0)
            
        # CellTypist 预处理 (log1p)
        # 必须先进行 normalize_total，因为 CellTypist 依赖于归一化后的数据
        sc.pp.normalize_total(adata_train_merged_raw, target_sum=1e4)
        sc.pp.log1p(adata_train_merged_raw)
        
        # 准备训练参数
        train_args = celltypist_base_train_args.copy()
        # 如果特征数（基因数）大于 300，开启特征选择（CellTypist 默认行为）
        if adata_train_merged_raw.n_vars >= 300:
            train_args['feature_selection'] = True
        else:
            train_args['feature_selection'] = False
            
        logger.info(f"训练 CellTypist 模型 (使用默认参数, feature_selection={train_args['feature_selection']})...")
        
        try:
            # 训练模型
            ct_model = celltypist.train(adata_train_merged_raw, labels=cell_type_col, **train_args)
            
            # 查询集预处理
            query_adata_for_pred = query_adata_original.copy()
            sc.pp.normalize_total(query_adata_for_pred, target_sum=1e4)
            sc.pp.log1p(query_adata_for_pred)
            
            # 预测
            # mode='best match' 是默认且推荐的模式
            predictions = celltypist.annotate(query_adata_for_pred, model=ct_model, majority_voting=False)
            pred_labels_cv = predictions.predicted_labels['predicted_labels'].astype(str)
            
            # 【修改点 2：GT Remapping (未知类重映射)】
            # 确保使用配置好的 cell_type_col
            true_labels_cv = query_adata_original.obs[cell_type_col].astype(str).copy()
            ref_classes = set(adata_train_merged_raw.obs[cell_type_col].astype(str).unique())
            unseen_mask = ~true_labels_cv.isin(ref_classes)
            
            if unseen_mask.any():
                logger.info(f"Query 中发现未见类型，将 {unseen_mask.sum()} 个细胞映射为 'Unknown'")
                true_labels_cv[unseen_mask] = "Unknown"
            
            # 【修改点 3：补全指标】
            acc_cv = accuracy_score(true_labels_cv, pred_labels_cv)
            unique_labels = list(set(true_labels_cv) | set(pred_labels_cv))
            f1_micro = f1_score(true_labels_cv, pred_labels_cv, labels=unique_labels, average='micro', zero_division=0)
            f1_macro = f1_score(true_labels_cv, pred_labels_cv, labels=unique_labels, average='macro', zero_division=0)
            f1_weighted = f1_score(true_labels_cv, pred_labels_cv, labels=unique_labels, average='weighted', zero_division=0)
            ari_cv = adjusted_rand_score(true_labels_cv, pred_labels_cv)
            ami_cv = adjusted_mutual_info_score(true_labels_cv, pred_labels_cv)
            nmi_cv = normalized_mutual_info_score(true_labels_cv, pred_labels_cv)
            
            # 记录结果
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
            logger.error(f"Fold {i_query_cv + 1} 失败: {e}", exc_info=True)
            cv_results_list.append({
                'fold_query_slice': query_s_name_cv,
                'status': 'Failed',
                'error': str(e)
            })
        
        # 垃圾回收
        del query_adata_original, train_adatas_this_fold_raw, adata_train_merged_raw
        if 'ct_model' in locals(): del ct_model
        if 'query_adata_for_pred' in locals(): del query_adata_for_pred
        gc.collect()

    # --- 4. 汇总保存逻辑 ---
    print("\n步骤 4: 结果汇总...")
    if cv_results_list:
        results_df = pd.DataFrame(cv_results_list)
        
        # 保存详细结果
        results_csv_path = os.path.join(final_output_dir, "celltypist_summary.csv")
        results_df.to_csv(results_csv_path, index=False)
        logger.info(f"CellTypist 交叉验证详细结果已保存至: {results_csv_path}")
        
        # 计算并打印平均值 (仅针对成功运行的 Fold)
        success_df = results_df[results_df['status'] == 'Success']
        if not success_df.empty:
            mean_metrics = success_df[['accuracy', 'f1_macro', 'f1_weighted', 'ari', 'nmi']].mean()
            print("\n平均性能指标:")
            print(mean_metrics)
    else:
        logger.warning("未生成任何结果。")

    logger.info("CellTypist 交叉验证完成。")

if __name__ == '__main__':
    main()
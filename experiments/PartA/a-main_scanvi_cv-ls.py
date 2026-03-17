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
import torch  # [新增] 用于显存管理
from anndata import concat

# 尝试导入 scvi-tools
try:
    import scvi
except ImportError:
    print("错误：scvi-tools 未安装。请运行 'pip install scvi-tools'")
    sys.exit(1)

from sklearn.metrics import (
    accuracy_score, f1_score, adjusted_rand_score,
    adjusted_mutual_info_score, normalized_mutual_info_score
)

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. 辅助函数
# ==============================================================================

def evaluate_predictions(true_labels, pred_labels, fold_info, method_name):
    """计算 7 大核心指标"""
    if true_labels is None or pred_labels is None:
        return {**fold_info, 'method': method_name, 'error': 'Missing labels'}
    try:
        # 统一转换为字符串，防止类别编码错乱
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
        logger.error(f"评估方法 {method_name} 时出错: {e}", exc_info=True)
        return {**fold_info, 'method': method_name, 'error': str(e)}

def run_scANVI_annotation(adata_ref, adata_query, batch_key, cell_type_key, fold_info, model_dir):
    """在一个交叉验证折叠中训练 scANVI (Arches 模式)"""
    method_name = "scANVI"
    start_time = time.time()
    
    model_ref = None
    model_query = None

    try:
        # --- 1. 训练 Reference ---
        logger.info("  1. 设置并训练参考 scANVI 模型...")
        # scVI 推荐使用 raw counts
        scvi.model.SCANVI.setup_anndata(
            adata_ref,
            layer="counts",
            batch_key=batch_key,
            labels_key=cell_type_key,
            unlabeled_category="Unknown"
        )
        
        # 使用默认参数确保公平比较 (latent=30 是标准配置)
        model_ref = scvi.model.SCANVI(adata_ref, n_latent=30)
        
        # [修改] 移除 max_epochs 限制，使用 early_stopping 确保收敛
        model_ref.train(
            early_stopping=True,
            train_size=0.9,
            early_stopping_patience=10,
            datasplitter_kwargs={'drop_last': True}
        )
        
        # 保存 Reference 模型以便 Query 使用 Arches 加载
        ref_model_path = os.path.join(model_dir, f"scANVI_ref_fold_{fold_info['fold']}")
        model_ref.save(ref_model_path, overwrite=True, save_anndata=False)

        # --- 2. 训练 Query (Arches) ---
        logger.info("  2. 使用 Arches 框架映射查询集...")
        
        # 加载 Query 数据到 Reference 模型架构
        model_query = scvi.model.SCANVI.load_query_data(
            adata_query,
            ref_model_path,
        )
        
        # 训练 Query (半监督/无监督适应)
        model_query.train(
            max_epochs=100, # Arches 适应通常不需要特别久，但给够上限
            plan_kwargs=dict(weight_decay=0.0),
            early_stopping=True,
            early_stopping_patience=10
        )
        
        # 预测
        pred_labels = model_query.predict()
        
        # --- 3. 评估 ---
        # 获取真实标签用于评估 (此时已包含 Unknown 重映射)
        true_labels = adata_query.obs[cell_type_key].values
        
        results = evaluate_predictions(true_labels, pred_labels, fold_info, method_name)
        results['runtime_seconds'] = time.time() - start_time
        
        logger.info(f"  scANVI 评估完毕: Macro-F1 = {results.get('f1_macro', 0.0):.4f}")
        return results

    except Exception as e:
        logger.error(f"scANVI 在折叠 {fold_info['fold']} 中失败: {e}", exc_info=True)
        return {**fold_info, 'method': method_name, 'error': str(e)}

    finally:
        # [关键] 局部显存清理
        del model_ref
        del model_query
        gc.collect()

# ==============================================================================
# 2. 主流程
# ==============================================================================

def main():
    print("--- scANVI 交叉验证脚本 (Optimized) ---")
    warnings.filterwarnings('ignore')

    # --- 配置参数 ---
    single_full_h5ad_input = "/content/drive/MyDrive/Cell_datasets/Zebrafish embryonic cells.h5ad"
    batch_obs_key = 'batch'
    cell_type_key = 'cell_type'
    
    # 高变基因数量 (推荐 2000-4000)
    N_HVG = 3000 

    output_base_dir = "/content/drive/MyDrive/"
    dataset_name = os.path.splitext(os.path.basename(single_full_h5ad_input))[0].replace(" ", "_")
    scanvi_cv_results_dir = os.path.join(output_base_dir, "CV_scANVI_Results", dataset_name)
    model_save_dir = os.path.join(output_base_dir, "model_cache", dataset_name, "scANVI")

    os.makedirs(scanvi_cv_results_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    
    logger.info(f"数据集: {dataset_name}")
    logger.info(f"结果保存路径: {scanvi_cv_results_dir}")

    # --- 数据加载与全局预处理 ---
    logger.info("正在加载数据...")
    try:
        adata_full = sc.read_h5ad(single_full_h5ad_input)
    except Exception as e:
        logger.error(f"加载失败: {e}")
        sys.exit(1)

        # ================= [修复开始] =================
    # 修复逻辑：确保 adata.X 不为空，否则 filter_genes 会报错
    if adata_full.X is None:
        logger.warning("检测到 adata.X 为 None，尝试从其他位置恢复数据...")
        if 'counts' in adata_full.layers:
            logger.info("从 adata.layers['counts'] 恢复 adata.X")
            adata_full.X = adata_full.layers['counts'].copy()
        elif adata_full.raw is not None:
            logger.info("从 adata.raw.X 恢复 adata.X")
            adata_full.X = adata_full.raw.X.copy()
        else:
            logger.error("严重错误：adata.X, layers['counts'], raw.X 均为空，无法处理数据。")
            sys.exit(1)

    # 1. 确保有 'counts' 层 (scVI 训练必须使用)
    if 'counts' not in adata_full.layers:
        logger.info("未找到 .layers['counts']，将 .X (原始计数) 备份到 layers。")
        adata_full.layers['counts'] = adata_full.X.copy()
    
    # 确保 X 和 layers['counts'] 都是 csr_matrix (节省内存并加速)
    from scipy import sparse
    if not sparse.issparse(adata_full.X):
        adata_full.X = sparse.csr_matrix(adata_full.X)
    if not sparse.issparse(adata_full.layers['counts']):
        adata_full.layers['counts'] = sparse.csr_matrix(adata_full.layers['counts'])
    # ================= [修复结束] =================
    
    # 2. 全局基础过滤
    sc.pp.filter_genes(adata_full, min_cells=10)
    
    # 3. 预计算 log1p (用于 HVG 计算，注意不要覆盖 counts 层)
    # 这一步是为了后续在循环内快速计算 HVG
    if 'log1p' not in adata_full.uns:
        target_sum = 1e4
        sc.pp.normalize_total(adata_full, target_sum=target_sum)
        sc.pp.log1p(adata_full)
    
    # 4. 按批次切分对象
    unique_batches = adata_full.obs[batch_obs_key].unique()
    adatas_for_cv = [adata_full[adata_full.obs[batch_obs_key] == b].copy() for b in unique_batches]
    slice_names_for_cv = [f"{dataset_name}_{str(b).replace(' ', '_')}" for b in unique_batches]
    
    # 释放全量数据内存
    del adata_full
    gc.collect()

    logger.info(f"数据切分完成，共 {len(adatas_for_cv)} 个批次。")

    # --- 交叉验证循环 ---
    all_cv_results = []
    
    for i_cv_fold, query_adata_orig in enumerate(adatas_for_cv):
        adata_ref_merged = None
        adata_query = None
        
        try:
            query_slice_name = slice_names_for_cv[i_cv_fold]
            logger.info(f"\n===== CV Fold {i_cv_fold + 1}/{len(adatas_for_cv)}: Query = {query_slice_name} =====")

            # 1. 构建参考集
            ref_adatas_list = [ad for idx, ad in enumerate(adatas_for_cv) if idx != i_cv_fold]
            adata_ref_merged = concat(ref_adatas_list, join='outer', merge='same')
            adata_query = query_adata_orig.copy()

            # 2. [关键优化] 基于参考集计算高变基因 (HVG)
            # 这能显著降低 scANVI 的显存占用并提高精度
            logger.info(f"  正在计算高变基因 (Top {N_HVG})...")
            
            # 使用 seurat 方法计算 HVG (需要 log1p 数据，我们在全局预处理已做)
            sc.pp.highly_variable_genes(
                adata_ref_merged,
                n_top_genes=N_HVG,
                subset=True, # 直接原地切片参考集
                flavor="seurat",
                batch_key=batch_obs_key # 考虑参考集内部的批次效应
            )
            hvg_genes = adata_ref_merged.var_names
            
            # 将查询集对齐到相同的 HVG
            adata_query = adata_query[:, hvg_genes].copy()
            logger.info(f"  特征对齐完成。训练特征数: {len(hvg_genes)}")

            # 3. 标签清洗与 GT Remapping (Open-Set 逻辑)
            # 清理空格
            for ad_obj in [adata_ref_merged, adata_query]:
                ad_obj.obs[cell_type_key] = ad_obj.obs[cell_type_key].astype(str).str.strip()
            
            # 检查未见类
            ref_labels = set(adata_ref_merged.obs[cell_type_key].unique())
            query_labels = adata_query.obs[cell_type_key].astype(str).values
            
            # 将 Query 中 Reference 没有的类别重命名为 "Unknown"
            # scANVI 是闭集模型，训练时不知道这些类，预测时会强行归类，导致错误。
            # 这里的重命名是为了在 calculate metrics 时，让 GT=Unknown，Pred=KnownClass，
            # 从而正确记为 False Positive/Negative，体现模型对新类的鲁棒性差。
            is_unknown = ~np.isin(query_labels, list(ref_labels))
            if is_unknown.any():
                logger.info(f"  [Open-Set] 将 {is_unknown.sum()} 个未见类别细胞标记为 'Unknown'")
                # 我们修改内存中的 query labels 用于评估，不修改原始数据
                query_labels[is_unknown] = "Unknown"
                adata_query.obs[cell_type_key] = query_labels

            # 4. 运行 scANVI
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
            logger.error(f"  Fold {i_cv_fold+1} 发生严重错误: {e}", exc_info=True)
        
        finally:
            # 5. [关键优化] 显存深度清理
            logger.info("  正在清理显存...")
            del adata_ref_merged
            del adata_query
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"  CUDA 缓存已清空。当前显存占用: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

    # --- 结果汇总 ---
    print("\n--- 结果汇总 ---")
    if all_cv_results:
        results_df = pd.DataFrame(all_cv_results)
        
        # 保存长格式
        long_path = os.path.join(scanvi_cv_results_dir, f"{dataset_name}_scANVI_detailed.csv")
        results_df.to_csv(long_path, index=False)
        
        # 生成摘要
        metric_cols = ['accuracy', 'f1_macro', 'f1_weighted', 'ari', 'nmi']
        available_cols = [c for c in metric_cols if c in results_df.columns]
        
        if available_cols:
            summary = results_df.groupby('method')[available_cols].agg(['mean', 'std'])
            print(summary)
            summary_path = os.path.join(scanvi_cv_results_dir, f"{dataset_name}_scANVI_summary.csv")
            summary.to_csv(summary_path)
            logger.info(f"摘要已保存至: {summary_path}")
    else:
        logger.error("未生成任何结果。")

    print("\n--- 脚本结束 ---")

if __name__ == '__main__':
    # 禁用多进程加载以防止死锁
    scvi.settings.dl_num_workers = 0
    # 设置随机种子
    scvi.settings.seed = 42
    main()
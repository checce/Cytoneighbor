import os
# --- 环境变量配置 (来自 scArches 脚本的兼容性设置) ---
os.environ["SCIPY_ARRAY_API"] = "1"
os.environ["SKLEARN_ENABLE_ARRAY_API"] = "0"

import logging
import pandas as pd
import numpy as np
import anndata
import scanpy as sc
import torch  # 用于显存管理
import gc
import time
import sys
import warnings
from anndata import concat
from sklearn.neighbors import KNeighborsClassifier

# --- 尝试导入 scvi ---
try:
    import scvi
except ImportError:
    print("错误：scvi-tools 未安装。请运行 'pip install scvi-tools'")
    sys.exit(1)

from sklearn.metrics import (
    accuracy_score, f1_score, adjusted_rand_score,
    adjusted_mutual_info_score, normalized_mutual_info_score
)

# --- 全局配置 ---
# 请根据实际情况修改路径
H5AD_PATH = "/content/drive/MyDrive/Cell_datasets/Zebrafish embryonic cells.h5ad"
OUTPUT_BASE_DIR = "/content/drive/MyDrive/"
DATASET_NAME = "Zebrafish embryonic cells"

BATCH_KEY = "batch"
CELL_TYPE_KEY = "cell_type"
N_HVG = 3000
N_NEIGHBORS = 15  # scArches KNN 参数 (对齐 Cytoneighbor)
SCVI_LATENT_DIM = 30
SEED = 42

# 目录准备
DATASET_CLEAN_NAME = DATASET_NAME.replace(" ", "_")
RESULTS_DIR = os.path.join(OUTPUT_BASE_DIR, "CV_scVI_Ecosystem_Results", DATASET_CLEAN_NAME)
MODEL_CACHE_DIR = os.path.join(OUTPUT_BASE_DIR, "model_cache", DATASET_CLEAN_NAME, "scVI_Ecosystem")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
scvi.settings.seed = SEED

# ==============================================================================
# 1. 辅助函数
# ==============================================================================

def evaluate_predictions(true_labels, pred_labels, fold_info, method_name):
    """计算 7 大核心指标"""
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
        logger.error(f"评估 {method_name} 失败: {e}")
        return {**fold_info, 'method': method_name, 'error': str(e)}

def run_joint_training(adata_ref, adata_query, fold_info):
    """
    联合训练核心逻辑：
    1. 训练 Reference scVI
    2. 分支 A: scVI -> Query Mapping -> Latent -> KNN (即 scArches)
    3. 分支 B: scVI -> 转 scANVI -> Query Mapping -> Predict (即 scANVI)
    """
    results = []
    
    # -----------------------------------------------------------
    # Step 1: 训练共享的基础模型 (Reference scVI)
    # -----------------------------------------------------------
    logger.info("  [Step 1] 训练共享的 Reference scVI 模型...")
    
    # scVI Setup
    scvi.model.SCVI.setup_anndata(
        adata_ref, 
        layer="counts", 
        batch_key=BATCH_KEY, 
        labels_key=CELL_TYPE_KEY
    )
    
    model_ref_scvi = scvi.model.SCVI(adata_ref, n_latent=SCVI_LATENT_DIM)
    
    # 训练 scVI (使用早停，不限制 Epochs)
    model_ref_scvi.train(
        early_stopping=True, 
        early_stopping_patience=10, 
        train_size=0.9
    )
    
    # 保存 Reference scVI
    ref_scvi_path = os.path.join(MODEL_CACHE_DIR, f"ref_scvi_fold_{fold_info['fold']}")
    model_ref_scvi.save(ref_scvi_path, overwrite=True, save_anndata=False)

    # -----------------------------------------------------------
    # Step 2: 分支 A - scArches (scVI + KNN)
    # -----------------------------------------------------------
    try:
        logger.info("  [Branch A] 运行 scArches (scVI Mapping + KNN)...")
        start_time = time.time()
        
        # 加载 Query 到 scVI (Arches 模式)
        model_query_scvi = scvi.model.SCVI.load_query_data(
            adata_query,
            model_ref_scvi
        )
        model_query_scvi.train(
            max_epochs=200, 
            plan_kwargs=dict(weight_decay=0.0), 
            early_stopping=True
        )
        
        # 提取隐向量
        latent_ref = model_ref_scvi.get_latent_representation()
        latent_query = model_query_scvi.get_latent_representation()
        
        # KNN 分类
        knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, n_jobs=-1)
        knn.fit(latent_ref, adata_ref.obs[CELL_TYPE_KEY].astype(str))
        pred_labels_knn = knn.predict(latent_query)
        
        # 评估
        true_labels = adata_query.obs[CELL_TYPE_KEY].values
        metrics_sca = evaluate_predictions(true_labels, pred_labels_knn, fold_info, "scArches")
        metrics_sca['runtime_seconds'] = time.time() - start_time
        results.append(metrics_sca)
        
        logger.info(f"    scArches 完成: Macro-F1 = {metrics_sca.get('f1_macro', 0):.4f}")
        
        # 清理分支 A 的内存
        del model_query_scvi, latent_ref, latent_query, knn
        gc.collect()
        
    except Exception as e:
        logger.error(f"scArches 分支失败: {e}", exc_info=True)
        results.append({**fold_info, 'method': 'scArches', 'error': str(e)})

    # -----------------------------------------------------------
    # Step 3: 分支 B - scANVI (Semi-supervised Transfer)
    # -----------------------------------------------------------
    try:
        logger.info("  [Branch B] 运行 scANVI (基于 scVI 微调)...")
        start_time = time.time()
        
        # 从训练好的 scVI 初始化 scANVI (Transfer Learning)
        model_ref_scanvi = scvi.model.SCANVI.from_scvi_model(
            model_ref_scvi,
            unlabeled_category="Unknown",
            labels_key=CELL_TYPE_KEY
        )
        
        # 微调 scANVI (通常 20-50 epochs 足够)
        model_ref_scanvi.train(
            max_epochs=50, 
            early_stopping=True, 
            early_stopping_patience=5
        )
        
        # 保存 Reference scANVI
        ref_scanvi_path = os.path.join(MODEL_CACHE_DIR, f"ref_scanvi_fold_{fold_info['fold']}")
        model_ref_scanvi.save(ref_scanvi_path, overwrite=True, save_anndata=False)
        
        # 加载 Query 到 scANVI (Arches 模式)
        model_query_scanvi = scvi.model.SCANVI.load_query_data(
            adata_query,
            model_ref_scanvi
        )
        model_query_scanvi.train(
            max_epochs=100, 
            plan_kwargs=dict(weight_decay=0.0), 
            early_stopping=True
        )
        
        # scANVI 直接预测
        pred_labels_scanvi = model_query_scanvi.predict()
        
        # 评估
        true_labels = adata_query.obs[CELL_TYPE_KEY].values
        metrics_scanvi = evaluate_predictions(true_labels, pred_labels_scanvi, fold_info, "scANVI")
        metrics_scanvi['runtime_seconds'] = time.time() - start_time
        results.append(metrics_scanvi)
        
        logger.info(f"    scANVI 完成: Macro-F1 = {metrics_scanvi.get('f1_macro', 0):.4f}")
        
        # 清理分支 B 的内存
        del model_ref_scanvi, model_query_scanvi
        
    except Exception as e:
        logger.error(f"scANVI 分支失败: {e}", exc_info=True)
        results.append({**fold_info, 'method': 'scANVI', 'error': str(e)})

    # 清理共享模型
    del model_ref_scvi
    gc.collect()
    
    return results

# ==============================================================================
# 2. 主执行流程
# ==============================================================================

def main():
    logger.info("=== 开始 scVI Ecosystem (scArches & scANVI) 联合交叉验证 ===")
    
    # 1. 数据加载 (采用 scanvi 脚本中的稳健加载逻辑)
    logger.info(f"正在加载数据: {H5AD_PATH}")
    try:
        adata_full = sc.read_h5ad(H5AD_PATH)
    except Exception as e:
        logger.error(f"加载失败: {e}")
        return

    # [关键修复] 检查并恢复 .X
    if adata_full.X is None:
        logger.warning("检测到 adata.X 为 None，尝试恢复...")
        if 'counts' in adata_full.layers:
            adata_full.X = adata_full.layers['counts'].copy()
        elif adata_full.raw is not None:
            adata_full.X = adata_full.raw.X.copy()
        else:
            logger.error("无法恢复数据 (X/layers/raw 均为空)")
            return

    # 确保 counts 层存在 (scVI 必须)
    if 'counts' not in adata_full.layers:
        logger.info("备份原始计数到 layers['counts']")
        adata_full.layers['counts'] = adata_full.X.copy()

    # 稀疏化以节省内存
    from scipy import sparse
    if not sparse.issparse(adata_full.X):
        adata_full.X = sparse.csr_matrix(adata_full.X)
    if not sparse.issparse(adata_full.layers['counts']):
        adata_full.layers['counts'] = sparse.csr_matrix(adata_full.layers['counts'])

    # 全局过滤
    sc.pp.filter_genes(adata_full, min_cells=20)
    
    # 预计算 log1p (用于后续 HVG 计算，不影响 counts 层)
    if 'log1p' not in adata_full.uns:
        sc.pp.normalize_total(adata_full, target_sum=1e4)
        sc.pp.log1p(adata_full)

    # 2. 切分批次
    unique_batches = adata_full.obs[BATCH_KEY].unique().tolist()
    logger.info(f"共发现 {len(unique_batches)} 个批次。")
    
    adatas_for_cv = [adata_full[adata_full.obs[BATCH_KEY] == b].copy() for b in unique_batches]
    slice_names_for_cv = [f"{DATASET_CLEAN_NAME}_batch_{str(b).replace(' ', '_')}" for b in unique_batches]
    
    del adata_full
    gc.collect()

    # 3. 交叉验证循环
    all_cv_results = []
    
    for i, query_adata_orig in enumerate(adatas_for_cv):
        adata_ref_merged = None
        adata_query = None
        
        try:
            query_slice_name = slice_names_for_cv[i]
            logger.info(f"\n>>>>> Fold {i + 1}/{len(adatas_for_cv)}: Query = {query_slice_name} <<<<<")
            
            # 构建参考集
            ref_list = [ad for idx, ad in enumerate(adatas_for_cv) if idx != i]
            adata_ref_merged = concat(ref_list, join='outer', merge='same')
            adata_query = query_adata_orig.copy()
            
            # [关键] 在循环内计算 HVG (防泄露)
            logger.info(f"  计算 HVG (Top {N_HVG}, based on Reference)...")
            sc.pp.highly_variable_genes(
                adata_ref_merged,
                n_top_genes=N_HVG,
                subset=True,
                flavor="seurat",
                batch_key=BATCH_KEY
            )
            hvg_genes = adata_ref_merged.var_names
            adata_query = adata_query[:, hvg_genes].copy()
            logger.info(f"  特征对齐完成。维度: {adata_ref_merged.n_vars}")
            
            # [关键] 标签清洗 & GT Remapping (Open-Set 公平性)
            for ad_obj in [adata_ref_merged, adata_query]:
                # 清洗空格等
                ad_obj.obs[CELL_TYPE_KEY] = ad_obj.obs[CELL_TYPE_KEY].astype(str).str.strip()
            
            ref_labels = set(adata_ref_merged.obs[CELL_TYPE_KEY].unique())
            query_labels = adata_query.obs[CELL_TYPE_KEY].values
            
            # 将 Query 中 Reference 没有的标签设为 Unknown
            is_unknown = ~np.isin(query_labels, list(ref_labels))
            if is_unknown.any():
                logger.info(f"  [Open-Set] 标记 {is_unknown.sum()} 个未见类细胞为 'Unknown'")
                query_labels[is_unknown] = "Unknown"
                adata_query.obs[CELL_TYPE_KEY] = query_labels # 更新内存中的标签用于评估

            # --- 调用联合训练 ---
            fold_info = {'fold': i + 1, 'query_slice': query_slice_name}
            fold_results = run_joint_training(
                adata_ref_merged, 
                adata_query, 
                fold_info
            )
            all_cv_results.extend(fold_results)
            
        except Exception as e:
            logger.error(f"Fold {i+1} 严重错误: {e}", exc_info=True)
            
        finally:
            # 内存清理
            del adata_ref_merged, adata_query
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("  显存已清理")

    # 4. 结果汇总
    if all_cv_results:
        results_df = pd.DataFrame(all_cv_results)
        
        # 保存详细结果
        long_path = os.path.join(RESULTS_DIR, f"{DATASET_CLEAN_NAME}_scVI_Ecosystem_detailed.csv")
        results_df.to_csv(long_path, index=False)
        
        # 打印并保存摘要
        summary = results_df.groupby('method')[['f1_macro', 'accuracy', 'ari', 'nmi']].agg(['mean', 'std'])
        summary_path = os.path.join(RESULTS_DIR, f"{DATASET_CLEAN_NAME}_scVI_Ecosystem_summary.csv")
        summary.to_csv(summary_path)
        
        print("\n=== 最终结果摘要 ===")
        print(summary)
        logger.info(f"结果已保存至: {RESULTS_DIR}")
    else:
        logger.error("未生成任何有效结果。")

if __name__ == "__main__":
    # 禁用多进程加载以防止死锁
    scvi.settings.dl_num_workers = 0
    main()
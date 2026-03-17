# ==================================================================
#                  主脚本与分析流程 (Main Script) - Nature Methods 优化版
# ==================================================================

import gc
import torch

import os
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import (
    accuracy_score, f1_score, adjusted_rand_score,
    normalized_mutual_info_score, adjusted_mutual_info_score
)

# --- 导入模型函数 ---
from NetWork.model import (
    load_data, fit, load_model, annotate, _log
)

import matplotlib.pyplot as plt # 新增导入
# --- 新增：定义一个绘图函数 ---
def plot_training_history(history: dict, fold_num: int, save_dir: str):
    """
    绘制训练历史曲线（损失和AUC）并保存。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    epochs_ran = len(history.get('train_loss', []))
    epoch_axis = range(1, epochs_ran + 1)
    
    # 子图1: 损失曲线
    ax1.plot(epoch_axis, history.get('train_loss', []), label='Training Loss', marker='.')
    # 检查是否有验证损失
    if 'val_loss' in history and all(np.isfinite(history['val_loss'])):
        ax1.plot(epoch_axis, history.get('val_loss', []), label='Validation Loss', marker='.')
    ax1.set_title(f'Fold {fold_num}: Model Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 子图2: 验证集AUC曲线
    if 'val_auc' in history and all(np.isfinite(history['val_auc'])):
        ax2.plot(epoch_axis, history.get('val_auc', []), label='Validation AUC', color='green', marker='.')
        # 找到最佳AUC的点
        best_auc_val = max(history['val_auc'])
        best_auc_epoch = history['val_auc'].index(best_auc_val) + 1
        ax2.axvline(x=best_auc_epoch, color='r', linestyle='--', label=f'Best AUC: {best_auc_val:.4f} at Epoch {best_auc_epoch}')
    ax2.set_title(f'Fold {fold_num}: Validation AUC')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('AUC')
    ax2.legend()
    ax2.grid(True)
    
    fig.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, f"training_history_fold_{fold_num}.png")
    plt.savefig(save_path)
    plt.close(fig) # 关闭图像，防止在Jupyter中重复显示
    _log(f"  训练历史曲线图已保存至: {save_path}")


# ==================================================================
# 1. 用户自定义参数
# ==================================================================
DATASET_PATH = 'E:\\Project\\Work2\\Cell_datasets\\Mouse_Kidney_mod1.h5ad'
BATCH_KEY_COLUMN = "batch" 
CELL_TYPE_COLUMN = "cell_type"  
EPOCHS_PER_FOLD = 100
RANDOM_SEED = 42
UNKNOWN_LABEL_STR = "Unknown"  # 定义未知类的标准字符串

dataset_name = os.path.splitext(os.path.basename(DATASET_PATH))[0]
output_dir = os.path.join("CV_Annotation_results", dataset_name)
os.makedirs(output_dir, exist_ok=True)
_log(f"所有输出文件将被保存在: {output_dir}")

# ==================================================================
# 2. 数据加载与初始清理
# ==================================================================
_log("--- 1. 数据加载与初步预处理 ---")
tmp_adata = sc.read_h5ad(DATASET_PATH)

_log(f"原始数据形状: {tmp_adata.shape}")
tmp_adata = tmp_adata[tmp_adata.obs[CELL_TYPE_COLUMN].notna()].copy()
_log(f"过滤缺失标签后，数据形状: {tmp_adata.shape}")

# 确保提取的是原始计数 (Raw Counts)
if 'counts' in tmp_adata.layers:
    _log("发现 '.layers['counts']'，将使用它作为原始计数。")
    tmp_adata.X = tmp_adata.layers['counts'].copy()
else:
    _log("未发现 '.layers['counts']'，假设 '.X' 中为原始计数。")
    tmp_adata.X = tmp_adata.X.copy()

# 统一批次列名，确保底层模型可以无缝识别
if BATCH_KEY_COLUMN != 'batch':
    tmp_adata.obs['batch'] = tmp_adata.obs[BATCH_KEY_COLUMN]

all_batches = tmp_adata.obs[BATCH_KEY_COLUMN].unique().tolist()
cross_validation_results =[]

# --- 检查断点 ---
completed_batches = []
failed_batches =[]
for i, batch_name in enumerate(all_batches):
    safe_batch_name = str(batch_name).replace(' ', '_').replace('/', '_')
    if os.path.exists(os.path.join(output_dir, f"model_fold_{i+1}_query_{safe_batch_name}.zip")):
        completed_batches.append(batch_name)
    else:
        failed_batches.append(batch_name)

_log(f"已完成批次: {completed_batches} | 待运行批次: {failed_batches}")

# ==================================================================
# 3. 训练恢复流程 (Loop 1: Training)
# ==================================================================
if failed_batches:
    _log("\n--- 开始运行未完成的训练批次 ---")
    for i, query_batch in enumerate(all_batches):
        if query_batch not in failed_batches:
            continue 
        
        _log(f"\n>>> 训练 FOLD {i+1}/{len(all_batches)}: Query Batch = '{query_batch}'")
        
        # a. 严格划分数据
        ref_mask = tmp_adata.obs[BATCH_KEY_COLUMN] != query_batch
        ref_tmp_adata = tmp_adata[ref_mask].copy()
        
        # b. 预处理：标准化 -> 对数化 -> 找高变基因 (极其重要的标准流程)
        sc.pp.normalize_total(ref_tmp_adata, target_sum=1e4)
        sc.pp.log1p(ref_tmp_adata)
        sc.pp.highly_variable_genes(ref_tmp_adata, n_top_genes=3000, flavor='seurat')
        ref_tmp_adata = ref_tmp_adata[:, ref_tmp_adata.var.highly_variable].copy()
        
        # c. 模型训练
        load_data(ref_tmp_adata, data=ref_tmp_adata.X, labels=ref_tmp_adata.obs[CELL_TYPE_COLUMN].to_numpy())
        safe_batch_name = str(query_batch).replace(' ', '_').replace('/', '_')
        model_path_for_fold = os.path.join(output_dir, f"model_fold_{i+1}_query_{safe_batch_name}.zip")
        
        fit(
            ref_tmp_adata,
            epochs=EPOCHS_PER_FOLD,
            random_state=RANDOM_SEED,
            synth_enable=True,               # 确保开启 CutMix 以支持关系学习
            save_best_model_path=model_path_for_fold
        )
        
        # 绘图
        if 'history' in ref_tmp_adata.uns.get('cytoneighbor', {}):
            plot_training_history(ref_tmp_adata.uns['cytoneighbor']['history'], fold_num=i+1, save_dir=output_dir)

# ==================================================================
# 4. 评估流程 (Loop 2: Evaluation with Open-Set Logic)
# ==================================================================
_log("\n--- 开始评估所有交叉验证折 (LOBO Open-Set Evaluation) ---")
for i, query_batch in enumerate(all_batches):
    _log(f"\n>>> 评估 FOLD {i+1}/{len(all_batches)}: Query Batch = '{query_batch}'")
    
    # a. 数据划分
    query_mask = tmp_adata.obs[BATCH_KEY_COLUMN] == query_batch
    query_tmp_adata = tmp_adata[query_mask].copy()
    ref_mask = tmp_adata.obs[BATCH_KEY_COLUMN] != query_batch
    ref_tmp_adata = tmp_adata[ref_mask].copy()

    # b. 基础预处理 (只做 normalize & log1p，不需要 HVG！模型内部会自动对齐基因)
    for adata_obj in[ref_tmp_adata, query_tmp_adata]:
        sc.pp.normalize_total(adata_obj, target_sum=1e4)
        sc.pp.log1p(adata_obj)

    safe_batch_name = str(query_batch).replace(' ', '_').replace('/', '_')
    model_path_for_fold = os.path.join(output_dir, f"model_fold_{i+1}_query_{safe_batch_name}.zip")

    # c. 加载模型与精细化推断
    load_model(query_tmp_adata, model_path_for_fold)
    
    # 使用优化的抗批次效应参数进行预测，并开启 Unknown 发现
    # annotate(
    #     reference=ref_tmp_adata, 
    #     reference_obs_key=CELL_TYPE_COLUMN,
    #     query=query_tmp_adata,
    #     enable_unknown=True,           # 核心：允许分离未见类型
    #     unknown_label=UNKNOWN_LABEL_STR,
    #     score_mode="power",            # 使用 Power 指数放大高置信度邻居
    #     power_alpha=2.0,               # 压制跨批次的背景噪声
    #     # min_top1=0.40,                # 设置稳健的判决门槛
    #     min_top1 = 0.2 , 
    #     # max_norm_entropy=0.85          # 允许一定程度的图谱混合
    #     max_norm_entropy=0.95 
    # )

    # annotate(
    #     reference=ref_tmp_adata, 
    #     reference_obs_key=CELL_TYPE_COLUMN,
    #     query=query_tmp_adata,
    #     enable_unknown=True,           
    #     score_mode="power",            
    #     power_alpha=1.0,               # 降低 alpha，不过度惩罚远距离
    #     min_top1=0.15,                 # 降低门槛
    #     max_norm_entropy=0.95          # 容忍混乱
    # )

    annotate(
        reference=ref_tmp_adata, 
        reference_obs_key=CELL_TYPE_COLUMN,
        query=query_tmp_adata,
        enable_unknown=False,          # <--- 关键：关闭 Unknown
        score_mode="logit",            # 使用标准 Logit
        temperature=1.0
    )
    
    # d. 真实标签映射 (Ground Truth Remapping) - 这是 Open-Set 评估的命脉！
    ref_classes = set(ref_tmp_adata.obs[CELL_TYPE_COLUMN].unique())
    true_labels = query_tmp_adata.obs[CELL_TYPE_COLUMN].astype(str).copy()
    
    # 查找 Query 中存在，但 Reference 中完全没有的细胞类型
    unseen_mask = ~true_labels.isin(ref_classes)
    unseen_types = true_labels[unseen_mask].unique()
    
    if len(unseen_types) > 0:
        _log(f"  [发现零样本新类] 在 Query 中发现了 Reference 未包含的细胞类型: {unseen_types}")
        # 将这些未见过的真实标签映射为 "Unknown"，以便公平计算 F1
        true_labels[unseen_mask] = UNKNOWN_LABEL_STR
    
    # e. 提取预测标签并计算指标
    pred_labels = query_tmp_adata.obs["cytoneighbor_prediction"].astype(str)
    
    metrics = {
        "query_batch": query_batch,
        "Accuracy": accuracy_score(true_labels, pred_labels),
        "F1-score (Micro)": f1_score(true_labels, pred_labels, average='micro', zero_division=0),
        "F1-score (Macro)": f1_score(true_labels, pred_labels, average='macro', zero_division=0),
        "F1-score (Weighted)": f1_score(true_labels, pred_labels, average='weighted', zero_division=0),
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "NMI": normalized_mutual_info_score(true_labels, pred_labels),
        "AMI": adjusted_mutual_info_score(true_labels, pred_labels)
    }
    
    cross_validation_results.append(metrics)
    _log(f"  指标收集完毕: Macro-F1 = {metrics['F1-score (Macro)']:.4f} | Accuracy = {metrics['Accuracy']:.4f}")

    del ref_tmp_adata, query_tmp_adata
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==================================================================
# 5. 结果汇总与保存
# ==================================================================
_log("\n" + "="*60 + "\n--- 交叉验证完成：结果汇总 ---\n" + "="*60)

if cross_validation_results:
    df_results = pd.DataFrame(cross_validation_results)
    cols_order =["query_batch", "Accuracy", "F1-score (Micro)", "F1-score (Macro)", "F1-score (Weighted)", "AMI", "ARI", "NMI"]
    df_results = df_results[cols_order]
    
    summary_csv_path = os.path.join(output_dir, "cross_validation_metrics_summary.csv")
    df_results.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    
    print("\n【各折详细性能指标】")
    print(df_results.to_string(index=False))
    
    df_summary = df_results.drop(columns=['query_batch']).agg(['mean', 'std']).T
    print("\n【性能指标汇总统计 (均值 ± 标准差)】")
    print(df_summary.to_string())
    _log(f"汇总表格已保存至: {summary_csv_path}")
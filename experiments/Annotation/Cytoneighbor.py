# ==================================================================

#                                           CytoNeighbor Main Script (Main Script) - Nature Methods Style
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


# --- Import model functions from cytoneighbor package ---

from NetWork.model import (
    load_data, fit, load_model, annotate, _log
)


import matplotlib.pyplot as plt
# --- Visualization utilities ---

def plot_training_history(history: dict, fold_num: int, save_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    

    epochs_ran = len(history.get('train_loss', []))
    epoch_axis = range(1, epochs_ran + 1)
    

    # Plot 1: Loss Curve

    ax1.plot(epoch_axis, history.get('train_loss', []), label='Training Loss', marker='.')
    # Plot validation loss if available

    if 'val_loss' in history and all(np.isfinite(history['val_loss'])):
        ax1.plot(epoch_axis, history.get('val_loss', []), label='Validation Loss', marker='.')
    ax1.set_title(f'Fold {fold_num}: Model Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    

    # Plot 2: Validation AUC Curve

    if 'val_auc' in history and all(np.isfinite(history['val_auc'])):
        ax2.plot(epoch_axis, history.get('val_auc', []), label='Validation AUC', color='green', marker='.')
        # Mark best AUC point

        best_auc_val = max(history['val_auc'])
        best_auc_epoch = history['val_auc'].index(best_auc_val) + 1
        ax2.axvline(x=best_auc_epoch, color='r', linestyle='--', label=f'Best AUC: {best_auc_val:.4f} at Epoch {best_auc_epoch}')
    ax2.set_title(f'Fold {fold_num}: Validation AUC')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('AUC')
    ax2.legend()
    ax2.grid(True)
    

    fig.tight_layout()
    

    # Save figure

    save_path = os.path.join(save_dir, f"training_history_fold_{fold_num}.png")
    plt.savefig(save_path)
    plt.close(fig) # Prevent display in Jupyter notebooks
    _log(f"Training history plot saved to: {save_path}")




# ==================================================================

# 1. Configuration & Paths
# ==================================================================

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
# Default dataset path; override via environment variable CYTONEIGHBOR_DATA or edit below

DATASET_PATH = os.environ.get('CYTONEIGHBOR_DATA', str(BASE_DIR / 'data' / 'Mouse_Kidney_mod1.h5ad'))
BATCH_KEY_COLUMN = "batch"
CELL_TYPE_COLUMN = "cell_type"
EPOCHS_PER_FOLD = 100
RANDOM_SEED = 42
UNKNOWN_LABEL_STR = "Unknown"  # Label for unknown cells
dataset_name = os.path.splitext(os.path.basename(DATASET_PATH))[0]
output_dir = os.path.join("CV_Annotation_results", dataset_name)
os.makedirs(output_dir, exist_ok=True)
_log(f"Output directory: {output_dir}")


# ==================================================================

# 2. Load Data
# ==================================================================

_log("--- Step 1. Loading dataset ---")
tmp_adata = sc.read_h5ad(DATASET_PATH)


_log(f"Original shape: {tmp_adata.shape}")
tmp_adata = tmp_adata[tmp_adata.obs[CELL_TYPE_COLUMN].notna()].copy()
_log(f"Shape after removing NA labels: {tmp_adata.shape}")


# Ensure raw counts (Raw Counts)

if 'counts' in tmp_adata.layers:
    _log("Using counts layer as raw matrix")
    tmp_adata.X = tmp_adata.layers['counts'].copy()
else:
    _log("Using X as raw counts")
    tmp_adata.X = tmp_adata.X.copy()


# Ensure batch key is named 'batch' for CytoNeighbor compatibility
if BATCH_KEY_COLUMN != 'batch':

    tmp_adata.obs['batch'] = tmp_adata.obs[BATCH_KEY_COLUMN]


all_batches = tmp_adata.obs[BATCH_KEY_COLUMN].unique().tolist()
cross_validation_results =[]


# --- Check completed folds ---

completed_batches = []
failed_batches =[]
for i, batch_name in enumerate(all_batches):
    safe_batch_name = str(batch_name).replace(' ', '_').replace('/', '_')
    if os.path.exists(os.path.join(output_dir, f"model_fold_{i+1}_query_{safe_batch_name}.zip")):
        completed_batches.append(batch_name)
    else:
        failed_batches.append(batch_name)


_log(f"Completed: {completed_batches} | Remaining: {failed_batches}")


# ==================================================================

# 3. Training Loop (Loop 1: Training)

# ==================================================================

if failed_batches:
    _log("\n--- Starting training for remaining folds ---")
    for i, query_batch in enumerate(all_batches):
        if query_batch not in failed_batches:
            continue
        

        _log(f"\n>>> Processing FOLD {i+1}/{len(all_batches)}: Query Batch = '{query_batch}'")
        

        # a. Build reference set

        ref_mask = tmp_adata.obs[BATCH_KEY_COLUMN] != query_batch
        ref_tmp_adata = tmp_adata[ref_mask].copy()
        

        # b. Preprocess reference (normalize -> log1p -> HVG)

        sc.pp.normalize_total(ref_tmp_adata, target_sum=1e4)
        sc.pp.log1p(ref_tmp_adata)
        sc.pp.highly_variable_genes(ref_tmp_adata, n_top_genes=3000, flavor='seurat')
        ref_tmp_adata = ref_tmp_adata[:, ref_tmp_adata.var.highly_variable].copy()
        

        # c. Load data and train

        load_data(ref_tmp_adata, data=ref_tmp_adata.X, labels=ref_tmp_adata.obs[CELL_TYPE_COLUMN].to_numpy())
        safe_batch_name = str(query_batch).replace(' ', '_').replace('/', '_')
        model_path_for_fold = os.path.join(output_dir, f"model_fold_{i+1}_query_{safe_batch_name}.zip")
        

        fit(
            ref_tmp_adata,
            epochs=EPOCHS_PER_FOLD,
            random_state=RANDOM_SEED,
            synth_enable=True,               # Enable CutMix augmentation
            save_best_model_path=model_path_for_fold
        )
        

        # Plot training history

        if 'history' in ref_tmp_adata.uns.get('cytoneighbor', {}):
            plot_training_history(ref_tmp_adata.uns['cytoneighbor']['history'], fold_num=i+1, save_dir=output_dir)


# ==================================================================

# 4. Evaluation Loop (Loop 2: Evaluation with Open-Set Logic)

# ==================================================================

_log("\n--- Starting Evaluation (LOBO Open-Set Evaluation) ---")
for i, query_batch in enumerate(all_batches):
    _log(f"\n>>> Processing FOLD {i+1}/{len(all_batches)}: Query Batch = '{query_batch}'")
    

    # a. Split query and reference

    query_mask = tmp_adata.obs[BATCH_KEY_COLUMN] == query_batch
    query_tmp_adata = tmp_adata[query_mask].copy()
    ref_mask = tmp_adata.obs[BATCH_KEY_COLUMN] != query_batch
    ref_tmp_adata = tmp_adata[ref_mask].copy()


    # b. Preprocess (normalize & log1p; HVG handled internally)

    for adata_obj in[ref_tmp_adata, query_tmp_adata]:
        sc.pp.normalize_total(adata_obj, target_sum=1e4)
        sc.pp.log1p(adata_obj)


    safe_batch_name = str(query_batch).replace(' ', '_').replace('/', '_')
    model_path_for_fold = os.path.join(output_dir, f"model_fold_{i+1}_query_{safe_batch_name}.zip")


    # c. Load model and annotate

    load_model(query_tmp_adata, model_path_for_fold)
    

    # Open-set unknown detection settings

    # annotate(

    #     reference=ref_tmp_adata, 

    #     reference_obs_key=CELL_TYPE_COLUMN,

    #     query=query_tmp_adata,

    #     enable_unknown=True,           # Enable unknown detection
    #     unknown_label=UNKNOWN_LABEL_STR,

    #     score_mode="power",            # Use Power scoring

    #     power_alpha=2.0,               # Power alpha parameter

    #     # min_top1=0.40,                # Minimum top-1 confidence
    #     min_top1 = 0.2 , 

    #     # max_norm_entropy=0.85          # Maximum normalized entropy

    #     max_norm_entropy=0.95 

    # )



    # annotate(

    #     reference=ref_tmp_adata, 

    #     reference_obs_key=CELL_TYPE_COLUMN,

    #     query=query_tmp_adata,

    #     enable_unknown=True,           

    #     score_mode="power",            

    #     power_alpha=1.0,               # Alpha parameter
    #     min_top1=0.15,                 # Minimum confidence threshold

    #     max_norm_entropy=0.95          # Entropy threshold

    # )



    annotate(
        reference=ref_tmp_adata,
        reference_obs_key=CELL_TYPE_COLUMN,
        query=query_tmp_adata,
        # enable_unknown=False,          # <--- Disable unknown detection

        enable_unknown = True,
        score_mode="logit",            # Use Logit scoring
        temperature=1.0
    )
    

    # d. Ground Truth Remapping (Open-Set fairness)

    ref_classes = set(ref_tmp_adata.obs[CELL_TYPE_COLUMN].unique())
    true_labels = query_tmp_adata.obs[CELL_TYPE_COLUMN].astype(str).copy()
    

    # Identify query cell types absent from reference

    unseen_mask = ~true_labels.isin(ref_classes)
    unseen_types = true_labels[unseen_mask].unique()
    

    if len(unseen_types) > 0:
        _log(f"  [Open-Set] Query contains unseen cell types: {unseen_types}")
        # Remap to "Unknown" for fair F1 calculation

        true_labels[unseen_mask] = UNKNOWN_LABEL_STR
    

    # e. Get predictions and calculate metrics

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
    _log(f"Results: Macro-F1 = {metrics['F1-score (Macro)']:.4f} | Accuracy = {metrics['Accuracy']:.4f}")


    del ref_tmp_adata, query_tmp_adata
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==================================================================

# 5. Results Summary

# ==================================================================

_log("\n" + "="*60 + "\n--- Cross-Validation Complete ---\n" + "="*60)


if cross_validation_results:
    df_results = pd.DataFrame(cross_validation_results)
    cols_order =["query_batch", "Accuracy", "F1-score (Micro)", "F1-score (Macro)", "F1-score (Weighted)", "AMI", "ARI", "NMI"]
    df_results = df_results[cols_order]
    

    summary_csv_path = os.path.join(output_dir, "cross_validation_metrics_summary.csv")
    df_results.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    

    print("")
    print(df_results.to_string(index=False))
    

    df_summary = df_results.drop(columns=['query_batch']).agg(['mean', 'std']).T
    print("")
    print(df_summary.to_string())
    _log(f"Results saved to: {summary_csv_path}")
import os
from pathlib import Path
os.environ["SCIPY_ARRAY_API"] = "1"
os.environ["SKLEARN_ENABLE_ARRAY_API"] = "0"

import logging
import pandas as pd
import numpy as np
import anndata
import scanpy as sc
import scvi
import gc

if not hasattr(anndata, 'read'):
    anndata.read = anndata.read_h5ad

import scarches as sca
from sklearn.metrics import (
    accuracy_score, f1_score, adjusted_rand_score,
    adjusted_mutual_info_score, normalized_mutual_info_score
)
from sklearn.neighbors import KNeighborsClassifier

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
H5AD_PATH = os.environ.get("CYTONEIGHBOR_DATA", str(PROJECT_ROOT / "data" / "Zebrafish embryonic cells.h5ad"))
OUTPUT_DIR = str(PROJECT_ROOT / "results" / "CV_Perfomace")
DATASET_NAME = Path(H5AD_PATH).stem
BATCH_KEY = "batch"
CELL_TYPE_KEY = "cell_type"
LATENT_DIM = 30
N_TOP_GENES = 3000
N_NEIGHBORS = 15  # [Modification 1: Align with Cytoneighbor at 15]

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
scvi.settings.seed = 42

def main():
    adata = anndata.read_h5ad(H5AD_PATH)

    # Ensure raw counts are used
    if 'counts' in adata.layers:
        adata.X = adata.layers['counts'].copy()

    sc.pp.filter_genes(adata, min_cells=20)
    adata.layers["counts"] = adata.X.copy()

    batches = adata.obs[BATCH_KEY].unique().tolist()
    adata_list = [adata[adata.obs[BATCH_KEY] == b].copy() for b in batches]

    results_list =[]

    for i, query_adata in enumerate(adata_list):
        query_batch_name = batches[i]
        logging.info(f"Cross-validation fold [{i + 1}/{len(batches)}]: query batch = {query_batch_name}")

        ref_adatas =[a for j, a in enumerate(adata_list) if i != j]
        ref_adata = anndata.concat(ref_adatas, join="inner")

        # Preprocessing to obtain HVG
        ref_adata_norm = ref_adata.copy()
        sc.pp.normalize_total(ref_adata_norm, target_sum=1e4)
        sc.pp.log1p(ref_adata_norm)
        sc.pp.highly_variable_genes(ref_adata_norm, n_top_genes=N_TOP_GENES, batch_key=BATCH_KEY)
        hvg_list = ref_adata_norm.var[ref_adata_norm.var.highly_variable].index

        ref_adata = ref_adata[:, hvg_list].copy()
        query_adata = query_adata[:, hvg_list].copy()

        # Model training
        sca.models.SCVI.setup_anndata(ref_adata, layer="counts", batch_key=BATCH_KEY, labels_key=CELL_TYPE_KEY)
        vae_ref = sca.models.SCVI(ref_adata, n_layers=2, n_latent=LATENT_DIM)

        # [Modification 2: Remove Max_epochs limit]
        vae_ref.train(early_stopping=True)

        vae_query = sca.models.SCVI.load_query_data(query_adata, vae_ref)
        vae_query.train(plan_kwargs=dict(weight_decay=0.0), early_stopping=True)

        # Label Transfer
        ref_latent = vae_ref.get_latent_representation()
        query_latent = vae_query.get_latent_representation()

        knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
        knn.fit(ref_latent, ref_adata.obs[CELL_TYPE_KEY].astype(str))
        predicted_labels = knn.predict(query_latent)

        # [Modification 3: GT Remapping (unknown class mapping) ensures fair open-set evaluation]
        true_labels = query_adata.obs[CELL_TYPE_KEY].astype(str).copy()
        ref_classes = set(ref_adata.obs[CELL_TYPE_KEY].astype(str).unique())
        unseen_mask = ~true_labels.isin(ref_classes)

        if unseen_mask.any():
            logging.info(f"In the query, an unknown type was not found. Therefore, {unseen_mask.sum()} cells were mapped to 'Unknown'.")
            true_labels[unseen_mask] = "Unknown"

        # [Modification 4: Complete the 7 core metrics]
        all_labels_in_fold = list(set(true_labels) | set(predicted_labels))
        metrics = {
            'query_batch': query_batch_name,
            'accuracy': accuracy_score(true_labels, predicted_labels),
            'f1_micro': f1_score(true_labels, predicted_labels, labels=all_labels_in_fold, average='micro', zero_division=0),
            'f1_macro': f1_score(true_labels, predicted_labels, labels=all_labels_in_fold, average='macro', zero_division=0),
            'f1_weighted': f1_score(true_labels, predicted_labels, labels=all_labels_in_fold, average='weighted', zero_division=0),
            'ari': adjusted_rand_score(true_labels, predicted_labels),
            'ami': adjusted_mutual_info_score(true_labels, predicted_labels),
            'nmi': normalized_mutual_info_score(true_labels, predicted_labels)
        }

        results_list.append(metrics)
        gc.collect()

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_scArches_summary.csv"))
    logging.info("scArches evaluation complete.")

if __name__ == "__main__":
    main()

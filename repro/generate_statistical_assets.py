import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(r"E:/Project/Cytoneighbor_Project")
RESULTS = ROOT / "results"
OUTDIR = ROOT / "repro" / "statistics"
OUTDIR.mkdir(parents=True, exist_ok=True)


def normalize_fold(raw: object) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "mean", "std"}:
        return None
    if "_batch_" in s:
        s = s.split("_batch_", 1)[1]
    if s.startswith("fold_"):
        s = s.replace("fold_", "")
    return s.strip()


def bh_fdr(pvals: List[float]) -> List[float]:
    arr = np.asarray(pvals, dtype=float)
    n = len(arr)
    if n == 0:
        return []
    order = np.argsort(arr)
    ranked = arr[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = min(prev, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out.tolist()


def paired_stats(diff: np.ndarray) -> Dict[str, float]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = diff.size
    if n == 0:
        return {
            "n": 0,
            "effect_mean": np.nan,
            "effect_sd": np.nan,
            "effect_se": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "cohen_dz": np.nan,
        }
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    se = float(sd / math.sqrt(n)) if n > 1 else np.nan
    if n > 1 and np.isfinite(se):
        tcrit = stats.t.ppf(0.975, n - 1)
        ci_low = mean - tcrit * se
        ci_high = mean + tcrit * se
        t_stat, p_val = stats.ttest_1samp(diff, 0.0, nan_policy="omit")
        t_stat = float(t_stat)
        p_val = float(p_val)
    else:
        ci_low = np.nan
        ci_high = np.nan
        t_stat = np.nan
        p_val = np.nan
    dz = float(mean / sd) if (n > 1 and sd > 0) else np.nan
    return {
        "n": int(n),
        "effect_mean": mean,
        "effect_sd": sd,
        "effect_se": se,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "t_stat": t_stat,
        "p_value": p_val,
        "cohen_dz": dz,
    }


def load_annotation_long() -> pd.DataFrame:
    base = RESULTS / "CV_Perfomace"
    rows: List[Dict[str, object]] = []

    for ds_dir in sorted([d for d in base.iterdir() if d.is_dir()]):
        dataset = ds_dir.name

        # Cytoneighbor
        cyto = ds_dir / "cyto.csv"
        if cyto.exists():
            d = pd.read_csv(cyto)
            d = d.rename(
                columns={
                    "query_batch": "fold",
                    "Accuracy": "accuracy",
                    "F1-score (Micro)": "f1_micro",
                    "F1-score (Macro)": "f1_macro",
                    "F1-score (Weighted)": "f1_weighted",
                    "ARI": "ari",
                    "AMI": "ami",
                    "NMI": "nmi",
                }
            )
            for _, r in d.iterrows():
                fold = normalize_fold(r.get("fold"))
                if fold is None:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "method": "Cytoneighbor",
                        "accuracy": r.get("accuracy"),
                        "f1_micro": r.get("f1_micro"),
                        "f1_macro": r.get("f1_macro"),
                        "f1_weighted": r.get("f1_weighted"),
                        "ari": r.get("ari"),
                        "ami": r.get("ami"),
                        "nmi": r.get("nmi"),
                    }
                )

        # scANVI/scArches from scVI ecosystem details
        for f in ds_dir.glob("*_scVI_Ecosystem_detailed.csv"):
            d = pd.read_csv(f)
            d = d[d["method"].isin(["scANVI", "scArches"])].copy()
            for _, r in d.iterrows():
                fold = normalize_fold(r.get("query_slice"))
                if fold is None:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "method": str(r["method"]),
                        "accuracy": r.get("accuracy"),
                        "f1_micro": r.get("f1_micro"),
                        "f1_macro": r.get("f1_macro"),
                        "f1_weighted": r.get("f1_weighted"),
                        "ari": r.get("ari"),
                        "ami": r.get("ami"),
                        "nmi": r.get("nmi"),
                    }
                )

        # Files with a single baseline method
        for f in sorted(ds_dir.glob("*.csv")):
            name = f.name
            method = None
            fold_col = None
            if name.startswith("celltypist_summary_"):
                method, fold_col = "CellTypist", "fold_query_slice"
            elif name.startswith("singleR_") and name.endswith("_summary.csv"):
                method = "SingleR"
            elif name.startswith("scibetCV_results_"):
                method = "SciBet"
            elif ("seraut_benchmark_results_" in name) or (name.startswith("benchmark_results_")):
                method = "Seurat"
            if method is None:
                continue
            d = pd.read_csv(f)
            if fold_col is None:
                fold_col = d.columns[0]
            for _, r in d.iterrows():
                fold = normalize_fold(r.get(fold_col))
                if fold is None:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "method": method,
                        "accuracy": r.get("accuracy"),
                        "f1_micro": r.get("f1_micro"),
                        "f1_macro": r.get("f1_macro"),
                        "f1_weighted": r.get("f1_weighted"),
                        "ari": r.get("ari"),
                        "ami": r.get("ami"),
                        "nmi": r.get("nmi"),
                    }
                )

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["dataset", "fold", "method"]).drop_duplicates()
    return out


def build_annotation_stats(anno: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["f1_macro", "accuracy", "ari", "nmi"]
    baseline_methods = sorted(set(anno["method"]) - {"Cytoneighbor"})

    rows = []
    for dataset in sorted(anno["dataset"].unique()):
        ds = anno[anno["dataset"] == dataset]
        cy = ds[ds["method"] == "Cytoneighbor"].set_index("fold")
        for m in baseline_methods:
            bl = ds[ds["method"] == m].set_index("fold")
            common = sorted(set(cy.index) & set(bl.index))
            if not common:
                continue
            for metric in metrics:
                diff = cy.loc[common, metric].to_numpy(dtype=float) - bl.loc[common, metric].to_numpy(dtype=float)
                st = paired_stats(diff)
                rows.append(
                    {
                        "family": "Annotation",
                        "dataset": dataset,
                        "metric": metric,
                        "baseline": m,
                        "comparison": f"Cytoneighbor - {m}",
                        **st,
                    }
                )

    paired = pd.DataFrame(rows)
    paired["q_value"] = np.nan
    mask = paired["p_value"].notna()
    if mask.any():
        paired.loc[mask, "q_value"] = bh_fdr(paired.loc[mask, "p_value"].astype(float).tolist())

    # random-effects meta across datasets for Macro-F1 only
    meta_rows = []
    for m in baseline_methods:
        sub = paired[(paired["metric"] == "f1_macro") & (paired["baseline"] == m)].copy()
        if sub.empty:
            continue
        # dataset-level estimates and within-dataset variances
        y = sub["effect_mean"].to_numpy(dtype=float)
        v = np.square(sub["effect_se"].to_numpy(dtype=float))
        keep = np.isfinite(y) & np.isfinite(v) & (v > 0)
        y = y[keep]
        v = v[keep]
        k = len(y)
        if k == 0:
            continue
        w = 1.0 / v
        y_fixed = float(np.sum(w * y) / np.sum(w))
        q = float(np.sum(w * np.square(y - y_fixed)))
        c = float(np.sum(w) - (np.sum(np.square(w)) / np.sum(w)))
        tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 and c > 0 else 0.0
        w_re = 1.0 / (v + tau2)
        y_re = float(np.sum(w_re * y) / np.sum(w_re))
        se_re = float(np.sqrt(1.0 / np.sum(w_re)))
        z = y_re / se_re if se_re > 0 else np.nan
        p = float(2.0 * (1.0 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
        ci_low = float(y_re - 1.96 * se_re)
        ci_high = float(y_re + 1.96 * se_re)
        meta_rows.append(
            {
                "family": "Annotation",
                "metric": "f1_macro",
                "baseline": m,
                "comparison": f"Cytoneighbor - {m}",
                "k_datasets": int(k),
                "effect_mean_re": y_re,
                "effect_se_re": se_re,
                "ci95_low_re": ci_low,
                "ci95_high_re": ci_high,
                "tau2": tau2,
                "z_stat": z,
                "p_value": p,
            }
        )
    meta = pd.DataFrame(meta_rows)
    meta["q_value"] = np.nan
    if not meta.empty and meta["p_value"].notna().any():
        m = meta["p_value"].notna()
        meta.loc[m, "q_value"] = bh_fdr(meta.loc[m, "p_value"].astype(float).tolist())
    return paired, meta


def load_integration_long() -> pd.DataFrame:
    mapping = {
        "Human immune": RESULTS / "CV_Rebatch" / "Human immune" / "Integration_Metrics.csv",
        "Pancreas": RESULTS / "CV_Rebatch" / "Pancreas" / "Integration_Metrics (1).csv",
        "Diabetic Kidney Disease": RESULTS / "CV_Rebatch" / "Diabetic Kidney Disease" / "Integration_Metrics (2).csv",
    }
    rows = []
    for dataset, fp in mapping.items():
        if not fp.exists():
            continue
        d = pd.read_csv(fp)
        # First column is method
        method_col = d.columns[0]
        ilisi_col = [c for c in d.columns if "iLISI" in c][0]
        clisi_col = [c for c in d.columns if "cLISI" in c][0]
        nmi_col = [c for c in d.columns if c.strip().startswith("NMI")][0]
        ari_col = [c for c in d.columns if c.strip().startswith("ARI")][0]
        for _, r in d.iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "method": str(r[method_col]),
                    "iLISI": float(r[ilisi_col]),
                    "cLISI": float(r[clisi_col]),
                    "NMI": float(r[nmi_col]),
                    "ARI": float(r[ari_col]),
                }
            )
    return pd.DataFrame(rows)


def build_integration_stats(integ: pd.DataFrame) -> pd.DataFrame:
    metrics = ["iLISI", "cLISI", "NMI", "ARI"]
    baselines = sorted(set(integ["method"]) - {"Cytoneighbor"})
    rows = []
    for baseline in baselines:
        b = integ[integ["method"] == baseline].set_index("dataset")
        c = integ[integ["method"] == "Cytoneighbor"].set_index("dataset")
        common = sorted(set(b.index) & set(c.index))
        if not common:
            continue
        for metric in metrics:
            diff = c.loc[common, metric].to_numpy(dtype=float) - b.loc[common, metric].to_numpy(dtype=float)
            st = paired_stats(diff)
            rows.append(
                {
                    "family": "Integration",
                    "metric": metric,
                    "baseline": baseline,
                    "comparison": f"Cytoneighbor - {baseline}",
                    "k_datasets": len(common),
                    **st,
                }
            )
    out = pd.DataFrame(rows)
    out["q_value"] = np.nan
    if not out.empty and out["p_value"].notna().any():
        m = out["p_value"].notna()
        out.loc[m, "q_value"] = bh_fdr(out.loc[m, "p_value"].astype(float).tolist())
    return out


def build_dkd_targeted() -> pd.DataFrame:
    dkd_json = json.loads((RESULTS / "disease_discovery" / "experiment_results.json").read_text(encoding="utf-8"))
    fs = dkd_json["characterization"]["functional_scores"]
    iscore = dkd_json["characterization"]["injury_scores"]
    rows = [
        {
            "family": "DKD targeted",
            "endpoint_id": "SLC5A2_within_S1S2_DKD_gt_Healthy",
            "effect_type": "Mann-Whitney U (pre-specified directional test)",
            "effect_value": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "p_value": np.nan,
            "q_value": np.nan,
            "evidence_level": "Targeted test specified in pipeline/report; raw test output not retained in experiment_results.json.",
        },
        {
            "family": "DKD targeted",
            "endpoint_id": "FunctionalScore_cluster0_minus_cluster2",
            "effect_type": "Descriptive difference",
            "effect_value": float(fs["0"] - fs["2"]),
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "p_value": np.nan,
            "q_value": np.nan,
            "evidence_level": "Descriptive state-level summary from exported DKD characterization.",
        },
        {
            "family": "DKD targeted",
            "endpoint_id": "InjuryScore_cluster2_minus_cluster0",
            "effect_type": "Descriptive difference",
            "effect_value": float(iscore["2"] - iscore["0"]),
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "p_value": np.nan,
            "q_value": np.nan,
            "evidence_level": "Descriptive state-level summary from exported DKD characterization.",
        },
    ]
    return pd.DataFrame(rows)


def build_mechanism_stats() -> pd.DataFrame:
    fp = RESULTS / "mechanism_analysis" / "experiment_results.json"
    j = json.loads(fp.read_text(encoding="utf-8"))
    case_a = j["ablation_results"]["case_a"]["recall_scores"]
    case_b = j["ablation_results"]["case_b"]["recall_scores"]
    ks = sorted([int(k) for k in case_a.keys()])
    diff = np.array([float(case_b[str(k)]) - float(case_a[str(k)]) for k in ks], dtype=float)
    st = paired_stats(diff)
    row = {
        "family": "Mechanism",
        "endpoint_id": "RecallK_caseB_minus_caseA",
        "k_values": ",".join(str(k) for k in ks),
        **st,
    }
    return pd.DataFrame([row])


def build_external_replication_registry() -> pd.DataFrame:
    ext_fp = OUTDIR / "dkd_external_replication_effects.csv"
    if ext_fp.exists():
        ext = pd.read_csv(ext_fp)
        # Keep a stable minimal registry schema while preserving richer columns if present.
        expected = [
            "endpoint",
            "definition",
            "direction_locked",
            "status",
            "cohort_id",
            "observed_direction",
            "effect_value",
            "ci95_low",
            "ci95_high",
            "p_value",
            "q_value",
            "evidence_level",
            "notes",
        ]
        for c in expected:
            if c not in ext.columns:
                ext[c] = np.nan
        return ext[expected].copy()

    rows = [
        {
            "endpoint": "Zonation_axis_direction",
            "definition": "SLC5A1 higher in S3-like context; SLC5A2/LRP2 higher in S1/S2-like context",
            "direction_locked": "Yes",
            "status": "Pending independent external DKD cohort in repository",
        },
        {
            "endpoint": "Injury_axis_direction",
            "definition": "VCAM1/HAVCR1 increase toward injured PT state",
            "direction_locked": "Yes",
            "status": "Pending independent external DKD cohort in repository",
        },
        {
            "endpoint": "Within_state_SLC5A2_direction",
            "definition": "Within S1/S2-like PT, DKD vs Healthy directional comparison",
            "direction_locked": "Yes",
            "status": "Pending independent external DKD cohort in repository",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    anno = load_annotation_long()
    anno.to_csv(OUTDIR / "annotation_fold_metrics_long.csv", index=False)

    anno_paired, anno_meta = build_annotation_stats(anno)
    anno_paired.to_csv(OUTDIR / "annotation_paired_effects.csv", index=False)
    anno_meta.to_csv(OUTDIR / "annotation_random_effects_meta.csv", index=False)

    integ = load_integration_long()
    integ.to_csv(OUTDIR / "integration_metrics_long.csv", index=False)
    integ_stats = build_integration_stats(integ)
    integ_stats.to_csv(OUTDIR / "integration_paired_effects.csv", index=False)

    dkd = build_dkd_targeted()
    dkd.to_csv(OUTDIR / "dkd_targeted_effects.csv", index=False)

    mech = build_mechanism_stats()
    mech.to_csv(OUTDIR / "mechanism_recallk_paired_effects.csv", index=False)

    ext = build_external_replication_registry()
    ext.to_csv(OUTDIR / "dkd_external_replication_registry.csv", index=False)

    # Concise execution summary
    summary = {
        "annotation_fold_rows": int(len(anno)),
        "annotation_paired_tests": int(len(anno_paired)),
        "annotation_meta_tests": int(len(anno_meta)),
        "integration_rows": int(len(integ)),
        "integration_tests": int(len(integ_stats)),
        "dkd_targeted_rows": int(len(dkd)),
        "dkd_external_replication_rows": int(len(ext)),
        "mechanism_rows": int(len(mech)),
    }
    (OUTDIR / "statistical_assets_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

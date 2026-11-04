"""Replay revision metrics from frozen, compact analysis inputs."""
from collections import Counter, defaultdict
from math import comb, fsum, log

import numpy as np

from check_summaries import check_close, check_files


def replay_rank_endpoints(tables):
    keys = ("reference_setting", "arm", "seed")
    observed = {tuple(row[k] for k in keys): row for row in tables["gene_cutmix_rank_counts"]}
    published = tables["gene_cutmix_final_neighbours"]
    if len(observed) != len(published):
        raise ValueError("Rank counts and final endpoint tables have different coverage")
    for expected in published:
        row = observed[tuple(expected[k] for k in keys)]
        n = int(row["n_query"])
        ref = int(row["n_delta_reference"])
        for key in ("n_query", "n_reference"):
            check_close(int(row[key]), expected[key], key)
        check_close(int(row["rank_depth"]) + 1, expected["first_hit_sentinel"], "censoring")
        check_close(int(row["no_delta_hit"]) / n, expected["no_hit_fraction"], "no hit")
        check_close(int(row["predicted_delta"]) / n, expected["pred_delta_rate"], "delta calls")
        check_close(int(row["sum_first_rank_with_censoring"]) / n,
                    expected["first_hit_capped_mean"], "first hit")
        for k in (1, 5, 10, 20, 50):
            delta_hits = int(row[f"total_delta_hits_at_{k}"])
            check_close(int(row[f"query_hits_at_{k}"]) / n, expected[f"query_hit_at_{k}"], "query hit")
            check_close(delta_hits / (n * ref), expected[f"mean_reference_fraction_at_{k}"], "reference recall")
            check_close(delta_hits / (n * k), expected[f"mean_delta_fraction_at_{k}"], "neighbour fraction")
    print(f"gene-level mixing: all final-rank endpoints reproduced for {len(published)} runs.")


def replay_donor_permutations(tables):
    expected = {(row["state"], row["estimand"]): row for row in tables["dkd_state_inference"]}
    groups = defaultdict(list)
    for row in tables["dkd_donor_replay_inputs"]:
        groups[row["state"]].append(row)
    rng = np.random.default_rng(42)
    p_values = defaultdict(list)
    for state, donors in sorted(groups.items()):
        for estimand in ("mean_X", "log1p_cpm_sum"):
            healthy = np.array([float(r[estimand]) for r in donors if r["condition"] == "Healthy_Reference"])
            disease = np.array([float(r[estimand]) for r in donors if r["condition"] == "Disease_Query"])
            record = expected[state, estimand]
            check_close(len(healthy), record["n_healthy_donors"], "healthy donors")
            check_close(len(disease), record["n_disease_donors"], "DKD donors")
            delta = disease.mean() - healthy.mean()
            check_close(delta, record["delta"], "donor contrast")
            pool = np.concatenate([healthy, disease])
            hits = 0
            for _ in range(10000):
                permuted = rng.permutation(pool)
                hits += float(permuted[len(healthy):].mean() - permuted[:len(healthy)].mean()) >= delta
            p = (hits + 1) / 10001
            check_close(p, record["p_permutation"], f"{state} {estimand} permutation")
            p_values[estimand].append((p, float(record["p_permutation_BH"])))
    for family in p_values.values():
        adjusted = 1.0
        for rank, (p, q) in zip(range(len(family), 0, -1), sorted(family, reverse=True)):
            adjusted = min(adjusted, p * len(family) / rank)
            check_close(adjusted, q, "donor BH")
    print("DKD: six donor permutation tests and both BH families reproduced.")


def replay_screening(tables):
    strata = tables["unknown_fold_type_counts"]
    folds = defaultdict(Counter)
    expected_by_type = []
    for row in strata:
        n, wrong, rejected = (int(row[k]) for k in ("n", "wrong", "rejected"))
        expected_by_type.append(rejected * wrong / n)
        folds[row["fold"]].update({k: int(row[k]) for k in
                                  ("n", "wrong", "rejected", "wrong_rejected", "correct_rejected")})
    reference = {
        "within_fold": fsum(f["rejected"] * f["wrong"] / f["n"] for f in folds.values()),
        "within_fold_and_true_type": fsum(expected_by_type),
    }
    for row in tables["unknown_random_reference"]:
        check_close(sum(f["n"] for f in folds.values()), row["n_query"], "screening denominator")
        for key in ("rejected", "wrong_rejected", "correct_rejected"):
            check_close(sum(f[key] for f in folds.values()), row[f"n_{key}"], key)
        check_close(reference[row["reference"]], row["expected_wrong"], "same-workload expectation")
    print("Unknown: both same-workload references reproduced from fold-by-type counts.")


def replay_common_graph(tables):
    groups = defaultdict(list)
    for row in tables["common_graph_contingencies"]:
        groups[row["method"], row["resolution"]].append(row)
    for row in tables["common_graph_metrics"]:
        matrix = groups[row["method"], row["resolution"]]
        label_counts, cluster_counts = Counter(), Counter()
        for entry in matrix:
            n = int(entry["n"])
            label_counts[entry["cell_type"]] += n
            cluster_counts[entry["cluster"]] += n
        n = sum(label_counts.values())
        check_close(n, row["n_cells"], "graph cells")
        h_label = -fsum(v / n * log(v / n) for v in label_counts.values() if v)
        h_cluster = -fsum(v / n * log(v / n) for v in cluster_counts.values() if v)
        mutual_info = fsum(
            int(e["n"]) / n * log(int(e["n"]) * n /
                                 (label_counts[e["cell_type"]] * cluster_counts[e["cluster"]]))
            for e in matrix if int(e["n"])
        )
        pair_counts = sum(comb(int(e["n"]), 2) for e in matrix)
        by_label = sum(comb(v, 2) for v in label_counts.values())
        by_cluster = sum(comb(v, 2) for v in cluster_counts.values())
        expectation = by_label * by_cluster / comb(n, 2)
        check_close(mutual_info / ((h_label + h_cluster) / 2), row["NMI"], "common graph NMI")
        check_close((pair_counts - expectation) / ((by_label + by_cluster) / 2 - expectation),
                    row["ARI"], "common graph ARI")
    print("Integration: NMI and ARI reproduced for the four common-graph partitions.")


def replay_pbmc_resources(tables):
    metrics = ("operation_seconds", "peak_process_RSS_MiB", "peak_CUDA_allocated_MiB")
    completed = {(row["N_total_unique"], row["phase"]): row
                 for row in tables["pbmc_stage_measurements"] if row["phase_status"] == "COMPLETED"}
    for row in tables["pbmc_resource_summary"]:
        for phase in ("fit", "transform", "annotate"):
            stage = completed[row["N_total_unique"], phase]
            for key in ("n_reference_fit_input", "n_heldout_query", "n_graph_index"):
                check_close(int(stage[key]), row[key], key)
            for metric in metrics:
                check_close(float(stage[metric]), row[f"{phase}_{metric}"], f"{phase} {metric}")
    print("PBMC: all twelve completed resource stages reproduce the wide summary.")


if __name__ == "__main__":
    tables = check_files()
    replay_rank_endpoints(tables)
    replay_donor_permutations(tables)
    replay_screening(tables)
    replay_common_graph(tables)
    replay_pbmc_resources(tables)

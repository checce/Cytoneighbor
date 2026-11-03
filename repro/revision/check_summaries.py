"""Verify and summarize the bundled revision evidence using standard Python."""
import csv
from hashlib import sha256
from math import isclose
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check_close(observed, recorded, label):
    if not isclose(observed, float(recorded), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{label}: recomputed {observed}, recorded {recorded}")


def check_files():
    tables = {}
    for entry in read_rows(HERE / "manifest.csv"):
        path = HERE / entry["file"]
        if sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {path.name}")
        rows = read_rows(path)
        if len(rows) != int(entry["rows"]):
            raise ValueError(f"Row count mismatch: {path.name}")
        tables[path.stem] = rows
    return tables


def check_unknown(tables):
    folds = tables["unknown_known_type_by_fold"]
    pooled = {row["metric"]: float(row["value"])
              for row in tables["unknown_known_type_pooled"]}
    for key in ("n", "wrong", "rejected", "wrong_rejected", "correct_rejected"):
        check_close(sum(int(row[key]) for row in folds), pooled[key], key)
    n, wrong, rejected = (pooled[key] for key in ("n", "wrong", "rejected"))
    retained = n - rejected
    accepted_correct = n - wrong - pooled["correct_rejected"]
    check_close(retained / n, pooled["coverage"], "coverage")
    check_close(accepted_correct / retained, pooled["retained_accuracy"], "retained accuracy")
    check_close(accepted_correct / n, pooled["all_accuracy_rejected_count_wrong"], "all-query accuracy")
    random_counts = {
        "within_fold": sum(float(row["random_expected_wrong_same_group_review_count"]) for row in folds),
        "within_fold_and_true_type": sum(float(row["type_stratified_random_expected_wrong"]) for row in folds),
    }
    for row in tables["unknown_random_reference"]:
        expected = random_counts[row["reference"]]
        check_close(expected, row["expected_wrong"], row["reference"])
        check_close(pooled["wrong_rejected"] / expected, row["observed_expected_ratio"], row["reference"])
    print(f"Known-type screening: {int(rejected):,}/{int(n):,} deferred, "
          f"{int(pooled['wrong_rejected']):,} errors and "
          f"{int(pooled['correct_rejected']):,} correct calls.")
    zebrafish = tables["unknown_zebrafish_by_type"]
    for unsupported, label in (("True", "reference-absent"), ("False", "known-type")):
        group = [row for row in zebrafish if row["unsupported"] == unsupported]
        n = sum(int(row["n"]) for row in group)
        rejected = sum(round(int(row["n"]) * float(row["rejection_rate"])) for row in group)
        print(f"Zebrafish {label}: {rejected:,}/{n:,} deferred.")


def summarize_augmentation(rows):
    for setting in sorted({row["reference_setting"] for row in rows}):
        arm_rows = [row for row in rows if row["reference_setting"] == setting]
        keys = {(row["arm"], int(row["seed"])) for row in arm_rows}
        expected = {(arm, seed) for arm in "ABDEF" for seed in (42, 43, 44)}
        if len(arm_rows) != len(expected) or keys != expected:
            raise ValueError(f"Unexpected arm and seed combinations in {setting}")
        averages = {arm: mean(float(row["query_hit_at_1"]) for row in arm_rows if row["arm"] == arm)
                    for arm in "ABDEF"}
        print(f"{setting}, mean query-hit@1: "
              + ", ".join(f"{arm}={value:.4f}" for arm, value in averages.items()))


def check_dkd(rows, composition):
    for estimand in sorted({row["estimand"] for row in rows}):
        family = sorted((row for row in rows if row["estimand"] == estimand),
                        key=lambda row: float(row["p_permutation"]))
        if len(family) != 3:
            raise ValueError(f"Expected three states for {estimand}")
        adjusted = 1.0
        for rank in range(len(family), 0, -1):
            row = family[rank - 1]
            adjusted = min(adjusted, float(row["p_permutation"]) * len(family) / rank)
            check_close(adjusted, row["p_permutation_BH"], f"{estimand} BH")
            check_close(float(row["disease"]) - float(row["healthy"]), row["delta"], row["state"])
        print(f"DKD {estimand}: three BH-adjusted p-values verified.")
    for row in composition:
        native = float(row["disease_native"]) - float(row["healthy_native"])
        standardized = float(row["disease_standardized"]) - float(row["healthy_standardized"])
        check_close(native, row["native_delta"], row["scheme"])
        check_close(standardized, row["standardized_delta"], row["scheme"])
        check_close(native - standardized, row["composition_association_remainder"], row["scheme"])


def main():
    tables = check_files()
    print(f"Verified {len(tables)} frozen aggregate tables.")
    check_unknown(tables)
    summarize_augmentation(tables["gene_cutmix_final_neighbours"])
    check_dkd(tables["dkd_state_inference"], tables["dkd_composition_standardization"])
    resources = sorted(tables["pbmc_resource_summary"], key=lambda row: int(row["N_total_unique"]))
    for row in resources:
        size = int(row["N_total_unique"])
        if int(row["n_reference_fit_input"]) + int(row["n_heldout_query"]) != size:
            raise ValueError(f"PBMC split does not sum to {size}")
        if int(row["n_graph_index"]) != size:
            raise ValueError(f"PBMC graph size differs from {size}")
        print(f"PBMC {size:,} cells, graph: {float(row['transform_operation_seconds']):.1f} s.")
    for row in tables["native_label_transfer"]:
        print(f"Label transfer, {row['method']}: accuracy {float(row['accuracy']):.4f}, "
              f"macro-F1 {float(row['macro_f1']):.4f}.")


if __name__ == "__main__":
    main()

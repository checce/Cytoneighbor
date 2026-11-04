# Revision Evidence

The bundled CSVs contain frozen results and compact replay inputs. The
[manifest](manifest.csv) gives their source, row count and SHA-256.
Start from the repository root.

```bash
python repro/revision/check_summaries.py
python repro/revision/replay_tables.py
```

The first command uses standard Python. The second requires NumPy and
recomputes the metrics below. Install NumPy with `python -m pip install numpy`
if you only want to run the replay. Both commands work without models,
expression matrices or the original results archive.

| Topic | Inputs and outputs | What the replay verifies |
| --- | --- | --- |
| PBMC scale | [Stage measurements](tables/pbmc_stage_measurements.csv), [wide summary](tables/pbmc_resource_summary.csv) | Twelve completed stages across 25k, 50k, 80k and 100k cells. Compare like stages within the recorded seed and environment. |
| Unknown | [Fold-by-type counts](tables/unknown_fold_type_counts.csv), [pooled screening](tables/unknown_known_type_pooled.csv), [random references](tables/unknown_random_reference.csv) and [zebrafish counts](tables/unknown_zebrafish_by_type.csv) | Known-type deferrals and both same-workload random references. Zebrafish known and reference-absent totals are reported separately. The defaults are heuristic and lack independent calibration. |
| Integration and marker exposure | [Common-graph contingencies](tables/common_graph_contingencies.csv), [common-graph metrics](tables/common_graph_metrics.csv), [native clustering](tables/native_graph_clustering.csv), [native label transfer](tables/native_label_transfer.csv), [feature provenance](tables/feature_provenance.csv) | NMI and ARI for the four common-graph partitions. The native graph comparison and native label transfer use separate endpoints. The main integration marker display uses the shared automatically selected features. The original marker-retained boundary protocol, bounded feature control and verification replay remain separate. |
| Training augmentation by gene-level mixing | [Rank counts](tables/gene_cutmix_rank_counts.csv), [final-neighbour endpoints](tables/gene_cutmix_final_neighbours.csv) and [training budgets](tables/gene_cutmix_training_budgets.csv) | Five arms and three seeds in both inference pools. Computes query-hit rate, authentic-reference recall, delta-neighbour fraction and mean censored first rank. These are final neural ranks. The original graph-connectivity ranking uses a different protocol. |
| DKD | [Donor-level replay inputs](tables/dkd_donor_replay_inputs.csv), [inference](tables/dkd_state_inference.csv) and [composition contrasts](tables/dkd_composition_standardization.csv) | Repeats the six one-sided donor permutations and both three-state BH families with seed 42. The two estimands are mean log-normalized expression and full-library CPM sensitivity. Composition contrasts are exploratory. Neither expression measure reached significance after correction. |

The replay begins with the recorded analysis inputs and does not retrain
models, rebuild neighbour graphs, rederive donor values from raw counts or
rerun hardware timing. A complete training-to-results replay also requires
the frozen splits, feature order, saved diagnostics and ranks, model
checkpoints and the DKD proximal-tubule export. See the
[study inputs](../../data/) and the [original results archive](../../results/)
for the currently linked downloads.

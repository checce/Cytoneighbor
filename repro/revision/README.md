# Revision Evidence

Use this directory to inspect the main comparisons added or clarified in
the revision. The CSVs contain frozen aggregate results and retain the
source precision. The [manifest](manifest.csv) gives each source table,
row count and checksum. It also records the local path fields omitted from
the exports.

```bash
python repro/revision/check_summaries.py
```

This command checks file integrity and recomputes selected summaries from
the bundled tables. It uses the Python standard library, takes no study
data and runs no training.

| Topic | Bundled evidence | Protocol details |
| --- | --- | --- |
| PBMC scale | [Stage summaries](tables/pbmc_resource_summary.csv) | Nested pools of 25k, 50k, 80k and 100k cells from the separate Seurat v4 PBMC resource. Fit, full-pool graph construction and held-out annotation have different workloads and should be compared within phase. The resource values describe a single recorded environment and seed. |
| Unknown | [Pooled counts](tables/unknown_known_type_pooled.csv), [fold counts](tables/unknown_known_type_by_fold.csv), [random references](tables/unknown_random_reference.csv) and [zebrafish counts](tables/unknown_zebrafish_by_type.csv) | Known-type screening uses 33,506 held-out predictions. Coverage, accuracy on retained calls and deferral of correct calls have different denominators. The zebrafish analysis evaluates reference-absent types separately. The thresholds are heuristic and lack independent calibration. |
| Integration and marker exposure | [Native graph clustering](tables/native_graph_clustering.csv), [label transfer](tables/native_label_transfer.csv), [common graph subgroups](tables/common_graph_subgroup_summary.csv), [feature provenance](tables/feature_provenance.csv) and [marker replay](tables/marker_control_verification_replay.csv) | Compare graph clustering and supervised label transfer separately. The main integration marker display uses the shared automatically selected features. The original marker-retained boundary protocol, bounded feature control and verification replay are separate. The bounded control does not rerun the full default 3,000-feature pipeline. |
| Gene-CutMix | [All final-neighbour endpoints](tables/gene_cutmix_final_neighbours.csv) and [training budgets](tables/gene_cutmix_training_budgets.csv) | Five arms and three seeds use a common 1,788-cell training reference. The same models are evaluated with the capped and expanded 15,337-cell inference references. Query-hit rate, authentic-reference recall and delta-neighbour fraction use different denominators. These files describe final neural ranking. The original graph connectivity ranking is a separate protocol. |
| DKD | [Donor-level tests](tables/dkd_state_inference.csv) and [composition contrasts](tables/dkd_composition_standardization.csv) | Six healthy and five DKD donors are the statistical units. The primary mean log-normalized expression and full-library CPM sensitivity use separate three-state BH families. Composition contrasts and state-matched patterns are exploratory. |

## Replaying the revision

The tables and arithmetic checks are available from a fresh clone. An
end-to-end replay of the revision also requires frozen analysis inputs
such as the PBMC split and feature order, held-out diagnostics and ranks,
saved graph artefacts, and the DKD proximal-tubule export. The existing
[original results archive](../../results/) is the starting point for the
original workflows. A complete public revision replay bundle is currently
unavailable. The [data guide](../../data/) identifies the study inputs
and the upstream PBMC download.

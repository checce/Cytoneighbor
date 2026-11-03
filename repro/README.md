# Reproducibility Assets

Start with the workflow that matches your goal.

| Goal | Start here |
| --- | --- |
| Try Cytoneighbor without study data | Run `python tests/test_quick_run.py` or open the [tutorials](../tutorials/). |
| Inspect the revision comparisons | Open the [revision evidence](revision/). Run `python repro/revision/check_summaries.py` to check the frozen tables without models or study data. |
| Run the original experiments | Get the [study inputs](../data/) and use the commands in the [main README](../README.md#-reproducing-the-paper). The [results archive](../results/) documents the original checkpoints and figure outputs. |

`configs/config.yaml` contains the default experiment settings.
[qc_threshold_manifest.csv](qc_threshold_manifest.csv) records preprocessing
and thresholds with source anchors. [statistics/](statistics/) contains the
original statistical summaries. Their generation script reads the original
saved outputs, so populate the required result folders first.

[locks/](locks/) and [container/](container/) record the historical `smn`
environment and build recipes. The exact locks include platform-specific
packages. For everyday use, install from source in a fresh environment as
shown in the main README. R baselines also require the relevant R packages.

The historical submission snapshot is identified as `submission-v1-freeze`
in the recorded environment manifest. The revision tables have their own
[manifest](revision/manifest.csv) and checksums.
